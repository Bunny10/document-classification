import os
from collections import OrderedDict
import copy
import json
import numpy as np
import tensorflow as tf
import torch

def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj


def set_seeds(seed, cuda):
    """ Set Numpy and PyTorch seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def pad_seq(seq, length):
    """Pad inputs to create uniformly sized inputs."""
    vector = np.zeros(length, dtype=np.int64)
    vector[:len(seq)] = seq
    vector[len(seq):] = 0 # mask_index=0
    return vector


def collate_fn(batch):
    """Custom collat function for batch processing."""
    # Make a deep copy
    batch_copy = copy.deepcopy(batch)
    processed_batch = {"X": [], "y": []}

    # Get max sequence length
    max_seq_len = max([len(sample["X"]) for sample in batch_copy])

    # CNN filter length requirement
    max_seq_len = max(4, max_seq_len)

    # Pad
    for i, sample in enumerate(batch_copy):
        seq = sample["X"]
        y = sample["y"]
        padded_seq = pad_seq(seq, max_seq_len)
        processed_batch["X"].append(padded_seq)
        processed_batch["y"].append(y)

    # Convert to appropriate tensor types
    processed_batch["X"] = torch.LongTensor(
        processed_batch["X"])
    processed_batch["y"] = torch.LongTensor(
        processed_batch["y"])

    return processed_batch


# Credit: https://github.com/yunjey/pytorch-tutorial
class TensorboardLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


# Extended from https://github.com/nmhkahn/torchsummaryX
def model_summary(model, x, *args, **kwargs):
    model_summary_list = []

    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = "{}_{}".format(module_idx, cls_name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                info["out"] = list(outputs[0].size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params"], info["macs"] = 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement()

                if name == "weight":
                    ksize = list(param.size())
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate operations in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                    # Reverse embedding dimensions
                    if "Embedding" in cls_name:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                        info["ksize"] = ksize

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            # Generalize batch size
            info["out"][0] = None

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    with torch.no_grad():
        model(x) if not (kwargs or args) else model(x, *args, **kwargs)

    for hook in hooks:
        hook.remove()

    model_summary_list.append("-"*100)
    model_summary_list.append("{:<15} {:>20} {:>20} {:>20} {:>20}"
        .format("Layer", "Kernel Shape", "Output Shape",
                "# Params (K)", "# Operations (M)"))
    model_summary_list.append("="*100)
    input_size = list(x.size()); input_size[0] = None
    model_summary_list.append("{:<15} {:>20}".format("Input", str(input_size)))

    total_params, total_macs = 0, 0
    for layer, info in summary.items():
        repr_ksize = str(info["ksize"])
        repr_out = str(info["out"])
        repr_params = info["params"]
        repr_macs = info["macs"]

        if isinstance(repr_params, (int, float)):
            total_params += repr_params
            repr_params = "{0:,.2f}".format(repr_params/1000)
        if isinstance(repr_macs, (int, float)):
            total_macs += repr_macs
            repr_macs = "{0:,.2f}".format(repr_macs/1000000)

        model_summary_list.append("{:<15} {:>20} {:>20} {:>20} {:>20}"
            .format(layer, repr_ksize, repr_out, repr_params, repr_macs))

        # for RNN, describe inner weights (i.e. w_hh, w_ih)
        for inner_name, inner_shape in info["inner"].items():
            model_summary_list.append("  {:<13} {:>20}".format(inner_name, str(inner_shape)))

    model_summary_list.append("="*100)
    model_summary_list.append("# Params:    {0:,.2f}K".format(total_params/1000))
    model_summary_list.append("# Operations: {0:,.2f}M".format(total_macs/1000000))
    model_summary_list.append("-"*100)
    model_summary_list.append("Input:         [batch_size, ...]")
    model_summary_list.append("Linear/weight: [input_hidden_dim, output_hidden_dim]")
    model_summary_list.append("Embedding:     [num_tokens, embedding_dim]")
    model_summary_list.append("Conv:          [input_dim, output_dim (num_filters), kernel_size]")
    model_summary_list.append("-"*100)

    return model_summary_list
