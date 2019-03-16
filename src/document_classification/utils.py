import os
import sys
import collections
import copy
import json
import logging
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import time
import torch

# Logger
ml_logger = logging.getLogger("ml_logger")

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


def clean_text(text):
    """Basic text preprocessing.
    """
    text = " ".join(word.lower() for word in text.split(" "))
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-zA-Z_]+", r" ", text)
    text = text.strip()
    return text


def train_val_test_split(df, shuffle, min_samples_per_class,
                         train_size, val_size, test_size):
    """Split the data into train/val/test splits."""
    # Split by category
    items = collections.defaultdict(list)
    for _, row in df.iterrows():
        items[row.y].append(row.to_dict())

    # Clean
    by_category = {k: v for k, v in items.items() if len(v) >= min_samples_per_class}

    # Class counts
    class_counts = {}
    for category in by_category:
        class_counts[category] = len(by_category[category])

    ml_logger.info("==> Classes:\n{0}".format(
        json.dumps(class_counts, indent=4, sort_keys=True)))

    # Create split data
    final_list = []
    for _, item_list in sorted(by_category.items()):
        if shuffle:
            np.random.shuffle(item_list)
        n = len(item_list)
        n_train = int(train_size*n)
        n_val = int(val_size*n)
        n_test = int(test_size*n)

      # Give data point a split attribute
        for item in item_list[:n_train]:
            item["split"] = "train"
        for item in item_list[n_train:n_train+n_val]:
            item["split"] = "val"
        for item in item_list[n_train+n_val:]:
            item["split"] = "test"

        # Add to final list
        final_list.extend(item_list)

    # df with split datasets
    split_df = pd.DataFrame(final_list)
    train_df = split_df[split_df.split == "train"]
    val_df = split_df[split_df.split == "val"]
    test_df = split_df[split_df.split == "test"]

    return train_df, val_df, test_df


def class_weights(df, vectorizer):
    """Get class counts for imbalances."""
    class_counts = df.y.value_counts().to_dict()
    def sort_key(item):
        return vectorizer.y_vocab.lookup_token(item[0])
    sorted_counts = sorted(class_counts.items(), key=sort_key)
    frequencies = [count for _, count in sorted_counts]
    class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)
    return class_weights


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


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


# Extended from https://github.com/nmhkahn/torchsummaryX
def model_summary(model, x, *args, **kwargs):
    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = "{}_{}".format(module_idx, cls_name)

            info = collections.OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                info["out"] = list(outputs[0].size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = collections.OrderedDict()
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
    summary = collections.OrderedDict()

    model.apply(register_hook)
    with torch.no_grad():
        model(x) if not (kwargs or args) else model(x, *args, **kwargs)

    for hook in hooks:
        hook.remove()

    print ("-"*100)
    print ("{:<15} {:>20} {:>20} {:>20} {:>20}"
        .format("Layer", "Kernel Shape", "Output Shape",
                "# Params (K)", "# Operations (M)"))
    print ("="*100)
    input_size = list(x.size()); input_size[0] = None
    print ("{:<15} {:>20}".format("Input", str(input_size)))

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

        print ("{:<15} {:>20} {:>20} {:>20} {:>20}"
            .format(layer, repr_ksize, repr_out, repr_params, repr_macs))

        # for RNN, describe inner weights (i.e. w_hh, w_ih)
        for inner_name, inner_shape in info["inner"].items():
            print ("  {:<13} {:>20}".format(inner_name, str(inner_shape)))

    print ("="*100)
    print ("# Params:     {0:,.2f}K".format(total_params/1000))
    print ("# Operations: {0:,.2f}M".format(total_macs/1000000))
    print ("-"*100)
    print ("Input:         [batch_size, ...]")
    print ("Linear/weight: [input_hidden_dim, output_hidden_dim]")
    print ("Embedding:     [num_tokens, embedding_dim]")
    print ("Conv:          [input_dim, output_dim (num_filters), kernel_size]")
    print ("-"*100)

class BatchLogger(object):
    def __init__(self, train_dataset, val_dataset, batch_size):
        num_train_batches = train_dataset.get_num_batches(batch_size)
        num_val_batches = val_dataset.get_num_batches(batch_size)
        self.num_samples = len(train_dataset) + len(val_dataset)
        self.num_batches = num_train_batches + num_val_batches
        self.batch_size = batch_size
        self.progress_bar_length = 12

    def log(self, batch_index, lr, train_loss, train_acc, val_loss=0, val_acc=0, start=None):
        sys.stdout.write("\r")
        sys.stdout.write("{0}/{1} [{2:<{3}}] - {4}s - lr: {5:.2E} - loss: {6:.3f} - acc: {7:.1f}% - val_loss: {8:.3f} - val_acc: {9:.1f}%".format(
            min((batch_index+1)*self.batch_size, self.num_samples),
            self.num_samples,
            "="*int(self.progress_bar_length*(batch_index+1)/self.num_batches),
            self.progress_bar_length,
            int((time.time()-start)*(self.num_batches - (batch_index+1))),
            lr, train_loss, train_acc, val_loss, val_acc))
        sys.stdout.flush()

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

    def log(self, model, history, step):
        # Tensorboard log scalar metrics
        self.scalar_summary("lr", history["learning_rate"], step)
        for metric in ["train_loss", "val_loss"]:
            value = history[metric][-1]
            self.scalar_summary("loss/{0}".format(metric), value, step)
        for metric in ["train_accuracy", "val_accuracy"]:
            value = history[metric][-1]
            self.scalar_summary("accuracy/{0}".format(metric), value, step)

        # Tensorboard log historgram weights
        for param, value in model.named_parameters():
            param = param.replace(".", "/")
            self.histo_summary(param, value.data.cpu().numpy(), step)
            try:
                self.histo_summary(param+"/grad", value.grad.data.cpu().numpy(), step)
            except AttributeError as e:
                continue
