import os
import collections
import json
import numpy as np
import pandas as pd

from document_classification.config import ml_logger

def split_data(df, shuffle, train_size, val_size, test_size):
    """Split the data into train/val/test splits."""
    # Split by category
    items = collections.defaultdict(list)
    for _, row in df.iterrows():
        items[row.y].append(row.to_dict())

    # Clean
    min_samples_per_class = 5
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
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        for item in item_list[n_train+n_val:]:
            item['split'] = 'test'

        # Add to final list
        final_list.extend(item_list)

    # df with split datasets
    split_df = pd.DataFrame(final_list)

    # Log split sizes
    ml_logger.info("==> Splits:\n{0}".format(split_df["split"].value_counts()))

    return split_df
