import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(data, data_dir, split_ratio=0.15):
    train_data, val_data = train_test_split(
        data, test_size=split_ratio, random_state=1)

    # Prepare train data for tensorflow datasets
    train_ids = train_data['ImageId'].values
    train_img_paths = [os.path.join(
        data_dir, "train_images", i) for i in train_ids]
    train_mask_paths = [
        [os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask1.png"),
         os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask2.png"),
         os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask3.png"),
         os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask4.png")]
        for i in train_ids]

    # Prepare val data for tensorflow datasets
    val_ids = val_data['ImageId'].values
    val_img_paths = [os.path.join(data_dir, "train_images", i)
                     for i in val_ids]
    val_mask_paths = [
        [os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask1.png"),
         os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask2.png"),
         os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask3.png"),
         os.path.join(data_dir, "train_masks", i.split(".")[0] + "_mask4.png")]
        for i in val_ids]

    return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths
