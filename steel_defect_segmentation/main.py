import os
import pandas as pd
from prepare_data import *

data_dir = "../data/severstal-steel-defect-detection"
train_path = os.path.join(data_dir, "train_images")
val_path = os.path.join(data_dir, "train_images")


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(data_dir, "prep_data.csv"))
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = prepare_data(data,
                                                                                    train_path, val_path)
