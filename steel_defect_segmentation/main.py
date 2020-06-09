import os
import pandas as pd
from prepare_data import prepare_data
from dataset import tfdata_generator
from models.unet import unet

data_dir = "../data/severstal-steel-defect-detection"
img_height = 256
img_width = 1600
n_classes = 4

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(data_dir, "prep_data.csv"))
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = \
        prepare_data(data, data_dir)

    data = tfdata_generator(
        train_img_paths, train_mask_paths, is_training=True)

    model = unet((img_height, img_width, 3), n_classes)
    model.summary()
