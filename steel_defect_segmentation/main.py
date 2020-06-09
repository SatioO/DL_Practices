import os
import pandas as pd
from prepare_data import prepare_data
from dataset import tfdata_generator
from models.unet import unet
from metrics import dice_coef, bce_dice_loss
from callbacks import mc, tb

data_dir = "../data/severstal-steel-defect-detection"
model_file = "top_weights.h5"
img_height = 256
img_width = 1600
n_classes = 4
batch_size = 16
epochs = 50

if __name__ == "__main__":
    # Prepare data to feed to tensorflow datasets
    data = pd.read_csv(os.path.join(data_dir, "prep_data.csv"))
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = \
        prepare_data(data, data_dir)

    # Prepare data for model training
    train_batches = tfdata_generator(
        train_img_paths, train_mask_paths, is_training=True, batch_size=batch_size)
    valid_batches = tfdata_generator(
        val_img_paths, val_mask_paths, is_training=False, batch_size=batch_size)

    # define model
    model = unet((img_height, img_width, 3), n_classes)
    model.compile(optimizer="adam", loss=bce_dice_loss, metrics=[dice_coef])

    # train a model
    history = model.fit_generator(
        train_batches, validation_data=valid_batches, epochs=epochs, verbose=1, callbacks=[mc, tb])
