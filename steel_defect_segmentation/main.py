import os
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from prepare_data import prepare_data
from dataset import tfdata_generator
from models.unet import unet
from metrics import dice_coef, bce_dice_loss

data_dir = "../data/severstal-steel-defect-detection"
model_file = "top_weights.h5"
img_height = 256
img_width = 1600
n_classes = 4
batch_size = 16
epochs = 50

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(data_dir, "prep_data.csv"))
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = \
        prepare_data(data, data_dir)

    train_batches = tfdata_generator(
        train_img_paths, train_mask_paths, is_training=True, batch_size=batch_size)
    valid_batches = tfdata_generator(
        val_img_paths, val_mask_paths, is_training=False, batch_size=batch_size)

    model = unet((img_height, img_width, 3), n_classes)
    model.compile(optimizer="adam", loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()

    tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
    mc = ModelCheckpoint(mode='min', filepath=model_file,
                         monitor='val_dice_coef',
                         save_best_only='True',
                         save_weights_only='True', verbose=1)

    history = model.fit_generator(
        train_batches, validation_data=valid_batches, epochs=epochs, verbose=1, callbacks=[tb, mc])
