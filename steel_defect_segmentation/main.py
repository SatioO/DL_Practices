import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from prepare_data import prepare_data
from dataset import tfdata_generator
from models.unet import unet
from convert_to_mask import *
from metrics import dice_coef, bce_dice_loss, dice_loss

data_dir = "/Users/brain/Desktop/DL/data/severstal-steel-defect-detection"
weights_file = "top_weights.h5"
img_height = 128
img_width = 800
n_classes = 4
batch_size = 8
epochs = 50

if __name__ == "__main__":
    # prepare data to feed to tensorflow datasets
    data = pd.read_csv(os.path.join(data_dir, "prep_data.csv"))
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = \
        prepare_data(data, data_dir)

    # prepare data for model training
    train_batches = tfdata_generator(
        train_img_paths, train_mask_paths, is_training=True, batch_size=batch_size)
    val_batches = tfdata_generator(
        val_img_paths, val_mask_paths, is_training=False, batch_size=batch_size)

    for k in [1, 2, 3, 4]:
        cnt = 0
        print("Sample images with Class {} defect:".format(k))
        for i in data[data[f'Defect_{k}'] != ''][['ImageId', f'Defect_{k}']].values:
            if cnt < 3:
                fig, (ax1, ax2) = plt.subplots(
                    nrows=1, ncols=2, figsize=(15, 7))
                img = cv2.imread(os.path.join(data_dir, 'train_images', str(i[0])))
                ax1.imshow(img)
                ax1.set_title(i[0])
                cnt += 1
                ax2.imshow(rle_to_mask(i[1]))
                ax2.set_title(i[0]+' mask'+str(k))
                plt.show()
    # define model
    # model = unet((img_height, img_width, 3), n_classes)
    # model.compile(optimizer="adam", loss=bce_dice_loss,
    #               metrics=[dice_coef])

    # # define callbacks
    # tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
    # mc = ModelCheckpoint(mode='min', filepath=weights_file,
    #                      monitor='val_dice_coef',
    #                      save_best_only='True',
    #                      save_weights_only='True', verbose=1)

    # train a model
    # history = model.fit_generator(
    #     train_batches, validation_data=val_batches, epochs=epochs, verbose=1)
