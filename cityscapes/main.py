import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import data_generator

tf.random.set_seed(42)

data_dir = "../data/cityscapes/"

train_imgs = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "leftImg8bit/train", "*/*.png")))
train_masks = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "gtFine/train", "*/*_gtFine_labelIds.png")))

val_imgs = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "leftImg8bit/val", "*/*.png")))
val_masks = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "gtFine/val", "*/*_gtFine_labelIds.png")))

print('Found', len(train_imgs), 'training images')
print('Found', len(val_imgs), 'validation images\n')

img_width = 224
img_height = 224

if __name__ == "__main__":
    # Prepare data for training
    train_batches = data_generator(
        train_imgs, train_masks, is_training=True, img_height=img_height, img_width=img_width)
    val_batches = data_generator(
        val_imgs, val_masks, is_training=False, img_height=img_height, img_width=img_width)

    for img in train_batches.take(4):
        fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].imshow(img[0][1])
        ax[1].imshow(np.reshape(img[1][1], (img_height, img_width)))
