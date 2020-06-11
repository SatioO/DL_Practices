import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../data/cityscapes/"

train_imgs = sorted(glob.glob(os.path.join(
    data_dir, "leftImg8bit/train", "*/*.png")))
train_masks = sorted(glob.glob(os.path.join(
    data_dir, "gtFine/train", "*/*_gtFine_labelIds.png")))

val_imgs = sorted(glob.glob(os.path.join(
    data_dir, "leftImg8bit/val", "*/*.png")))
val_masks = sorted(glob.glob(os.path.join(
    data_dir, "gtFine/val", "*/*_gtFine_labelIds.png")))

print('Found', len(train_imgs), 'training images')
print('Found', len(val_imgs), 'validation images\n')


def read_img(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    return img, mask


train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_masks))
train_dataset = train_dataset.map(
    read_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

if __name__ == "__main__":
    for i in train_dataset.take(10):
        print(np.unique(i[1]))
