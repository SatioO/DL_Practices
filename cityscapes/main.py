import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from dataset import data_generator
from model import deeplabv3

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

img_width = 512
img_height = 512
n_classes = 33
batch_size = 16

if __name__ == "__main__":
    # Prepare data for training
    train_batches = data_generator(
        train_imgs, train_masks, is_training=True, img_height=img_height, img_width=img_width)
    val_batches = data_generator(
        val_imgs, val_masks, is_training=False, img_height=img_height, img_width=img_width)

    # Prepare model for training
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = deeplabv3(img_height, img_width, n_classes)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])

    tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
    mc = ModelCheckpoint(mode='min', filepath='top_weights.h5',
                         monitor='val_loss',
                         save_best_only='True',
                         save_weights_only='True', verbose=1)
    callbacks = [mc, tb]

    model.fit(train_batches,
              steps_per_epoch=len(train_imgs) // batch_size,
              epochs=300,
              validation_data=val_batches,
              validation_steps=len(val_imgs) // batch_size,
              callbacks=callbacks)
