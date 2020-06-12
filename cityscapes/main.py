import os
import tensorflow as tf
import numpy as np
from IPython.display import clear_output
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from dataset import data_generator
from model import deeplabv3
from visualization import create_mask, display

tf.random.set_seed(42)
np.random.seed(1)

data_dir = "../data/cityscapes/tfrecords"

img_width = 224
img_height = 224
n_classes = 34
batch_size = 16


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


if __name__ == "__main__":
    # Prepare data for training
    train_ds = tf.data.Dataset.list_files(
        os.path.join(data_dir, "train-*.tfrecord"))
    val_ds = tf.data.Dataset.list_files(
        os.path.join(data_dir, "val-*.tfrecord"))

    train_batches = data_generator(
        train_ds, is_training=True, img_height=img_height, img_width=img_width, batch_size=batch_size)
    val_batches = data_generator(
        val_ds, is_training=False, img_height=img_height, img_width=img_width, batch_size=batch_size)

    # Visualize sample
    for image, mask in train_batches.take(1):
        sample_image, sample_mask = image[0], mask[0]
    display([sample_image, sample_mask])

    # Prepare model for training
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
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
    dc = DisplayCallback()
    callbacks = [mc, tb, dc]

    model.fit(train_batches,
              steps_per_epoch=100,
              epochs=300,
              validation_data=val_batches,
              validation_steps=30,
              callbacks=callbacks)
