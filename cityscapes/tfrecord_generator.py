import os
import math
import tensorflow as tf

data_dir = "../data/cityscapes/"

train_imgs = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "leftImg8bit/train", "*/*.png")))
train_masks = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "gtFine/train", "*/*_gtFine_labelIds.png")))

val_imgs = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "leftImg8bit/val", "*/*.png")))
val_masks = sorted(tf.io.gfile.glob(os.path.join(
    data_dir, "gtFine/val", "*/*_gtFine_labelIds.png")))


NUM_SHARDS = 4
OUTPUT_DIR = os.path.join(data_dir, "tfrecords")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def img_seg_to_example(img_path, mask_path, img, mask):
    feature = {
        'image/filename': _bytes_feature(img_path.encode('utf8')),
        'image/encoded': _bytes_feature(img),
        'label/filename': _bytes_feature(mask_path.encode('utf8')),
        'label/encoded': _bytes_feature(mask),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfdata_generator(images, masks, category):
    num_images = len(images)
    num_per_shard = int(math.ceil(num_images / NUM_SHARDS))

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    for shard_id in range(NUM_SHARDS):
        output_filename = os.path.join(
            OUTPUT_DIR, '%s-%05d-of-%05d.tfrecord' % (category, shard_id, NUM_SHARDS))

        print(f"Processing: {output_filename}")

        with tf.io.TFRecordWriter(output_filename) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):

                img = tf.io.gfile.GFile(images[i], 'rb').read()
                img = tf.cast(tf.image.decode_png(img, channels=3), tf.float32)
                img = tf.image.resize(img, [224, 224])

                mask = tf.io.gfile.GFile(masks[i], 'rb').read()
                mask = tf.image.decode_png(mask, channels=1)
                mask = tf.cast(tf.image.resize(mask, [224, 224]), tf.uint8)

                example = img_seg_to_example(
                    images[i], masks[i], tf.io.serialize_tensor(img), tf.io.serialize_tensor(mask))

                writer.write(example.SerializeToString())

            print(
                f"tfrecord conversion completed for {category} shared {shard_id}")


if __name__ == "__main__":
    tfdata_generator(train_imgs, train_masks, "train")
    tfdata_generator(val_imgs, val_masks, "val")
