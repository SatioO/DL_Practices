import tensorflow as tf


def data_generator(images, masks, is_training, img_height=224, img_width=224, batch_size=16, buffer_size=5000):
    def parse_fn(img_path, mask_path):
        img = tf.io.read_file(img_path)
        img = tf.cast(tf.image.decode_png(img, channels=3), tf.float32)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.clip_by_value(img, 0, 255)
        img = tf.image.per_image_standardization(img)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(tf.image.resize(
            mask, [img_height, img_width]), tf.uint8)

        return img, mask

    def flip(img, mask):
        img = tf.image.random_flip_left_right(img, seed=1)
        mask = tf.image.random_flip_left_right(mask, seed=1)

        return img, mask

    def color(img, mask):
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_saturation(img, 0.4, 1.2)
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_contrast(img, 0.4, 1.2)

        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    if is_training:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    augmentations = [flip, color]

    if is_training:
        for f in augmentations:
            if tf.random.uniform([1], 0, 1) > 0.6:
                dataset = dataset.map(
                    f, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
