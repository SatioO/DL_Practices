import tensorflow as tf


def tfdata_generator(images, labels, is_training, batch_size=16, buffer_size=50000):
    def parse_fn(filename, labels):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        mask_out = tf.zeros((256, 1600, 1), dtype=tf.uint8)
        for j in range(4):
            mask = tf.io.read_file(labels[j])
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.image.convert_image_dtype(mask, tf.uint8)
            mask_out = tf.concat([mask_out, mask], 2)

        return img, mask_out[:, :, 1:]

    def flip(image, labels):
        image = tf.image.random_flip_left_right(image, seed=1)
        labels = tf.image.random_flip_left_right(labels, seed=1)
        image = tf.image.random_flip_up_down(image, seed=1)
        labels = tf.image.random_flip_up_down(labels, seed=1)

        return image, labels

    def color(image, labels):
        image = tf.image.random_hue(image, 0.05)
        image = tf.image.random_saturation(image, 0.4, 1.2)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.4, 1.2)

        return image, labels

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(parse_fn, num_parallel_calls=4)

    augmentations = [flip, color]

    if is_training:
        for f in augmentations:
            if tf.random.uniform(()) > 0.5:
                dataset = dataset.map(f, num_parallel_calls=4)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
