import tensorflow as tf

tf.random.set_seed(42)


def tfdata_generator(images, labels, is_training, batch_size=16, buffer_size=5000):
    '''Construct a data generator using tf.Dataset'''

    def parse_function(filename, labels):
        # reading image
        image_string = tf.io.read_file(filename)
        # decode image as tensor of dtype uint8
        image = tf.image.decode_jpeg(image_string, channels=3)

        # convert to float values in range [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [128, 800])  # resize to desired size

        # reading label masks
        y = tf.zeros((128, 800, 1), dtype=tf.uint8)
        for j in range(4):
            mask_string = tf.io.read_file(labels[j])
            mask = tf.image.decode_jpeg(mask_string)
            mask = tf.image.convert_image_dtype(mask, tf.uint8)

            y = tf.concat([y, mask], 2)

        return image, y[:, :, 1:]

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
        dataset = dataset.shuffle(buffer_size)  # depends on sample size

    # Transform and batch data at the same time
    dataset = dataset.map(parse_function, num_parallel_calls=4)

    augmentations = [flip, color]

    if is_training:
        for f in augmentations:
            if tf.random.uniform([1], 0, 1) > 0.6:
                dataset = dataset.map(f, num_parallel_calls=4)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
