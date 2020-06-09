import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def conv2d_block(x, n_filters, kernel_size, block, batchnorm=False):
    for i in range(2):
        conv_name = f'{block}_conv{i+1}'
        bn_name = f'{block}_bn{i+1}'
        act_name = f'{block}_act{i+1}'

        x = layers.Conv2D(n_filters, kernel_size,
                          kernel_initializer='he_normal', padding='same', name=conv_name)(x)
        if batchnorm:
            x = layers.BatchNormalization(name=bn_name)(x)
        x = layers.ReLU(name=act_name)(x)

    return x


def downsample_network(input_shape, n_classes=2, pooling='max', include_top=False):
    img = layers.Input(shape=input_shape)

    c1 = conv2d_block(img, 32, kernel_size=3, batchnorm=True, block='block1')
    x = layers.MaxPooling2D((2, 2))(c1)
    x = layers.Dropout(0.4)(x)

    c2 = conv2d_block(x, 64, kernel_size=3, batchnorm=True, block='block2')
    x = layers.MaxPooling2D((2, 2))(c2)
    x = layers.Dropout(0.4)(x)

    c3 = conv2d_block(x, 128, kernel_size=3, batchnorm=True, block='block3')
    x = layers.MaxPooling2D((2, 2))(c3)
    x = layers.Dropout(0.4)(x)

    c4 = conv2d_block(x, 256, kernel_size=3, batchnorm=True, block='block4')
    x = layers.MaxPooling2D((2, 2))(c4)
    x = layers.Dropout(0.4)(x)

    c5 = conv2d_block(x, 512, kernel_size=3, batchnorm=True, block='block5')

    if include_top:
        x = layers.Flatten()(c5)
        x = layers.Dense(n_classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(c5)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(c5)

    return keras.Model(inputs=img, outputs=x, name='encoder')


def unet(input_shape, n_classes):
    img_input = layers.Input(shape=input_shape)
    # I particularly like this setup as it makes it easy to replace encoder with any other arch without much efforts
    down_layers = [
        'block1_act2',
        'block2_act2',
        'block3_act2',
        'block4_act2',
        'block5_act2'
    ]
    base_model = downsample_network(input_shape, include_top=False)
    base_layers = [base_model.get_layer(layer).output for layer in down_layers]
    encoder = keras.Model(inputs=base_model.input, outputs=base_layers)

    # downsampling path
    x = encoder(img_input)
    return keras.Model(inputs=img_input, outputs=x)


if __name__ == "__main__":
    model = unet((256, 256, 3), 1)
    model.summary()
