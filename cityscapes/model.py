import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = layers.Lambda(lambda x: bilinear_upsample(x, size),
                      output_shape=size, name=name)(tensor)
    return y


def ASPP(img_input):
    # atrous spatial pyramid pooling
    dims = keras.backend.int_shape(img_input)

    # pool_1x1conv2d
    img_pool = layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(img_input)
    img_pool = layers.Conv2D(filters=256, kernel_size=1, padding='same',
                             kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(img_pool)
    img_pool = layers.BatchNormalization(name='bn_1')(img_pool)
    img_pool = layers.Activation('relu', name='relu_1')(img_pool)

    img_pool = Upsample(tensor=img_pool, size=[dims[1], dims[2]])

    # atrous 1
    y_1 = layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                        kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(img_input)
    y_1 = layers.BatchNormalization(name='bn_2')(y_1)
    y_1 = layers.Activation('relu', name='relu_2')(y_1)

    # atrous 6
    y_6 = layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                        kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(img_input)
    y_6 = layers.BatchNormalization(name='bn_3')(y_6)
    y_6 = layers.Activation('relu', name='relu_3')(y_6)

    # atrous 12
    y_12 = layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                         kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(img_input)
    y_12 = layers.BatchNormalization(name='bn_4')(y_12)
    y_12 = layers.Activation('relu', name='relu_4')(y_12)

    # atrous 18
    y_18 = layers.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                         kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(img_input)
    y_18 = layers.BatchNormalization(name='bn_5')(y_18)
    y_18 = layers.Activation('relu', name='relu_5')(y_18)

    # concatenate sampled layers
    y = layers.Concatenate(name='ASPP_concat')(
        [img_pool, y_1, y_6, y_12, y_18])

    y = layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                      kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = layers.BatchNormalization(name=f'bn_final')(y)
    y = layers.Activation('relu', name=f'relu_final')(y)

    return y


def deeplabv3(img_height, img_width, n_classes):
    base_model = tf.keras.applications.ResNet50(input_shape=(img_height, img_width, 3),
                                                include_top=False, weights='imagenet'
                                                )
    base_model.trainable = False

    img_features = base_model.get_layer("conv5_block3_2_relu").output
    x_a = ASPP(img_features)

    x_a = Upsample(tensor=x_a, size=[img_height // 4, img_width // 4])
    x_b = base_model.get_layer("conv2_block3_2_relu").output
    x_b = layers.Conv2D(filters=48, kernel_size=1, padding='same',
                        kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
    x_b = layers.BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = layers.Activation('relu', name='low_level_activation')(x_b)

    x = layers.Concatenate(name='decoder_concat')([x_a, x_b])

    x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                      kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
    x = layers.BatchNormalization(name=f'bn_decoder_1')(x)
    x = layers.Activation('relu', name='activation_decoder_1')(x)

    x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                      kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
    x = layers.BatchNormalization(name=f'bn_decoder_2')(x)
    x = layers.Activation('relu', name='activation_decoder_2')(x)

    x = Upsample(x, [img_height, img_width])
    x = layers.Conv2D(n_classes, (1, 1), name='output_layer')(x)

    return keras.Model(base_model.input, x, name="deeplabv3")


if __name__ == "__main__":
    model = deeplabv3(224, 224, 33)
    model.summary()
