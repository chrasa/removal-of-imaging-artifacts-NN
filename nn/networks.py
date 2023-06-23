import tensorflow as tf

class NetworkGenerator:
    @staticmethod
    def get_model(model_name, stride):
        if model_name == "UNet":
            return NetworkGenerator.adapted_unet(stride)
        elif model_name == "ConvNN":
            return NetworkGenerator.convolutional_network()
        elif model_name == "ResNet":
            return NetworkGenerator.residual_network(stride)
        elif model_name == "ConvAuto":
            return NetworkGenerator.convolutional_autoencoder(stride)
        else:
            raise NotImplementedError()

    @staticmethod
    def conv_with_batchnorm(inputs, n_filters, kernel_size):
        x = tf.keras.layers.Conv2D(n_filters, kernel_size, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        return x

    @staticmethod
    def residual_layer_block(inputs, n_filters, kernel_size, strides=1):
        y = NetworkGenerator.conv_with_batchnorm(inputs, n_filters, kernel_size)

        y = tf.keras.layers.Conv2D(n_filters, kernel_size, strides, padding='same')(y) 
        y = tf.keras.layers.Add()([inputs, y])
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation('relu')(y)

        return y

    @staticmethod
    def residual_network(stride):
        shape = (350, 175, 1) if stride == 5 else (344, 168, 1)
        inputs = tf.keras.layers.Input(shape=shape)

        x = tf.keras.layers.Conv2D(32, stride, strides=stride, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        for _ in range(3):
            x = NetworkGenerator.residual_layer_block(x, 32, 3, 1)

        x = tf.keras.layers.UpSampling2D(stride)(x)
        x = NetworkGenerator.conv_with_batchnorm(x, 32, stride)

        outputs = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)

        return model

    @staticmethod
    def convolutional_network():
        inputs = tf.keras.layers.Input(shape=(350, 175, 1))

        x = NetworkGenerator.conv_with_batchnorm(inputs, 8, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 16, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 16, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 16, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 16, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 16, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 16, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 8, 5)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)
        
        return model

    @staticmethod
    def convolutional_autoencoder(stride):
        shape = (350, 175, 1) if stride == 5 else (344, 168, 1)
        inputs = tf.keras.layers.Input(shape=shape) 

        x = tf.keras.layers.Conv2D(16, stride, stride, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = NetworkGenerator.conv_with_batchnorm(x, 32, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 64, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 32, 5)
        
        x = tf.keras.layers.UpSampling2D(stride)(x)
        x = NetworkGenerator.conv_with_batchnorm(x, 16, stride)

        outputs = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)

        return model

    @staticmethod
    def dual_conv_block(inputs, n_filters, kernel_size):
        x = NetworkGenerator.conv_with_batchnorm(inputs, n_filters, kernel_size)
        x = NetworkGenerator.conv_with_batchnorm(x, n_filters, kernel_size)

        return x
        
    @staticmethod
    def contracting_layers(x, n_filters, kernel_size, downsample_stride):
        f = NetworkGenerator.dual_conv_block(x, n_filters, kernel_size)
        p = tf.keras.layers.Conv2D(n_filters, downsample_stride, strides=downsample_stride, padding='same')(f)
        p = tf.keras.layers.BatchNormalization()(p)
        p = tf.keras.layers.Activation('relu')(p)

        return f, p

    @staticmethod
    def expanding_layers(x, copied_features, n_filters, kernel_size, upsample_stride):
        x = tf.keras.layers.UpSampling2D(upsample_stride)(x)
        x = NetworkGenerator.conv_with_batchnorm(x, n_filters, upsample_stride)

        x = tf.keras.layers.concatenate([x, copied_features])

        x = NetworkGenerator.dual_conv_block(x, n_filters, kernel_size)

        return x

    @staticmethod
    def adapted_unet(stride):
        shape = (350, 175, 1) if stride == 5 else (344, 168, 1)
        inputs = tf.keras.layers.Input(shape=shape)

        f1, p1 = NetworkGenerator.contracting_layers(inputs, 16, 5, stride) 

        x = NetworkGenerator.conv_with_batchnorm(p1, 32, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 64, 5)

        middle = NetworkGenerator.conv_with_batchnorm(x, 128, 5)

        x = NetworkGenerator.conv_with_batchnorm(middle, 64, 5)
        x = NetworkGenerator.conv_with_batchnorm(x, 32, 5)

        u8 = NetworkGenerator.expanding_layers(x, f1, 16, 5, stride)

        outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u8)
        
        model = tf.keras.Model(inputs, outputs)

        return model