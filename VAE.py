import keras.backend
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import os
import pickle as pkl
from keras.datasets import mnist
from CW_RNN import ClockworkRNN

tf.compat.v1.disable_eager_execution()


class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2

        self.encoder = None
        self.decoder = None
        self.model = None
        self._model_input = None
        self.reconstruction_loss_weight = 10000

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._ssmim_loss, metrics=['accuracy'])

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True
                       )

    def save(self, save_folder="."):
        self._create_folder(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,  # [28, 28, 1]
            self.conv_filters,  # [2, 4, 8]
            self.conv_kernels,  # [3, 5, 3]
            self.conv_strides,  # [1, 2, 2]
            self.latent_space_dim  # 2

        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pkl.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def reconstruct(self, images):
        latent_representation = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representation)
        return reconstructed_images, latent_representation

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _ssmim_loss(self, y_target, y_predicted):
        return 1 - tf.reduced_mean(tf.image.ssim(y_target, y_predicted, 1.0))

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        # Writing the KL Div bet ween our distro and the normal distro
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) + K.exp(self.log_variance), axis=1)
        return kl_loss

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="VAE")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # [1,2,3] ->6
        dense_layer = Dense(num_neurons, name="Decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """ we will mirror backwards from the convolutional block and stop at first block"""
        for layer_index in reversed(range(self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):

        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        cwrnn_layer = ClockworkRNN(periods=[1, 2, 4, 8],
                                   units_per_period=8,
                                   input_shape=(None, 1),
                                   output_units=1)
        x = cwrnn_layer(x)
        # x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_output_layer_{self._num_conv_layers}"
        )
        cwrnn_layer = ClockworkRNN(periods=[1, 2, 4, 8],
                                   units_per_period=8,
                                   input_shape=(28,28,1),
                                   output_units=1)
        x = cwrnn_layer(x)
        # x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        cwrnn_layer = ClockworkRNN(periods=[1, 2, 4, 8],
                                   units_per_period=8,
                                   input_shape=(28, 28, 1),
                                   output_units=1)
        x = cwrnn_layer(x)
        # x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add  a bottleneck with Gaussian sampling  (Dense layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        # The gaussian sampling for the latent layer
        def point_sample_from_normal_distro(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.0, stddev=1.0)
            sample_point = mu + K.exp(log_variance / 2) * epsilon
            return sample_point

        # The Lambda function helps us wrap a function into the graph of layers
        x = Lambda(point_sample_from_normal_distro, name="encoder_output")([self.mu, self.log_variance])
        return x


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    # x_train = x_train.reshape(x_train.shape, (1,))
    x_test = x_test.astype("float32") / 255
    # x_test = x_test.reshape(x_train.shape, (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 1, 1, 1),
        latent_space_dim=1
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size, epochs)
    return vae


if __name__ == "__main__":
    LEARNING_RATE = 0.0003
    BATCH_SIZE = 32
    EPOCHS = 100
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=1
    )
    autoencoder.summary()
    # x_train, _, _, _ = load_mnist()
    # autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # autoencoder.save("model-vae-mnist   ")
