# This is a sample Python script.
# import module
# from Ae import Autoencoder

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Ae import Autoencoder
from keras.datasets import mnist

LEARNING_RATE = 0.0003
BATCH_SIZE = 32
EPOCHS = 50


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255

    # x_train = x_train.reshape(x_train.shape)
    x_test = x_test.astype("float32") / 255
    # x_test = x_test.reshape(x_train.shape)

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    x_train, _, _, _ = load_mnist()
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    autoencoder2 = Autoencoder.load("model")
    autoencoder2.summary()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
