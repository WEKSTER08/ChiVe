import keras.backend
import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import os
import pickle as pkl
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

class CHIVE:

    """Hierarchical Variational AutoEncoder"""

    def __init__(self, latent_space_dim):
        self.latent_space_dim = latent_space_dim
        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input = None
        self.frnn_shape = (3,16)  # shape -- (1, 1) -> (f0, MFCC)
        self.phrnn_shape = (3,16)  # shape -- (1, 1, 1) -> (AvgF0, AvgMFCC, Duration)
        self.sylrnn_shape = (3,16)  # shape -- (1, 1, 1) -> (FRNN, PHRNN, linguistic feature)
        self.input_shape = [self.frnn_shape, self.phrnn_shape, self.sylrnn_shape]

        self._build()

    def _build(self):
        self._build_encoder()

    def _build_encoder(self):
        """The encoder input consists of 3 parts:
            X_prosodic -concat  (X[0],X[1])
                x[0]--> concat(f0, c0) for the frame_rate RNN
                x[1]--> duration for the phone_rate RNN
            x[2]--> X_linguistic"""
        frame_rate_rnn_input = Input(shape=self.frnn_shape, name="frnn_input")
        phone_rate_rnn_input = Input(shape=self.phrnn_shape, name="phrnn_input")
        syllable_rate_rnn_input = Input(shape=self.sylrnn_shape, name="sylrnn_input")
        encoder_input = [frame_rate_rnn_input, phone_rate_rnn_input, syllable_rate_rnn_input]

        # Asyncronously feeding the framerateRNN and the Phonerate RNN's input

        # Adding frame_rate_rnn
        frame_rate_rnn = self.add_rnn_layer(frame_rate_rnn_input, self.frnn_shape)
        # print("hi")

        # Adding phone_rate_rnn
        phone_rate_rnn = self.add_rnn_layer(phone_rate_rnn_input, self.phrnn_shape)

        """syllable_rate_rnn takes inputs from frame_rate_rnn,
        phone_rate_rnn, and linguistic_features as input"""
        # Prosodic_feature are a concatenation of frame_rate and phone_rate_rnn

        merged_layer = Concatenate(name='concatenated_layer')([frame_rate_rnn, phone_rate_rnn_input, syllable_rate_rnn_input])
        # Adding syllable_rate_rnn
        syllable_rate_rnn = self.add_rnn_layer(merged_layer, self.sylrnn_shape)

        # Adding bottleneck layer
        bottleneck = self._add_bottleneck(syllable_rate_rnn)

        # Creating model
        self.model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    # To get model summary
    def summary(self):
        self.encoder.summary()

    # compile the model
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.encoder.compile(optimizer=optimizer,
                           loss='mean_squared_error', metrics=['accuracy'])

    # train the model....
    def train(self,x_train,y_train,batch_size, num_epochs):
        frnn_train,phrnn_train,sylrnn_train = x_train
        self.encoder.fit(x=[frnn_train,phrnn_train,sylrnn_train],
                        y=y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        shuffle=True
                        )

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def add_rnn_layer(self, layer_input, shape):
        x = layer_input
        # Adding an LSTM layer
        x = LSTM(units=64, return_sequences=True)(x)
        return x
    def _add_bottleneck(self, x):
        lstm_units = 1  # You can adjust the number of LSTM units as needed
        x = LSTM(lstm_units, return_sequences=False, name="bottleneck_layer")(x)
        return x
   


if __name__ == "__main__":

    chive = CHIVE(
        latent_space_dim=1
    )
    chive.summary()

    num_samples = 1000  # Number of synthetic samples
    frnn_sequence_length = 3  # Length of the frame rate RNN sequence
    phrnn_sequence_length = 3  # Length of the phone rate RNN sequence
    sylrnn_sequence_length = 3  # Length of the syllable rate RNN sequence

    # Create synthetic data for each input feature
    frnn_data = np.random.rand(num_samples, frnn_sequence_length, 16)  # Adjust input shape as needed
    phrnn_data = np.random.rand(num_samples, phrnn_sequence_length, 16)  # Adjust input shape as needed
    sylrnn_data = np.random.rand(num_samples, sylrnn_sequence_length, 16)
    # print(phrnn_data[:10]) 

    # Create synthetic data for the timing signals
    # timing_signals = (2, 4, 8)  # Adjust timing signals as needed
    # timing_signal_data = np.array([np.ones((num_samples, sequence_length, 1)) * ts for ts, sequence_length in zip(timing_signals, [frnn_sequence_length, phrnn_sequence_length, sylrnn_sequence_length])])

    # Split the data into training and validation sets
    train_size = 0.8  # 80% for training, 20% for validation
    frnn_train, frnn_val, phrnn_train, phrnn_val, sylrnn_train, sylrnn_val = train_test_split(frnn_data, phrnn_data, sylrnn_data, test_size=1 - train_size, random_state=4)
    # Create TensorFlow datasets
    # batch_size = 32  # Adjust the batch size as needed
    # train_dataset = tf.data.Dataset.from_tensor_slices((frnn_train, phrnn_train, sylrnn_train)).shuffle(buffer_size=num_samples).batch(batch_size)
    # val_dataset = tf.data.Dataset.from_tensor_slices((frnn_val, phrnn_val, sylrnn_val)).batch(batch_size)

    train_data = list(zip(frnn_train, phrnn_train, sylrnn_train))

    # Combine the validation data arrays into a list of tuples
    val_data = list(zip(frnn_val, phrnn_val, sylrnn_val))

    # Shuffle the training data
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    

    # Split the shuffled training data back into individual arrays
    shuffled_frnn_train, shuffled_phrnn_train, shuffled_sylrnn_train = zip(*train_data)
    shuffled_frnn_val, shuffled_phrnn_val, shuffled_sylrnn_val = zip(*val_data)

    # Create batches manually
    batch_size = 16  # Adjust the batch size as needed

    # Split the shuffled training data into batches
    num_batches = len(train_data) // batch_size
    train_batches = [
        (
            shuffled_frnn_train[i * batch_size : (i + 1) * batch_size],
            shuffled_phrnn_train[i * batch_size : (i + 1) * batch_size],
            shuffled_sylrnn_train[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(num_batches)
    ]

    val_batches = [
        (
            shuffled_frnn_val[i * batch_size : (i + 1) * batch_size],
            shuffled_phrnn_val[i * batch_size : (i + 1) * batch_size],
            shuffled_sylrnn_val[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(num_batches)
    ]

    chive.compile()
 
    dummy_targets = np.random.rand(800, 1)
    # print(dummy_targets[:20],frnn_train[:20],phrnn_train[:20],sylrnn_train[:20])
    # chive.train(x_train=[frnn_train,phrnn_train,sylrnn_train],y_train=dummy_targets,batch_size=16,num_epochs=50)

    representations = chive.encoder.predict([frnn_data, phrnn_data, sylrnn_data])
    print("Latent space representations:")
    # print(representations)