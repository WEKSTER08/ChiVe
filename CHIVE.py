import keras.backend
import tensorflow as tf
from keras import Model
import keras.backend as K
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import os
import pickle as pkl
from keras.datasets import mnist
from CW_RNN import ClockworkRNN


class CHIVE:

    """Clockwork HIerarchial Variational autoEncoder"""

    def __init__(self,timing_signal, latent_space_dim ):
        
        self.timing_signal = timing_signal
        self.latent_space_dim = latent_space_dim
        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input = None
        self.frnn_shape = (1,1) #shape -- (1,1) -> (f0, MFCC)
        self.phrnn_shape = (1,1,1) #shape -- (1,1,1) ->(AvgF0,AvgMFCC,Duration)
        self.sylrnn_shape = (1,1,1) #shape -- (1,1,1) -> (FRNN,PHRNN,linguistic feature)
        self.input_shape = [self.frnn_shape,self.phrnn_shape,self.sylrnn_shape]

        self._build()

    def _build(self):
        self._build_encoder()

    def _build_encoder(self):
        pass

    def _build_encoder(self):
        """The encoder input consists of 3 parts:
            X_prosodic -concat  (X[0],X[1])
                x[0]--> concat(f0,c0) for the frame_rate RNN
                x[1]--> duration for the phone_rate RNN
            x[2]--> X_linguistic"""
        frame_rate_rnn_input = Input(shape=self.frnn_shape, name="frnn_input")
        phone_rate_rnn_input = Input(shape=self.phrnn_shape, name="phrnn_input")
        syllable_rate_rnn_input = Input(shape=self.sylrnn_shape, name="sylrnn_input")
        encoder_input = [frame_rate_rnn_input,phone_rate_rnn_input,syllable_rate_rnn_input]

        ##Asyncronously feeding the framerateRNN and the Phonerate RNN's input

        # adding frame_rate_rnn
        frame_rate_rnn = self.add_rnn_layer(frame_rate_rnn_input,self.timing_signal[0],self.frnn_shape)

        # adding phone_rate_rnn
        phone_rate_rnn = self.add_rnn_layer(phone_rate_rnn_input,self.timing_signal[1],self.phrnn_shape)

        """syllable_rate_rnn takes inputs from frame_rate_rnn,
        phone_rate_rnn and linguistic_features as input"""
        # prosodic_feature are a concatenation of frame_rate and phone_rate_rnn

        merged_layer = Concatenate(name='concatenated_layer')([frame_rate_rnn, phone_rate_rnn_input, syllable_rate_rnn_input])
        # adding syllable_rate_rnn       
        syllable_rate_rnn = self.add_rnn_layer(merged_layer,self.timing_signal[2],self.sylrnn_shape)

        #adding bottleneck layer
        bottleneck = self._add_bottleneck(syllable_rate_rnn)

        #creating model
        self.model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def summary(self):
        self.encoder.summary()

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def add_rnn_layer(self, layer_input, timing_signal, shape):
        x = layer_input
        # we have to update shape
        #adding the clockwork RNN to 
        x = self.add_cwrnn_layer(timing_signal,x,shape)
        return x

    def add_cwrnn_layer(self,clock_rate, x, shape):
        cwrnn_layer = ClockworkRNN(periods=[1,2*clock_rate,3*clock_rate],
                                   units_per_period=2,
                                   input_shape=(shape),
                                   output_units=1)

        x = cwrnn_layer(x)
        return x
    def _add_bottleneck(self, x):
        lstm_units = 1  # You can adjust the number of LSTM units as needed
        x = LSTM(lstm_units, return_sequences=False, name="lstm")(x)
        return x
    def _add_bottleneck(self, x):
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
        # x = Lambda(point_sample_from_normal_distro, name="encoder_output")([self.mu, self.log_variance])
        # Replace this line:
        # x = Lambda(point_sample_from_normal_distro, name="encoder_output")([self.mu, self.log_variance])

        # With these lines:
        sampled_point = Lambda(point_sample_from_normal_distro, name="sampled_point")([self.mu, self.log_variance])
        x = Reshape(self._shape_before_bottleneck)(sampled_point)

        return x

    

if __name__ == "__main__":
    
    chive = CHIVE (
        timing_signal = (2,4,8),
        latent_space_dim = 4
    )
    chive.summary()