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


class CHIVE:

    """Clockwork HIerarchial Variational autoEncoder"""

    def __init__(self,input_shape, timing_signal, latent_space_dim ):
        self.input_shape = input_shape
        self.timing_signal = timing_signal
        self.latent_space_dim = latent_space_dim
        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input = None
        self.rnn_shape = None

        self._build()

    def _build(self):
        self._build_encoder()

    def _build_encoder(self):
        """The encoder input consists of 3 parts:
            X_prosodic -concat  (X[0],X[1])
                x[0]--> concat(f0,c0) for the frame_rate RNN
                x[1]--> duration for the phone_rate RNN
            x[2]--> X_linguistic"""
        
        encoder_input = self._add_encoder_input()

        # adding frame_rate_rnn
        frame_rate_rnn = self.add_rnn_layer(encoder_input[0],timing_signal[0])

        # adding phone_rate_rnn
        phone_rate_rnn = self.add_rnn_layer(encoder_input[1],timing_signal[1])

        """syllable_rate_rnn takes inputs from frame_rate_rnn,
        phone_rate_rnn and linguistic_features as input"""
        # prosodic_feature are a concatenation of frame_rate and phone_rate_rnn
        prosodic_features = self.concat(frame_rate_rnn, phone_rate_rnn)

        # adding syllable_rate_rnn 
        syllable_rate_rnn_input = self.concat(prosodic_features,X[2])       
        syllable_rate_rnn = self.add_rnn_layer(syllable_rate_rnn_input,timing_signal[3])

        #adding bottleneck layer
        bottleneck = self._add_bottleneck(syllable_rate_rnn)

        #creating model
        self.model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def add_rnn_layer(self, encoder_input, timing_signal):
        x = encoder_input
        # we have to update shape
        #adding the clockwork RNN to 
        for i in range(len(timing_signal)):
            x = add_cwrnn_layer(timing_signal[i],x)
        return x

    def add_cwrnn_layer(clock_rate, x):
        cwrnn_layer = ClockworkRNN(periods=[1,2*clock_rate,3*clock_rate],
                                   units_per_period=2,
                                   input_shape=(self.rnn_shape),
                                   output_units=1)

        x = cwrnn_layer(x)
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
        x = Lambda(point_sample_from_normal_distro, name="encoder_output")([self.mu, self.log_variance])
        return x

    

