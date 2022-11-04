"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
# keras
from tensorflow.keras.layers import GRU, RNN, Dense, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Necessary Packages
from utils import batch_generator, extract_time, random_generator

tf.compat.v1.disable_eager_execution()


def timegan(ori_data, parameters):
    """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        """Min-Max Normalizer.

        Args:
          - data: raw data

        Returns:
          - norm_data: normalized data
          - min_val: minimum values (for renormalization)
          - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Build a RNN networks
    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    show_step_info = parameters["show_train_step_info"]
    z_dim = dim
    gamma = 1

    X = Input(shape=[max_seq_len, dim], name='input_time-series_features')
    Z = Input(shape=[max_seq_len, z_dim], name='random_variables')
    T = Input(shape=[], name='input_time_information')


    # ------------- create model

    def rnn_cell(n_layers, hidden_units, output_units, name):
        return Sequential([GRU(units=hidden_units,
                           return_sequences=True,
                           name=f'GRU_{i + 1}') for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid',
                             name='OUT')], name=name)

    # autoencoder
    embedder = rnn_cell(n_layers=num_layers, 
                    hidden_units=hidden_dim,  
                    output_units=hidden_dim,
                    name='embedder')
    
    recovery = rnn_cell(n_layers=num_layers, 
                    hidden_units=hidden_dim, 
                    output_units=dim,
                    name='recovery')
    
    H = embedder(X)
    X_tilde = recovery(H)
    
    autoencoder = Model(inputs=X,
                    outputs=X_tilde,
                    name='Autoencoder')
    autoencoder.summary()
    plot_model(autoencoder,
           to_file='result/autoencoder.png',
           show_shapes=True)


    # generator and discriminator
    generator = rnn_cell(n_layers=3, 
                     hidden_units=hidden_dim, 
                     output_units=hidden_dim, 
                     name='Generator')
    discriminator = rnn_cell(n_layers=3, 
                            hidden_units=hidden_dim, 
                            output_units=1, 
                            name='Discriminator')
    supervisor = rnn_cell(n_layers=2, 
                          hidden_units=hidden_dim, 
                          output_units=hidden_dim, 
                          name='Supervisor')

    E_hat = generator(Z)
    H_hat = supervisor(E_hat)
    X_hat = recovery(H_hat)
    Y_fake = discriminator(H_hat)
    Y_real = discriminator(H)
    Y_fake_e = discriminator(E_hat)

    ## supervised
    adversarial_supervised = Model(inputs=Z,
                               outputs=Y_fake,
                               name='AdversarialNetSupervised')
    adversarial_supervised.summary()
    plot_model(adversarial_supervised, to_file='result/adversarial_supervised.png', show_shapes=True)

    # synthetic data
    synthetic_data = Model(inputs=Z,
                       outputs=X_hat,
                       name='SyntheticData')
    plot_model(synthetic_data, to_file='result/synthetic_data.png', show_shapes=True)

    ## Real
    discriminator_model = Model(inputs=X,
                            outputs=Y_real,
                            name='DiscriminatorReal')
    discriminator_model.summary()
    plot_model(discriminator_model, to_file='result/discriminator_model.png', show_shapes=True)

    ## AdversarialNet
    adversarial_embedder = Model(inputs=Z,
                    outputs=Y_fake_e,
                    name='AdversarialNet')
    adversarial_embedder.summary()
    plot_model(adversarial_embedder, to_file='result/adversarial_embedder.png', show_shapes=True)


    # ------------- training