import os
import numpy as np
import tensorflow as tf


def swish(x, beta=1):
    return x * tf.keras.backend.sigmoid(beta * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sampling(args):
    """Returns sample from a distribution N(args[0], diag(args[1]))
    Sampling from the distribution q(t|x) = N(t_mean, exp(t_log_var)) with reparametrization trick.

    The sample should be computed with reparametrization trick.

    The inputs are tf.Tensor
        args[0]: (batch_size x latent_dim) mean of the desired distribution
        args[1]: (batch_size x latent_dim) logarithm of the variance vector of the desired distribution

    Returns:
        A tf.Tensor of size (batch_size x latent_dim), the samples.
    """
    t_mean, t_log_var = args

    epsilon = tf.random.normal(tf.shape(t_log_var),name="epsilon")

    return t_mean + epsilon * tf.exp(t_log_var/2)

def loss_function(x, x_decoded, t_mean, t_log_var):
    """Returns the value of negative Variational Lower Bound

    The inputs are tf.Tensor
        x: (batch_size x number_of_pixels) matrix with one image per row with zeros and ones
        x_decoded: (batch_size x number_of_pixels) mean of the distribution p(x | t), real numbers from 0 to 1
        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)

    Returns:
        A tf.Tensor with one element (averaged across the batch), VLB
    """

    loss = tf.reduce_sum(x * tf.math.log(x_decoded + 1e-19) + (1 - x) * tf.math.log(1 - x_decoded + 1e-19), axis=1)
    regularisation = 0.5 * tf.reduce_sum(-t_log_var + tf.math.exp(t_log_var) + tf.math.square(t_mean) - 1, axis=1)

    return tf.reduce_mean(-loss + regularisation, axis=0)


def encoder(input_dim, hidden_dim, latent_dim, activation):
    '''
    Encoder network.
    Returns the mean and the log variances of the latent distribution
    '''
    '''
    encoder = tf.keras.Sequential(name='encoder')
    encoder.add(tf.keras.Input(shape=(input_dim,)))
    for hidden_layer_dim in hidden_dim:
        encoder.add(tf.keras.layers.Dense(hidden_layer_dim, activation=None))
        encoder.add(tf.keras.layers.BatchNormalization())
        encoder.add(tf.keras.layers.Activation(activation))
    encoder.add(tf.keras.layers.Dense(2 * latent_dim))
    '''
    X_input = tf.keras.Input(shape=(input_dim,))
    encoder = X_input
    for hidden_layer_dim in hidden_dim:
        encoder = tf.keras.layers.Dense(hidden_layer_dim, activation=None)(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation(activation)(encoder)
    encoder = tf.keras.layers.Dense(2 * latent_dim)(encoder)
    model = tf.keras.Model(inputs=X_input,outputs=encoder, name='encoder')
    return model


def decoder(latent_dim, hidden_dim, output_dim, activation):
    '''
    Decoder network
    It assumes that the image is a normalized black & white image so each pixel ranges between 0 and 1
    '''
    '''
    decoder = tf.keras.Sequential(name='decoder')
    decoder.add(tf.keras.Input(shape=(latent_dim,)))
    for hidden_layer_dim in hidden_dim:
        decoder.add(tf.keras.layers.Dense(hidden_layer_dim, activation=None))
        # decoder.add(tf.keras.layers.BatchNormalization())
        decoder.add(tf.keras.layers.Activation(activation))

    decoder.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))
    '''
    X_input = tf.keras.Input(shape=(latent_dim,))
    decoder = X_input
    for hidden_layer_dim in hidden_dim:
        decoder = tf.keras.layers.Dense(hidden_layer_dim, activation=None)(decoder)
        # decoder.add(tf.keras.layers.BatchNormalization())
        decoder = tf.keras.layers.Activation(activation)(decoder)

    decoder = tf.keras.layers.Dense(output_dim, activation='sigmoid')(decoder)
    model = tf.keras.Model(inputs=X_input, outputs=decoder, name='decoder')
    return model

def VAE(input_dim, latent_dim, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, mode='train'):

    encoder_hidden_layers = [1000]
    decoder_hidden_layers = [1000]
    act_fun = 'swish'

    ## DEFINE MODEL ##
    if mode == 'train':
        # Encoder
        x = tf.keras.Input(shape=(input_dim,))
        e = encoder(input_dim,encoder_hidden_layers,latent_dim,act_fun)
        h = e(x)
        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])
        d = decoder(latent_dim,decoder_hidden_layers,input_dim,act_fun)
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = x
        output = x_decoded
    elif mode == 'sample':
        # Decoder
        t = tf.keras.Input(shape=(latent_dim,))
        t_mean = tf.zeros_like(t)
        t_log_var = tf.zeros_like(t)
        d = decoder(latent_dim,decoder_dims,input_dim,act_fun)
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = t
        output = x_decoded
        x = x_decoded

    loss = loss_function(x,x_decoded,t_mean,t_log_var)
    model = tf.keras.Model(input,output)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),loss=lambda x,y: loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model
