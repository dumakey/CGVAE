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


#@tf.function
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

def conv2D_block(X, num_channels, f, p, s, dropout, **kwargs):
    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'

    if p != 0:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
    else:
        net = X
    net = tf.keras.layers.Conv2D(num_channels, kernel_size=f, strides=s, padding='valid',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)

    if activation == 'leakyrelu':
        rate = 0.3
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)

    return net

def inception_block(X, num_channels, f, p, s, reg):
    net = []
    for i in range(num_channels.__len__()):
        padded = tf.keras.layers.ZeroPadding2D((p[i][0], p[i][1]))(X)
        net.append(
            tf.keras.layers.Conv2D(num_channels[i],kernel_size=f[i],strides=(s[i][0],s[i][1]),padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg[0],l2=reg[1]))(padded))
    net.append(tf.keras.layers.MaxPool2D(pool_size=2, padding='same', strides=1)(X))

    net = tf.keras.layers.Concatenate(axis=-1)(net)

    return net

def get_padding(f, s, nin, nout):
    padding = []
    for i in range(f.__len__()):
        p = int(np.floor(0.5 * ((nout - 1) * s[i] + f[i] - nin)))
        nchout = int(np.floor((nin + 2 * p - f[i]) / s[i] + 1))
        if nchout != nout:
            padding.append(p + 1)
        else:
            padding.append(p)

    return padding

def inception_model(image_shape, latent_dim, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    in_shape = (image_shape[1],image_shape[0])
    input_shape = tuple(list(in_shape) + [1])
    X_input = tf.keras.layers.Input(shape=(np.prod(image_shape),))
    net = tf.keras.layers.Reshape(input_shape)(X_input)
    net = conv2D_block(net,num_channels=32,f=5,p=0,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                                    'activation':activation})
    net = tf.keras.layers.AvgPool2D(pool_size=3,strides=3)(net)
    net = conv2D_block(net,num_channels=64,f=5,p=0,s=1,dropout=dropout,kwargs={'l2_reg': l2_reg, 'l1_reg': l1_reg,
                                                                                     'activation': activation})
    net = tf.keras.layers.AvgPool2D(pool_size=3, strides=3)(net)
    net = inception_block(X=net,num_channels=[256,128,64],f=[3,5,9],p=[(8,11),(9,12),(11,14)],s=[(2,2),(2,2),(2,2)],reg=[l1_reg,l2_reg])
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=50, activation=None, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=50, activation=None, kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=2*latent_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)
    model = tf.keras.Model(inputs=X_input, outputs=net)

    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

    return model

def encoder_lenet(input_dim, latent_dim, hidden_layers, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
        
    in_shape = (input_dim[1],input_dim[0])
    input_shape = tuple(list(in_shape) + [1])
    X_input = tf.keras.layers.Input(shape=(np.prod(input_shape),))
    net = tf.keras.layers.Reshape(input_shape)(X_input)
    net = conv2D_block(net,num_channels=32,f=5,p=0,s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                               'activation':activation})
    net = tf.keras.layers.AvgPool2D(pool_size=2,strides=2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    for layer in hidden_layers:
        net = tf.keras.layers.Dense(units=layer,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation(activation)(net)
        net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=2*latent_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)

    encoder = tf.keras.Model(inputs=X_input,outputs=net,name='encoder_lenet')
    
    return encoder

def decoder_lenet(output_dim, latent_dim, hidden_layers, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    out_shape = np.prod(output_dim)
    adap_dim = int(np.sqrt(hidden_layers))
    adap_layer_shape = tuple(list((adap_dim,adap_dim)) + [1])
        
    X_input = tf.keras.layers.Input(shape=latent_dim)
    for layer in hidden_layers:
        net = tf.keras.layers.Dense(units=layer,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X_input)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation(activation)(net)
        net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=adap_dim**2,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Reshape(adap_layer_shape)(net)
    net = conv2D_block(net,num_channels=32,f=5,p=0,s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                                  'activation':activation})
    net = tf.keras.layers.AvgPool2D(pool_size=2,strides=2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=out_shape,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('sigmoid')(net)

    decoder = tf.keras.Model(inputs=X_input,outputs=net,name='decoder_lenet')
    
    return decoder
    
def encoder(input_dim, hidden_dim, latent_dim, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
    '''
    Encoder network.
    Returns the mean and the log variances of the latent distribution
    '''
        
    encoder = tf.keras.Sequential(name='encoder')
    encoder.add(tf.keras.Input(shape=(input_dim,)))
    for hidden_layer_dim in hidden_dim:
        encoder.add(tf.keras.layers.Dense(hidden_layer_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg)))
        encoder.add(tf.keras.layers.BatchNormalization())
        if activation == 'leakyrelu':
            rate = 0.3
            encoder.add(tf.keras.layers.LeakyReLU(rate))
        else:
            encoder.add(tf.keras.layers.Activation(activation))
        encoder.add(tf.keras.layers.Dropout(dropout))
    encoder.add(tf.keras.layers.Dense(2*latent_dim,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg)))

    return encoder
    
    
def decoder(latent_dim, hidden_dim, output_dim, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
    '''
    Decoder network
    It assumes that the image is a normalized black & white image so each pixel ranges between 0 and 1
    '''
    
    decoder = tf.keras.Sequential(name='decoder')
    decoder.add(tf.keras.Input(shape=(latent_dim,)))
    for hidden_layer_dim in hidden_dim:
        decoder.add(tf.keras.layers.Dense(hidden_layer_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg)))
        decoder.add(tf.keras.layers.BatchNormalization())
        if activation == 'leakyrelu':
            rate = 0.3
            decoder.add(tf.keras.layers.LeakyReLU(rate))
        else:
            decoder.add(tf.keras.layers.Activation(activation))
        decoder.add(tf.keras.layers.Dropout(dropout))
    decoder.add(tf.keras.layers.Dense(output_dim,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg),activation='sigmoid'))
    
    return decoder

def VAE(input_dim, latent_dim, encoder_hidden_layers, decoder_hidden_layers, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0,
        activation='relu', mode='train', architecture='flat'):

    in_shape_unrolled = np.prod(input_dim)

    if architecture == 'flat':
        e = encoder(in_shape_unrolled,encoder_hidden_layers,latent_dim,l2_reg,l1_reg,dropout,activation)
        d = decoder(latent_dim,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
    elif architecture == 'cnn':
        e = encoder_lenet(input_dim,latent_dim,encoder_hidden_layers,l2_reg,l1_reg,dropout,activation)
        d = decoder_lenet(input_dim,latent_dim,decoder_hidden_layers,l2_reg,l1_reg,dropout,activation)
    elif architecture == 'mixed':
        #e = inception_model(input_dim,latent_dim,alpha,l2_reg,l1_reg,dropout,activation)
        e = encoder_lenet(input_dim,latent_dim,encoder_hidden_layers,l2_reg,l1_reg,dropout,activation)
        d = decoder(latent_dim,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)

    ## DEFINE MODEL ##
    if mode == 'train':
        # Encoder
        x = tf.keras.Input(shape=(in_shape_unrolled,))
        h = e(x)
        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = x
        output = x_decoded
    elif mode == 'sample':
        # Decoder
        t = tf.keras.Input(shape=(latent_dim,))
        t_mean = tf.zeros_like(t)
        t_log_var = tf.zeros_like(t)
        if architecture == 'flat':
            d = decoder(latent_dim,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
        elif architecture == 'cnn':
            d = decoder_lenet(input_dim,latent_dim,decoder_hidden_layers,l2_reg,l1_reg,dropout,activation)
        elif architecture == 'mixed':
            d = decoder(latent_dim,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = t
        output = x_decoded
        x = x_decoded

    loss = loss_function(x,x_decoded,t_mean,t_log_var)
    model = tf.keras.Model(input,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss=lambda x,y: loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model
