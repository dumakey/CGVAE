# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
import numpy as np
import matplotlib.pyplot as plt
disable_eager_execution()

def swish(x, beta=1):
    return x * tf.keras.backend.sigmoid(beta * x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_datasets():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

    return (x_train, y_train), (x_test, y_test)

def vlb_binomial(x, x_decoded, t_mean, t_log_var):
    """Returns the value of negative Variational Lower Bound
    
    The inputs are tf.Tensor
        x: (batch_size x number_of_pixels) matrix with one image per row with zeros and ones
        x_decoded: (batch_size x number_of_pixels) mean of the distribution p(x | t), real numbers from 0 to 1
        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)
    
    Returns:
        A tf.Tensor with one element (averaged across the batch), VLB
    """

    loss = tf.reduce_sum(x * tf.math.log(x_decoded+1e-19) + (1-x) * tf.math.log(1-x_decoded+1e-19), axis=1)
    regularisation = 0.5 * tf.reduce_sum(-t_log_var + tf.math.exp(t_log_var) + tf.math.square(t_mean) - 1, axis=1)
    
    return  tf.reduce_mean( -loss + regularisation, axis=0)

def encoder(input_dim, hidden_dim, latent_dim, activation):
    '''
    Encoder network.
    Returns the mean and the log variances of the latent distribution
    '''    

    encoder = tf.keras.Sequential(name='encoder')
    encoder.add(tf.keras.Input(shape=(input_dim,)))
    for hidden_layer_dim in hidden_dim:
        encoder.add(tf.keras.layers.Dense(hidden_layer_dim,activation=None))
        encoder.add(tf.keras.layers.BatchNormalization())
        encoder.add(tf.keras.layers.Activation(activation))
    encoder.add(tf.keras.layers.Dense(2*latent_dim))
    
    return encoder

def decoder(latent_dim, hidden_dim, output_dim,activation):
    '''
    Decoder network
    It assumes that the image is a normalized black & white image so each pixel ranges between 0 and 1
    '''
    decoder = tf.keras.Sequential(name='decoder')
    decoder.add(tf.keras.Input(shape=(latent_dim,)))
    for hidden_layer_dim in hidden_dim:
        decoder.add(tf.keras.layers.Dense(hidden_layer_dim,activation=None))
        #decoder.add(tf.keras.layers.BatchNormalization())
        decoder.add(tf.keras.layers.Activation(activation))

    decoder.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))
    
    return decoder
    
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

    epsilon = tf.random.normal(tf.shape(t_log_var), name="epsilon")
    
    return t_mean + epsilon * tf.exp(t_log_var/2)

def model(input_dim, latent_dim, encoder_dims, decoder_dims, act_fun, mode):

    ## DEFINE MODEL ##
    if mode == 'train':
        # Encoder
        x = tf.keras.Input(shape=(input_dim,))
        e = encoder(input_dim,encoder_dims,latent_dim,act_fun)
        h = e(x)
        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])
        d = decoder(latent_dim,decoder_dims,input_dim,act_fun)
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

    loss = vlb_binomial(x,x_decoded,t_mean,t_log_var)
    model = tf.keras.Model(input,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.008), loss=lambda x, y: loss)

    return model

def conditional_model(n_labels, input_dim, latent_dim, encoder_dims, decoder_dims, act_fun, mode):

    ## DEFINE MODEL ##
    if mode == 'train':
        # Encoder
        x = tf.keras.Input(shape=(input_dim,))
        label = tf.keras.Input(shape=(n_labels,))
        e = encoder(input_dim+n_labels,encoder_dims,latent_dim,act_fun)
        h = e(tf.keras.layers.concatenate([x,label]))

        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])
        d = decoder(latent_dim+n_labels,decoder_dims,input_dim,act_fun)
        x_decoded = d(tf.keras.layers.concatenate([t,label]))

        # Declare inputs/outputs for the model
        input = [x,label]
        output = x_decoded
    elif mode == 'sample':
        t = tf.keras.Input(shape=(latent_dim,))
        label = tf.keras.Input(shape=(n_labels,))
        t_mean = tf.zeros_like(t)
        t_log_var = tf.zeros_like(t)
        d = decoder(latent_dim+n_labels,decoder_dims,input_dim,act_fun)
        x_decoded = d(tf.keras.layers.concatenate([t,label]))

        # Declare inputs/outputs for the model
        input = [t,label]
        output = x_decoded
        x = x_decoded

    loss = vlb_binomial(x,x_decoded,t_mean,t_log_var)
    conditional_model = tf.keras.Model(input,output)
    conditional_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.008),loss=lambda x, y: loss)

    return conditional_model

def generate_conditional_samples(label, cond_vae, latent_dim, decoder_dims, act_fun):

    ## BUILD DECODER ##
    n_samples, n_labels = label.shape
    output_dim = cond_vae.input[0].shape[1]
    decoder = conditional_model(n_labels,output_dim,latent_dim,None,decoder_dims,act_fun,mode='sample')
    # Retrieve decoder weights
    model_weights = cond_vae.weights
    j = 0
    for weight in model_weights:
        j_layer_shape = weight.get_shape()[0]
        if j_layer_shape != (latent_dim+n_labels):
            j += 1
        else:
            break
    decoder_input_layer_idx = j

    decoder_weights = cond_vae.get_weights()[decoder_input_layer_idx:]
    decoder.set_weights(decoder_weights)

    ## SAMPLE IMAGES ##
    t_means_std = tf.zeros((1,latent_dim))
    t_log_var_std = tf.zeros((1,latent_dim))
    y_label = tf.convert_to_tensor(label,dtype=np.float32)

    X_cond_samples = np.zeros([n_samples,output_dim])
    for i in range(n_samples):
        t = tf.keras.layers.Lambda(sampling)([t_means_std, t_log_var_std])
        label_oh = tf.reshape(y_label[i],(1,n_labels))
        X_cond_samples[i,:] = decoder.predict([t,label_oh],steps=1)

    return X_cond_samples

def generate_samples(vae, latent_dim, decoder_dims, n_samples, act_fun):

    ## BUILD DECODER ##
    output_dim = vae.input.shape[1]
    decoder = model(output_dim,latent_dim,None,decoder_dims,act_fun,mode='sample')
    # Retrieve decoder weights
    model_weights = vae.weights
    j = 0
    for weight in model_weights:
        j_layer_shape = weight.get_shape()[0]
        if j_layer_shape != latent_dim:
            j += 1
        else:
            break
    decoder_input_layer_idx = j

    decoder_weights = vae.get_weights()[decoder_input_layer_idx:]
    decoder.set_weights(decoder_weights)

    ## SAMPLE IMAGES ##
    t = tf.random.normal(shape=(1,latent_dim))
    X_samples = np.zeros([n_samples,output_dim])
    for i in range(n_samples):
        X_samples[i,:] = decoder.predict(t,steps=1)

    return X_samples

if __name__ == "__main__":
    ## PARAMETERS ##
    batch_size = 100
    input_dim = 784
    latent_dim = 25
    encoder_hidden_layers = [1000]
    decoder_hidden_layers = [1000]
    act_fun = 'swish'
    epochs = 20
    n_samples = 10


    ## TRAIN ##
    (x_train, y_train), (x_test, y_test) = get_datasets()
    img_size = int(np.sqrt(x_train.shape[1]))
    VAE = model(input_dim,latent_dim,encoder_hidden_layers,decoder_hidden_layers,act_fun,mode='train')
    history = VAE.fit(x=x_train,y=x_train,shuffle=True,epochs=epochs,batch_size=batch_size,
                           validation_data=(x_test,x_test),verbose=1)


    '''
    ## PLOT GENERATED TRAINING DATA ## 
    fig = plt.figure(figsize=(10, 10))
    for fid_idx, (data, title) in enumerate(zip([x_train, x_test], ['Train', 'Validation'])):
        n = 10  # figure with 10 x 2 digits
        figure = np.zeros((img_size*n,img_size*2))
        for i in range(n_samples):
            x_dec = VAE.predict(data[i,:].reshape((1,img_size**2)))
            x_dec = np.reshape(x_dec,(img_size,img_size))
            figure[i*img_size:(i+1)*img_size,:img_size] = data[i,:].reshape(img_size,img_size)
            figure[i*img_size:(i+1)*img_size,img_size:] = x_dec
        ax = fig.add_subplot(1,2,fid_idx + 1)
        ax.imshow(figure, cmap='Greys_r')
        ax.set_title(title)
        ax.axis('off')
    plt.show()
    '''
    '''
    ## GENERATE NEW DATA - SAMPLING ##
    X_samples = generate_samples(VAE,latent_dim,decoder_hidden_layers,n_samples,act_fun)
    print()
    
    # Show the sampled images.
    plt.figure()
    for i in range(n_samples):
        ax = plt.subplot(n_samples//5 + 1,5,i+1)
        plt.imshow(X_samples[i,:].reshape(img_size,img_size), cmap='gray')
        ax.axis('off')
    plt.show()
    print()
    '''

    ## CONDITIONAL VAE##
    (x_train, y_train), (x_test, y_test) = get_datasets()
    n_labels = y_train.shape[1]
    img_size = int(np.sqrt(x_train.shape[1]))
    cond_VAE = conditional_model(n_labels,input_dim,latent_dim,encoder_hidden_layers,decoder_hidden_layers,act_fun,mode='train')
    history = cond_VAE.fit(x=[x_train,y_train],y=x_train,shuffle=True,epochs=epochs,batch_size=batch_size,
                           validation_data=([x_test,y_test],x_test),verbose=2)
    '''
    fig = plt.figure(figsize=(10, 10))
    for fid_idx, (x_data, y_data, title) in enumerate(zip([x_train,x_test], [y_train,y_test], ['Train', 'Validation'])):
        n = 10  # figure with 10 x 2 digits
        figure = np.zeros((img_size*n,img_size*2))
        for i in range(n_samples):
            x_dec = VAE.predict([[x_data[i,:]],[y_data[i,:]]])
            x_dec = np.reshape(x_dec,(img_size,img_size))
            figure[i*img_size:(i+1)*img_size,:img_size] = x_data[i,:].reshape(img_size,img_size)
            figure[i*img_size:(i+1)*img_size,img_size:] = x_dec
        ax = fig.add_subplot(1,2,fid_idx + 1)
        ax.imshow(figure, cmap='Greys_r')
        ax.set_title(title)
        ax.axis('off')
    plt.show()
    '''
    labels = np.array([1,0,3])
    n_samples = labels.size
    labels_oh = np.zeros((n_samples,10))
    labels_oh[np.arange(n_samples),labels] = 1
    X_cond_samples = generate_conditional_samples(labels_oh,cond_VAE,latent_dim,decoder_hidden_layers,act_fun)

    # Show the sampled images.
    plt.figure()
    for i in range(n_samples):
        ax = plt.subplot(n_samples,1,i+1)
        plt.imshow(X_cond_samples[i,:].reshape(img_size,img_size), cmap='gray')
        ax.axis('off')
    plt.show()
    print()