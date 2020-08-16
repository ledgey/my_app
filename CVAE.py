import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

    

def build_cvae(z_dim = 8,
    inter_dim = 256,
    seq_length = 128,                  
    n_features = 13):
        inputs = Input(shape=(seq_length,n_features))
        x_encoded = Conv1D(inter_dim, 3, padding='same', activation='relu')(inputs)
        x_encoded = MaxPooling1D(pool_size=2, padding='same')(x_encoded)
        x_encoded = Dropout(0.5)(x_encoded)
        x_encoded = Conv1D(inter_dim, 3, padding='same', activation='relu')(x_encoded)
        x_encoded = MaxPooling1D(pool_size=2, padding='same')(x_encoded)
        x_encoded = BatchNormalization()(x_encoded)
        x_encoded = Conv1D(inter_dim, 3, padding='same', activation='relu')(x_encoded)
        x_encoded = MaxPooling1D(pool_size=2, padding='same')(x_encoded)
        x_encoded = Dropout(0.5)(x_encoded)
        x_encoded = Flatten()(x_encoded)
        x_encoded = Dense(500, activation='relu')(x_encoded)
        x_encoded = Dropout(0.5)(x_encoded)
        x_encoded = Dense(25, activation='relu')(x_encoded)
        mu = Dense(z_dim, activation='linear')(x_encoded)
        log_var = Dense(z_dim, activation='linear')(x_encoded)
        z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])
        encoder = Model(inputs, [mu, log_var, z], name='encoder')

        latent_inputs = Input(shape=(z_dim,))
        z_decoder = Dense(z_dim, activation='relu')(latent_inputs)
        z_decoder = Dense(25, activation='relu')(z_decoder)
        z_decoder = Dense(500, activation='relu')(z_decoder)
        z_decoder = Dense(int(seq_length/8)*inter_dim, activation='relu')(z_decoder)
        z_decoder = Reshape((int(seq_length/8), inter_dim))(z_decoder)
        z_decoder = UpSampling1D(2)(z_decoder)
        z_decoder = Conv1D(inter_dim,3,padding='same', activation='relu')(z_decoder)
        z_decoder = UpSampling1D(2)(z_decoder)
        z_decoder = Conv1D(inter_dim,3,padding='same', activation='relu')(z_decoder)
        z_decoder = UpSampling1D(2)(z_decoder)
        z_decoder = Conv1D(inter_dim,3,padding='same', activation='relu')(z_decoder)
                        # No activation
        decoder_output = Dense(n_features, activation='relu')(z_decoder)

        decoder = Model(latent_inputs, decoder_output, name='decoder')
        
        outputs = decoder(encoder(inputs)[2])
        # build model
        cvae = Model(inputs, outputs)
        
        # loss
        reconstruction_loss = tf.reduce_mean(binary_crossentropy(inputs, outputs)) * (seq_length*n_features)
        kl_loss = 1 + log_var - tf.square(mu) - tf.exp(log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss

        # build model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        cvae.add_loss(total_loss)
        cvae.compile(optimizer='rmsprop')
        return encoder, decoder, cvae

