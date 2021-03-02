import numpy as np
import netCDF4 as nc
import random
import keras
import keras.layers as layers
from keras.layers import Input, Dense, Reshape, Dropout, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import os
random.seed(2021)
img_shape = (63,78,1)
batch_size = 5
latent_dim = 50



merra_VAE_path = r'/scratch/menglinw/Downscale_data/MERRA2/TransferLearning_result/merra_VAE_result/result1'
merra_encoder = keras.models.load_model(os.path.join(merra_VAE_path, 'encoder_model'), compile=False)
merra_encoder = Model(merra_encoder.input, merra_encoder.layers[-4].output)
merra_decoder = keras.models.load_model(os.path.join(merra_VAE_path, 'decoder_model'), compile=False)
merra_encoder.trainable = False
merra_decoder.trainable = False


g_input = Input(shape=(499, 788, 1))
x = layers.Conv2D(64, (10,10))(g_input)
x = layers.MaxPool2D((2,2))(x)
x = layers.Conv2D(128, (10,10))(x)
x = layers.MaxPool2D((3,4))(x)
x = layers.Conv2D(256, (16, 18))(x)
x = Dense(64)(x)
x = Dense(1)(x)
x = merra_encoder(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers. Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var)*epsilon


latent = layers.Lambda(sampling)([z_mean, z_log_var])
trans_encoder = Model(g_input, latent)

decoder_input = layers.Input(shape=(latent_dim,))
x = merra_decoder(decoder_input)
x = layers.Conv2DTranspose(256, (16,18))(x)
x = layers.UpSampling2D((3,4))(x)
x = layers.Conv2DTranspose(128, (10,10))(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2DTranspose(1, (14,11))(x)
x = Dense(1)(x)
trans_decoder = Model(decoder_input, x)


vae_input = layers.Input(shape=(499, 788, 1))
vae_encoder_output = trans_encoder(vae_input)
vae_decoder_output = trans_decoder(vae_encoder_output)
trans_VAE = Model(vae_input, vae_decoder_output)

def loss_func(z_mean, z_log_var):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(z_mean, z_log_var):
        kl_loss = -0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

trans_VAE.compile(optimizer=Adam(lr=0.00005), loss=loss_func(z_mean, z_log_var))


def data_split(file_path_g5nr_05, file_path_g5nr_06, target_var, n_validate=10):
    '''

    Parameters
    ----------
    X : TYPE ndarray
        independent variable, (train and test)
    Y : TYPE ndarray
        dependent variable, (train and test)
    train_ratio : float
        ration of the train set
    sequential : TYPE Bool, optional
        sample sequentially along time. The default is False.

    Returns
    -------
    None.

    '''
    G5NR_data_05 = nc.Dataset(file_path_g5nr_05)
    G5NR_data_06 = nc.Dataset(file_path_g5nr_06)
    target_data_05 = G5NR_data_05.variables[target_var]
    target_data_06 = G5NR_data_06.variables[target_var]
    target_data = np.zeros((365 * 2, 499, 788, 1))
    target_data[0:365, :, :, 0] = target_data_05[0:365]
    target_data[365:365 * 2, :, :, 0] = target_data_06[:]
    G5NR_data_05, G5NR_data_06, target_data_05, target_data_06 = None, None, None, None
    target_data = target_data / target_data.max()
    n = target_data.shape[0]
    n_train = n - n_validate

    train_ind = random.sample(range(n), n_train)
    vali_ind = np.setdiff1d(list(range(n)), train_ind)
    (train_d, val) = (target_data[train_ind, :, :, :],
                      target_data[vali_ind, :, :, :])
    print('training data shape is', train_d.shape[0])
    print('validation data shape is', val.shape[0])
    return train_d, val, target_data


file_path_g5nr_05 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
file_path_g5nr_06 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
train_dat, val_dat, all_dat = data_split(file_path_g5nr_05, file_path_g5nr_06, 'TOTEXTTAU', n_validate= 35)
callbacks_list = [
        keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
                ),
        keras.callbacks.ModelCheckpoint(
                filepath= 'trans_VAE.h5',
                monitor='val_loss',
                save_best_only=True
                ),
        keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
                )
        ]

history = trans_VAE.fit(x=train_dat, y=train_dat, shuffle=True, epochs=200,
                        batch_size=1, validation_data=(val_dat, val_dat), callbacks=callbacks_list)

np.save('loss_history.npy', history.history)
pred_dat = trans_encoder.predict(all_dat)
trans_encoder.save('trans_encoder_model')
trans_decoder.save('trans_decoder_model')
np.save('pred_dat', pred_dat)