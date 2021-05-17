# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:39:57 2021

@author: 96349
"""


import numpy as np
import pandas as pd
import netCDF4 as nc
import keras
import keras.layers as layers
from keras.layers import Input, Dense, Reshape, Dropout, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import random_shift
from keras import backend as K
import json
import random
import matplotlib.pyplot as plt


img_shape = (63, 78, 1)
batch_size = 5
latent_dim = 500

# Encoder
input_img = keras.Input(shape=img_shape)
x = layers.Conv2D(32, 3, activation='relu')(input_img)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
shape_before_flattening = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers. Dense(latent_dim)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var)*epsilon


z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = Model(input_img, z)

# decoder
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2DTranspose(32, 2, activation='relu')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2DTranspose(32, 2, activation='relu')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2DTranspose(32, 2, activation='relu')(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2DTranspose(32, (2,17), activation='relu')(x)
x = layers.Conv2D(1, 3, padding='same')(x)
decoder = Model(decoder_input, x)

vae_input = layers.Input(shape=img_shape)
vae_encoder_output = encoder(vae_input)
vae_decoder_output = decoder(vae_encoder_output)
vae = Model(vae_input, vae_decoder_output)


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

vae.compile(optimizer=Adam(lr=0.0005), loss=loss_func(z_mean, z_log_var))


def data_split(merra_file_path, target_var, n_validate = 10):

        merra_data = nc.Dataset(merra_file_path)
        merra_data = merra_data.variables[target_var]
        merra_ndarray = np.zeros(list(merra_data.shape)+[1])
        n = merra_data.shape[0]
        for i in range(n):
            merra_ndarray[i,:,:,0] = merra_data[i]
        n_train = n - n_validate
        train_ind = random.sample(range(n), n_train)
        vali_ind = np.setdiff1d(list(range(n)), train_ind)
        
        (train_d,val) = (merra_ndarray[train_ind, :, :], merra_ndarray[vali_ind, :, :])
        print('training data shape is', train_d.shape[0])
        print('validation data shape is', val.shape[0])
        return train_d, val, merra_ndarray

file_path = '/scratch/menglinw/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'

train_dat, val_dat, all_dat = data_split(file_path,'TOTEXTTAU', 500)

callbacks_list = [
        keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
                ),
        keras.callbacks.ModelCheckpoint(
                filepath= 'merra_VAE.h5',
                monitor='val_loss',
                save_best_only=True
                ),
        keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
                )
        ]
history = vae.fit(x=train_dat, y = train_dat, shuffle=True, epochs = 500, 
                  batch_size = 10, validation_data = (val_dat,val_dat),
                  callbacks=callbacks_list)

def plot_loss(history, name='loss_curve_merraLSTM.jpg'):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label="train loss")
    plt.plot(history.history['val_loss'], label='test loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(name)


plot_loss(history, name='merra_VAE_loss_curve.jpg')
pred_dat = encoder.predict(all_dat)
encoder.save('encoder_model')
decoder.save('decoder_model')
np.save('pred_dat',pred_dat)

