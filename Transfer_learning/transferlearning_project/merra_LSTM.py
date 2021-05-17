import numpy as np
import pandas as pd
import netCDF4 as nc
from tensorflow import keras
import keras.layers as layers
from keras.layers import Input, Dense, Reshape, Dropout, Concatenate, LSTM
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import random_shift
from keras import backend as K
import os
import random
import matplotlib.pyplot as plt
from scipy import stats



lagging = 10
latent_dim = 500


def data_split(latent_data_path, test_size=50, lagging=5):
    '''
    output: train set, test set, test index
    '''
    latent_dat = np.load(os.path.join(latent_data_path, 'pred_dat.npy'))
    n = latent_dat.shape[0] - lagging
    t = lagging + 1
    m = latent_dat.shape[1]
    data = np.zeros((n, t, m))
    for i in range(data.shape[0]-1):
        start = i
        end = i + 6
        data[i, :, :] = latent_dat[start:end, :]
    test_indx = random.sample(range(n), test_size)
    train_indx = np.setdiff1d(list(range(n)), test_indx)
    return data[train_indx], data[test_indx], np.array(test_indx)+lagging


# Model define
input1 = Input(shape=(lagging, latent_dim))
x = layers.BatchNormalization()(input1)
x = LSTM(100, return_sequences=True)(input1)
x = LSTM(75, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(50, return_sequences=True)(x)
x = LSTM(50)(x)
x = Dense(latent_dim, activation='softmax')(x)
merra_model = Model(input1, x)
merra_model.compile(optimizer=Adam(0.0002, 0.5), loss=keras.losses.MeanSquaredError())

# training
latent_dat_path = '/scratch/menglinw/Downscale_data/MERRA2/TransferLearning_result/merra_VAE_result/result2'
train_dat, test_dat, test_index = data_split(latent_dat_path, 300)

callbacks_list = [
        keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
                ),
        keras.callbacks.ModelCheckpoint(
                filepath= 'latent_LSTM.h5',
                monitor='val_loss',
                save_best_only=True
                ),
        keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
                )
        ]
merra_history = merra_model.fit(train_dat[:,:5,:],train_dat[:,-1,:],
                    validation_data=(test_dat[:,:5,:], test_dat[:,-1,:]),
                    epochs=10, batch_size=5, callbacks=callbacks_list)
print('The shape of input test data is:', test_dat[:,:5,:].shape)
merra_pred_dat = merra_model.predict(test_dat[:,:5,:])
print('MERRA LSTM model predicted successfully!')

def plot_loss(history, name='loss_curve_merraLSTM.jpg'):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label="train loss")
    plt.plot(history.history['val_loss'], label='test loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(name)


plot_loss(merra_history)

# trans LSTM
merra_model.trainable = False
input2 = Input(shape=(lagging, latent_dim))
y = LSTM(100, return_sequences=True)(input2)
y = LSTM(latent_dim, return_sequences=True)(y)
y = merra_model(y)
trans_model = Model(input2, y)
trans_model.compile(optimizer=Adam(0.0002, 0.5), loss=keras.losses.MeanSquaredError())
trans_latent_dat_path = '/scratch/menglinw/Downscale_data/MERRA2/TransferLearning_result/trans_VAE_result/result2'
t_train_dat, t_test_dat, t_test_index = data_split(trans_latent_dat_path, 100)
print('train shape:', t_train_dat.shape)
print('test shape:', t_test_dat.shape)

callbacks_list_2 = [
        keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
                ),
        keras.callbacks.ModelCheckpoint(
                filepath= 'trans_LSTM.h5',
                monitor='val_loss',
                save_best_only=True
                ),
        keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
                )
        ]
trans_history = trans_model.fit(t_train_dat[:,:5,:], t_train_dat[:,-1,:],
                    validation_data=(t_test_dat[:,:5,:], t_test_dat[:,-1,:]),
                    epochs=50, batch_size=5, callbacks=callbacks_list_2)
plot_loss(trans_history, 'loss_curve_transLSTM.jpg')
pred_trans_dat = trans_model.predict(t_test_dat[:,:5,:])

trans_decoder_path = trans_latent_dat_path
                     #"C:\Users\96349\Documents\Downscale_data\result\TransferLearning\trans_VAE\trans_VAEResult1"
trans_decoder = keras.models.load_model(os.path.join(trans_decoder_path, 'trans_decoder_model'), compile=False)
pred_G_data = trans_decoder.predict(pred_trans_dat)


def data_extract(file_path_g5nr_05, file_path_g5nr_06, target_var, indx):
    G5NR_data_05 = nc.Dataset(file_path_g5nr_05)
    G5NR_data_06 = nc.Dataset(file_path_g5nr_06)
    target_data_05 = G5NR_data_05.variables[target_var]
    target_data_06 = G5NR_data_06.variables[target_var]
    target_data = np.zeros((365 * 2, 499, 788, 1))
    target_data[0:365, :, :, 0] = target_data_05[0:365]
    target_data[365:365 * 2, :, :, 0] = target_data_06[:]
    G5NR_data_05, G5NR_data_06, target_data_05, target_data_06 = None, None, None, None
    return target_data[indx, :, :, 0]


file_path_g5nr_05 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
file_path_g5nr_06 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
true_G_data = data_extract(file_path_g5nr_05, file_path_g5nr_06, 'TOTEXTTAU', t_test_index)


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return((r_value**2, p_value))


def result_R2_cal(predicted_y, true_y):
    n_image = predicted_y.shape[0]
    rs_p_y = predicted_y.reshape(n_image, 499*788)
    rs_t_y = true_y.reshape(n_image, 499*788)
    R_list = []
    for i in range(n_image):
        r2,p = rsquared(rs_p_y[i], rs_t_y[i])
        abs_err = np.absolute(rs_p_y[i] -rs_t_y[i]).mean()
        a_p_err = (100*np.absolute((rs_p_y[i] -rs_t_y[i]))/rs_t_y[i]).mean()
        R_list.append([r2, abs_err, a_p_err])
    return pd.DataFrame( R_list, columns=['R_square','abs_err', 'a_p_err'])


r2_result = result_R2_cal(pred_G_data, true_G_data)
fig = plt.figure(figsize=(12, 6))
plt.plot(r2_result['R_square'], label="R square")
plt.plot(r2_result['abs_err'], label='abs error')
plt.plot(r2_result['a_p_err'], label='abs error')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('predicted_G5NR_summary.jpg')