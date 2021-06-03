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

lagging = 5
latent_dim = 50

# 1. load merra_LSTM
merra_LSTM_path = r'C:\Users\96349\Documents\Downscale_data\result\TransferLearning\merra_LSTM\merraLSTM_Result1'
    # r'/scratch/menglinw/Downscale_data/MERRA2/TransferLearning_result/merra_LSTM/result1'
merra_LSTM = keras.models.load_model(os.path.join(merra_LSTM_path, 'latent_LSTM.h5'), compile=False)
trans_decoder_path = r"C:\Users\96349\Documents\Downscale_data\result\TransferLearning\trans_VAE\trans_VAEResult1"
    #r'/scratch/menglinw/Downscale_data/MERRA2/TransferLearning_result/trans_VAE_result/result1'
trans_decoder = keras.models.load_model(os.path.join(trans_decoder_path, 'trans_decoder_model'), compile=False)
# 2. define trans_LSTM
# 3. RE-define data_split function:
    # test should be monthly
    # return train_data, test_data, test_index
# 4. train trans_LSTM
# 5. plot loss_curve
# 6. predict on test set
# 7. load trans_decoder
# 8. decode
# 9. calculate R2 and plot
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
input1 = Input(batch_shape=(5, lagging, latent_dim))
x = layers.BatchNormalization()(input1)
x = LSTM(500, return_sequences=True)(input1)
x = LSTM(450, return_sequences=True)(x)
x = LSTM(400, return_sequences=True)(x)
x = LSTM(350, return_sequences=True)(x)
x = LSTM(300, return_sequences=True)(x)
x = LSTM(200, return_sequences=True)(x)
x = LSTM(100, return_sequences=True)(x)
x = LSTM(50)(x)
x = Dense(50, activation='softmax')(x)
model = Model(input1, x)
model.compile(optimizer=Adam(0.0002, 0.5), loss=keras.losses.MeanSquaredError())

# training
latent_dat_path = '/scratch/menglinw/Downscale_data/MERRA2/TransferLearning_result/merra_VAE_result/result1'
train_dat, test_dat, test_index = data_split(latent_dat_path, 150)

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
history = model.fit(train_dat[:,:5,:],train_dat[:,-1,:],
                    validation_data=(test_dat[:,:5,:], test_dat[:,-1,:]),
                    epochs=50, batch_size=5, callbacks=callbacks_list)
model.save('merra_LSTM')
fig = plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label='test loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_curve.jpg')