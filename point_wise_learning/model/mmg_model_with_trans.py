import numpy as np
import h5py
import os
from tensorflow import keras
import keras.layers as layers
from keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Dropout
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
import nc_process
import time


class mmg_model():
    def __init__(self, file_path_h5, within_merra_model_path):
        within_merra_model = keras.models.load_model(os.path.join(within_merra_model_path, 'within_merra_model.h5'),
                                              compile=False)
        self.merra_model = Model(within_merra_model.input, within_merra_model.layers[-10].output)
        self.train_file = h5py.File(os.path.join(file_path_h5, 'Norm_train_data.h5'), 'r')
        self.test_file = h5py.File(os.path.join(file_path_h5, 'Norm_test_data.h5'), 'r')
        # define model
        self.model = self.define_model()

    def _generator(self, data_file, batch_size, is_train=True, permut=False, validation=False):
        if validation:
            indexs = np.arange(0, len(data_file['X'])-73*499*788, 1)
        else:
            indexs = np.arange(0, len(data_file['X']), 1)
        #indexs = np.arange(0, 10000, 1)
        if permut:
            np.random.shuffle(indexs)
        while True:
            for i in range(0, len(indexs), batch_size):
                if is_train:
                    yield [data_file['X'][i:i+batch_size, :5], data_file['X'][i:i+batch_size, 5:16],
                           data_file['X'][i:i+batch_size, 16:]], data_file['Y'][i:i+batch_size]
                else:
                    yield [data_file['X'][i:i+batch_size, :5], data_file['X'][i:i+batch_size, 5:16],
                           data_file['X'][i:i+batch_size, 16:]]

    def train(self, epochs=100):
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='mmg_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3
            )
        ]
        history = self.model.fit_generator(generator=self._generator(data_file=self.train_file, batch_size=788,
                                                                     is_train=True, permut=True),
                                           epochs=epochs,
                                           steps_per_epoch=len(self.train_file['X'])/788,
                                           validation_data=self._generator(data_file=self.test_file, batch_size=788,
                                                                           is_train=True, permut=True, validation=True),
                                           validation_steps=len(self.test_file['X'])/788,
                                           callbacks=callbacks_list
                                           )
        fig = plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label="train loss")
        plt.plot(history.history['val_loss'], label='test loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss_curve.jpg')

    def _evaluate(self, pred_data, true_data):
        if pred_data.shape != true_data.shape:
            print('Please check data consistency!')
            raise ValueError
        RMSE_out = np.zeros(pred_data.shape[1:])
        R2_out = np.zeros(pred_data.shape[1:])
        for i in range(pred_data.shape[1]):
            for j in range(pred_data.shape[2]):
                RMSE_out[i,j] = np.square(pred_data[:,i,j] - true_data[:,i,j]).mean()
                R2_out[i,j], _ = nc_process.rsquared(pred_data[:,i,j], true_data[:,i,j])
        return RMSE_out, R2_out

    def predict(self):
        y_hat = self.model.predict(self.test_file['X'][-73*499*788:])
        y_hat = y_hat.reshape((73, 499, 788))
        y = self.test_file['Y'][-73*499*788:].reshape((73, 499, 788))
        seq_RMSE_mat, seq_R2_mat = self._evaluate(y_hat, y)
        np.save('seq_RMSE_mat', seq_RMSE_mat)
        np.save('seq_R2_mat', seq_R2_mat)
        y_hat.dump('pred_y')
        y.dump('true_y')


    def define_model(self):
        def unit_layer(nodes, input, with_dropout=False, activation='LeakyReLU'):
            if activation == 'LeakyReLU':
                x = layers.Dense(8, kernel_initializer="he_normal")(input)
                x = BatchNormalization()(x)
                if with_dropout:
                    x = Dropout(0.5)(x)
                x = LeakyReLU(alpha=0.1)(x)
            else:
                x = layers.Dense(8, kernel_initializer="he_normal", activation=activation)(input)
                x = BatchNormalization()(x)
                if with_dropout:
                    x = Dropout(0.5)(x)
            return x

        def input_to_encoder(input):
            x = unit_layer(8, input)
            x = unit_layer(16, x)
            x = unit_layer(32, x, with_dropout=True, activation='tanh')
            return x

        def mapping_to_target_range(x, target_min=0, target_max=6):
            x02 = K.tanh(x) + 1  # x in range(0,2)
            scale = (target_max - target_min) / 2.
            return x02 * scale + target_min

        input1 = Input(shape=(5))  # input of AOD, lat, lon, day, elev
        input2 = Input(shape=(11))  # input of merra
        input3 = Input(shape=(8))  # input of mete
        t_input1 = self.merra_model(input1)
        encode1 = input_to_encoder(input1)
        encode2 = input_to_encoder(input2)
        encode3 = input_to_encoder(input3)
        X = layers.Concatenate(axis=1)([encode1, t_input1, encode2, encode3])
        X = unit_layer(64, X)
        X = unit_layer(32, X)
        X = unit_layer(16, X)
        X = unit_layer(8, X)
        X = unit_layer(16, X)
        X = unit_layer(32, X)
        X = unit_layer(64, X)
        X = layers.Dense(1, activation=mapping_to_target_range)(X)
        model = Model([input1, input2, input3], X)
        model.compile(optimizer='adam', loss="mean_squared_error")
        return model







if __name__ == '__main__':
    start = time.time()
    file_path_h5 = '/scratch/menglinw/Downscale_data/FLAT_DATA/data1'
    within_merra_model_path = '/scratch/menglinw/Downscale_data/MERRA2/Pointwise_learning/mete_merra_model_result/within_merra_model_result/result5'
    # columns: AOD, latitude, longitude, day, elevation, mera_vars, mete_vars
    # merra_var = ['BCEXTTAU', 'DUEXTTAU', 'OCEXTTAU', 'SUEXTTAU', 'TOTEXTTAU', 'BCSMASS', 'DUSMASS25', 'DUSMASS',
    #              'OCSMASS', 'SO4SMASS', 'SSSMASS']
    #  mete_var = ['u10', 'v10', 'd2m', 't2m', 'blh', 'uvb', 'msl', 'tp']
    model = mmg_model(file_path_h5, within_merra_model_path)
    model.train()
    print('Train time:', (time.time()-start)/60, 'mins')
    model.predict()
    print('Duriation:', (time.time()-start)/60, 'mins')