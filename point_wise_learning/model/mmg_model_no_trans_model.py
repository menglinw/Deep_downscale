import numpy as np
import h5py
import os
from tensorflow import keras
import keras.layers as layers
from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
import matplotlib.pyplot as plt
import nc_process
import time


class mmg_model():
    def __init__(self, file_path_h5):
        self.train_file = h5py.File(os.path.join(file_path_h5, 'Train_data.h5'), 'r')
        self.test_file = h5py.File(os.path.join(file_path_h5, 'Test_data.h5'), 'r')
        # define model
        self.model = self.define_model()

    def _generator(self, data_file, batch_size, is_train=True, permut=False):
        indexs = np.arange(0, len(data_file['X']), 1)
        #indexs = np.arange(0, 10000, 1)
        if permut:
            np.random.shuffle(indexs)
        while True:
            for i in range(0, len(indexs), batch_size):
                if is_train:
                    yield data_file['X'][i:i+batch_size], data_file['Y'][i:i+batch_size]
                else:
                    yield data_file['X'][i:i+batch_size]

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
                                                                           is_train=True, permut=True),
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
        y_hat = self.model.predict(self.test_file['X'])
        y_hat = y_hat.reshape((int(len(self.test_file['Y'])/499/788), 499, 788))
        y = self.test_file['Y'][:].reshape((int(len(self.test_file['Y'])/499/788), 499, 788))
        seq_RMSE_mat, seq_R2_mat = self._evaluate(y_hat, y)
        np.save('seq_RMSE_mat', seq_RMSE_mat)
        np.save('seq_R2_mat', seq_R2_mat)
        y_hat.dump('pred_y')
        y.dump('true_y')


    def define_model(self):
        input = Input(shape=(5+11+8))
        x = layers.Dense(32, kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = layers.Dense(16, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = layers.Dense(32, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = layers.Dense(32, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = layers.Dense(32, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(32, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(32, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(32, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(64, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(64, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(64, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(128, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(128, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(128, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(64, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(64, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(32, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(16, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(8, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Dense(1, kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        x = Activation("relu")(x)
        model = Model(input, x)
        model.compile(optimizer='adam', loss="mean_squared_error")
        return model






if __name__ == '__main__':
    start = time.time()
    file_path_h5 = '/scratch/menglinw/Downscale_data/FLAT_DATA/data1'
    # columns: AOD, latitude, longitude, day, elevation, mera_vars, mete_vars
    # merra_var = ['BCEXTTAU', 'DUEXTTAU', 'OCEXTTAU', 'SUEXTTAU', 'TOTEXTTAU', 'BCSMASS', 'DUSMASS25', 'DUSMASS',
    #              'OCSMASS', 'SO4SMASS', 'SSSMASS']
    #  mete_var = ['u10', 'v10', 'd2m', 't2m', 'blh', 'uvb', 'msl', 'tp']
    model = mmg_model(file_path_h5)
    model.train()
    print('Train time:', (time.time()-start)/60, 'mins')
    model.predict()
    print('Duriation:', (time.time()-start)/60, 'mins')