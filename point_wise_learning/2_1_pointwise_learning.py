import numpy as np
import nc_process
import netCDF4 as nc
import random
import data_processing
from tensorflow import keras
import keras.layers as layers
from keras.layers import Input, Dense, Reshape, Dropout, Concatenate, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
import matplotlib.pyplot as plt


class direct_point_learn():
    def __init__(self, file_path_g_05, file_path_g_06, file_path_m, split_type='random_split'):
        self.file_path_g_05 = file_path_g_05
        self.file_path_g_06 = file_path_g_06
        self.file_path_m = file_path_m
        self.split_type = split_type
        self.target_var = 'TOTEXTTAU'
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        self.g_data = np.concatenate((self.data_g5nr_to_array(g05_data), self.data_g5nr_to_array(g06_data)), axis=0)
        m_data = nc.Dataset(file_path_m)
        self.m_data = m_data.variables[self.target_var][range(1826, 1826+730),:,:]
        # define lat&lon of MERRA and G5NR
        M_lons = m_data.variables['lon'][:]
        self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        M_lats = m_data.variables['lat'][:]
        self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
        G_lons = g05_data.variables['lon'][:]
        self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
        G_lats = g05_data.variables['lat'][:]
        self.G_lats = (G_lats-G_lats.mean())/G_lats.std()
        # define model
        self.model = self.define_model()

    def random_split(self, ratio=0.3, seed=2021):
        random.seed(seed)
        test_index = random.sample(range(1, 729), int(730 * ratio))
        train_index = list(set(list(range(729))) - set(test_index))
        return np.array(train_index), np.array(test_index)

    def data_g5nr_to_array(self, nc_data, time_start=0, time_length=365):
        time_interval = range(time_start, time_start + time_length)
        out = nc_data.variables[self.target_var][:][time_interval, :, :]
        return out
    
    def check(self):
        print(self.g_data.shape)
        print(self.m_data.shape)

    def raw_to_table(self, xg_data, xm_data, y_data):
        if xg_data.shape[0] != xm_data.shape[0] or xg_data.shape != y_data.shape:
            print('Please check the data consistency!')
            raise ValueError
        train_x_table = np.zeros((np.prod(xg_data.shape), 4))
        m_list = np.zeros((np.prod(xg_data.shape), 1))
        for i in range(xg_data.shape[0]):
            train_x_table[i*np.prod(xg_data.shape[1:]):(i+1)*np.prod(xg_data.shape[1:])] = data_processing.\
                image_to_table(xg_data[i], self.G_lats, self.G_lons, (i%365)/365)
            m_list[i*np.prod(xg_data.shape[1:]): (i+1)*np.prod(xg_data.shape[1:]),0] = data_processing.\
                resolution_downward(xm_data[i],self.M_lats, self.M_lons, self.G_lats, self.G_lons).\
                reshape(np.prod(xg_data.shape[1:]))
        train_x_table = np.concatenate((train_x_table, m_list), 1)
        train_y_table = y_data.reshape(np.prod(y_data.shape))
        return train_x_table, train_y_table

    def define_model(self):
        input = Input(shape=(5))
        x = layers.Dense(8, kernel_initializer="he_normal")(input)
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

    def evaluate(self, pred_data, true_data):
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



    def train(self):
        if self.split_type == 'random_split':
            train_x_index, test_y_index = self.random_split()
            train_y_index = train_x_index+1
            test_x_index = test_y_index-1
            #print(train_y_index)
            # train raw data
            train_xg_data = self.g_data[train_x_index,:,:]
            train_xm_data = self.m_data[train_x_index,:,:]
            train_y_data_3d = self.g_data[train_y_index,:,:]
            train_x_data, train_y_data = self.raw_to_table(train_xg_data, train_xm_data,train_y_data_3d)
            # test raw data
            test_xg_data = self.g_data[test_x_index,:,:]
            test_xm_data = self.m_data[test_x_index,:,:]
            self.test_y_data_3d = self.g_data[test_y_index,:,:]
            self.test_x_data, test_y_data = self.raw_to_table(test_xg_data, test_xm_data, self.test_y_data_3d)
            callbacks_list = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='latent_LSTM.h5',
                    monitor='val_loss',
                    save_best_only=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=5
                )
            ]
            history = self.model.fit(train_x_data, train_y_data,
                                     validation_data= (self.test_x_data[:100*499*788], test_y_data[:100*499*788]),
                                     batch_size=499*788, epochs=100)
            self.model.save('pointwise_1_randomsplit')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label="train loss")
            plt.plot(history.history['val_loss'], label='test loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig('loss_curve.jpg')
        else:
            # TODO sequential (cross-crosvalidation) 30%
            test_index = np.array(range(73-1, 4*73))
            self.test_index = test_index
            train_x_index = np.array(list(set(range(729))-set(test_index)))
            train_y_index = train_x_index + 1
            # train raw data
            train_xg_data = self.g_data[train_x_index, :, :]
            train_xm_data = self.m_data[train_x_index, :, :]
            train_y_data_3d = self.g_data[train_y_index, :, :]
            train_x_data, train_y_data = self.raw_to_table(train_xg_data, train_xm_data, train_y_data_3d)

            # test raw data
            test_xg_data = self.g_data[test_index[:-1], :, :]
            test_xm_data = self.m_data[test_index[:-1], :, :]
            self.test_y_data_3d = self.g_data[test_index[1:], :, :]
            self.test_x_data, test_y_data = self.raw_to_table(test_xg_data, test_xm_data, self.test_y_data_3d)
            callbacks_list = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='latent_LSTM.h5',
                    monitor='val_loss',
                    save_best_only=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=5
                )
            ]
            history = self.model.fit(train_x_data, train_y_data,
                                     validation_data=(self.test_x_data[:100*499*788], test_y_data[:100*499*788]),
                                     batch_size=499*788, epochs=50)
            self.model.save('pointwise_1_sequential')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label="train loss")
            plt.plot(history.history['val_loss'], label='test loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig('loss_curve.jpg')

    def predict(self):
        if self.split_type == 'random_split':
            pred_y = self.model.predict(self.test_x_data)
            pred_y = pred_y.reshape(self.test_y_data_3d.shape)
            RMSE_mat, R2_mat = self.evaluate(pred_y, self.test_y_data_3d)
            np.save('RMSE_mat', RMSE_mat)
            np.save('R2_mat', R2_mat)
        else:
            pred_y = self.model.predict(self.test_x_data)
            pred_y = pred_y.reshape(self.test_y_data_3d.shape)
            RMSE_mat, R2_mat = self.evaluate(pred_y, self.test_y_data_3d)
            np.save('RMSE_mat', RMSE_mat)
            np.save('R2_mat', R2_mat)
            # TODO step-by-step predict
            test_xg_data = self.g_data[self.test_index, :, :]
            test_xm_data = self.m_data[self.test_index, :, :]
            pred_seq_y = np.zeros_like(self.test_y_data_3d)
            prev_g_image = test_xg_data[0,:,:]
            prev_m_image = test_xm_data[0,:,:]
            for i in range(len(self.test_index[1:])):
                m_list = np.zeros((np.prod(prev_g_image.shape), 1))
                input_x_data = data_processing.image_to_table(prev_g_image, self.G_lats, self.G_lons,
                                                              (self.test_index[i + 1] % 365) / 365)
                m_list[:, 0] = data_processing.resolution_downward(prev_m_image,self.M_lats, self.M_lons,
                                                                self.G_lats, self.G_lons).\
                    reshape(np.prod(prev_g_image.shape))
                input_x_data = np.concatenate((input_x_data, m_list), 1)
                pred_seq_sub_y = self.model.predict_on_batch(input_x_data)
                prev_g_image = pred_seq_sub_y.reshape(prev_g_image.shape)
                prev_m_image = test_xm_data[i+1,:,:]
                pred_seq_y[i, :, :] = prev_g_image
            seq_RMSE_mat, seq_R2_mat = self.evaluate(pred_seq_y, self.test_y_data_3d)
            np.save('seq_RMSE_mat', seq_RMSE_mat)
            np.save('seq_R2_mat', seq_R2_mat)
            pred_seq_y.dump('seq_pred_y')
            self.test_y_data_3d.dump('true_y')


if __name__=='__main__':
    # define some universal  variables
    # file path of G5NR 06-07, and 05-06
    file_path_g_06 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
        #r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
        #r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    # file path of MERRA-2 data
    file_path_m = '/scratch/menglinw/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
        #r'C:\Users\96349\Documents\Downscale_data\MERRA2\MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    model = direct_point_learn(file_path_g_05, file_path_g_06, file_path_m, split_type='sequential')
    model.train()
    model.predict()