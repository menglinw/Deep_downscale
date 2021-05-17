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
import os


class point_trans_model():
    def __init__(self, merra_model_path, file_path_g_05, file_path_g_06):
        self.merra_model_path = merra_model_path
        self.target_var = 'TOTEXTTAU'
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        self.g_data = np.concatenate((self.data_g5nr_to_array(g05_data), self.data_g5nr_to_array(g06_data)), axis=0)
        G_lons = g05_data.variables['lon'][:]
        self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
        G_lats = g05_data.variables['lat'][:]
        self.G_lats = (G_lats-G_lats.mean())/G_lats.std()
        # define model
        self.model = self.define_model()

    def data_g5nr_to_array(self, nc_data, time_start=0, time_length=365):
        time_interval = range(time_start, time_start + time_length)
        out = nc_data.variables[self.target_var][:][time_interval, :, :]
        return out

    def define_model(self):
        merra_model = keras.models.load_model(os.path.join(self.merra_model_path, 'pointwise_1_randomsplit'), compile=False)
        merra_model = Model(merra_model.input, merra_model.layers[-4].output)
        merra_model.trainable = False
        input = Input(shape=(4))
        x = layers.Dense(4, kernel_initializer="he_normal")(input)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Activation("linear")(x)
        x = merra_model(x)
        x = layers.Dense(8, kernel_initializer="he_normal")(x)
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

    def raw_to_table(self, x_data, y_data):
        if x_data.shape != y_data.shape:
            print('Please check the data consistency!')
            raise ValueError
        train_x_table = np.zeros((np.prod(x_data.shape), 4))
        for i in range(x_data.shape[0]):
            train_x_table[i*np.prod(x_data.shape[1:]):(i+1)*np.prod(x_data.shape[1:])] = data_processing.\
                image_to_table(x_data[i], self.G_lats, self.G_lons, (i%365)/365)
        train_y_table = y_data.reshape(np.prod(y_data.shape))
        return train_x_table, train_y_table

    def train(self, test_season=1):
        self.test_season=test_season
        test_index = np.array(range((1+(test_season-1)*3)*73-2, (1+test_season*3) * 73 -1))
        self.test_index = test_index
        train_x_index = np.array(list(set(range(729)) - set(test_index)))
        train_y_index = train_x_index + 1
        # train raw data
        train_x_data = self.g_data[train_x_index, :, :]
        train_y_data_3d = self.g_data[train_y_index, :, :]
        train_x_data, train_y_data = self.raw_to_table(train_x_data,  train_y_data_3d)
        # sequential prediction
        # test raw data
        test_x_data_3d = self.g_data[test_index[:-1], :, :]
        self.test_x_data_3d = test_x_data_3d
        self.test_y_data_3d = self.g_data[test_index[1:], :, :]
        self.test_x_data, test_y_data = self.raw_to_table(test_x_data_3d, self.test_y_data_3d)
        history = self.model.fit(train_x_data, train_y_data, batch_size=499, epochs=50)
        self.model.save('trans_pointwise_model')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label="train loss")
        # plt.plot(history.history['val_loss'], label='test loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss_curve.jpg')

    def predict(self):
        pred_y = self.model.predict(self.test_x_data)
        pred_y = pred_y.reshape(self.test_y_data_3d.shape)
        RMSE_mat, R2_mat = self.evaluate(pred_y, self.test_y_data_3d)
        np.save('RMSE_mat', RMSE_mat)
        np.save('R2_mat', R2_mat)
        # TODO step-by-step predict
        pred_seq_y = np.zeros_like(self.test_y_data_3d)
        prev_image = self.test_x_data_3d[0,:,:]
        for i in range(len(self.test_index[1:])):
            input_x_data = data_processing.image_to_table(prev_image, self.G_lats, self.G_lons, (self.test_index[i+1]%365)/365)
            pred_seq_sub_y = self.model.predict_on_batch(input_x_data)
            prev_image = pred_seq_sub_y.reshape(prev_image.shape)
            pred_seq_y[i,:,:] = prev_image
        seq_RMSE_mat, seq_R2_mat = self.evaluate(pred_seq_y, self.test_y_data_3d)
        np.save('seq_RMSE_mat', seq_RMSE_mat)
        np.save('seq_R2_mat', seq_R2_mat)
        np.save('seq_pred_y', pred_seq_y)
        np.save('true_y', self.test_y_data_3d)

if __name__=="__main__":

    # merra_model_path = r'C:\Users\96349\Documents\Downscale_data\result\Pointwise_learning\merra_model\result1'
    merra_model_path = r'/scratch/menglinw/Downscale_data/MERRA2/Pointwise_learning/result_merra/m_result3'
    file_path_g_06 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    # file_path_g_06 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    # file_path_g_05 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    model = point_trans_model(merra_model_path, file_path_g_05, file_path_g_06)
    model.train(test_season=3)
    model.predict()