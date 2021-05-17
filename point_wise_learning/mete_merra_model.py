import numpy as np
import nc_process
import netCDF4 as nc
import data_processing
from tensorflow import keras
import keras.layers as layers
from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
import matplotlib.pyplot as plt
import datetime

# Input: MERRA-2, meteology data
#
# output: G5NR data


class mete_merra_to_g5nr_model():
    def __init__(self, file_path_g_05, file_path_g_06, file_path_m, file_path_mete05, file_path_mete06, file_path_mete07,
                 test_season=1):
        # define path and target variable
        self.test_season = test_season
        self.file_path_g_05 = file_path_g_05
        self.file_path_g_06 = file_path_g_06
        self.file_path_m = file_path_m
        self.file_path_mete05 = file_path_mete05
        self.file_path_mete06 = file_path_mete06
        self.file_path_mete07 = file_path_mete07
        self.target_var = 'TOTEXTTAU'

        # read g5nr data
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        self.g_data = np.concatenate((self._data_g5nr_to_array(g05_data), self._data_g5nr_to_array(g06_data)), axis=0)

        # read merra data as nc
        self.m_data = nc.Dataset(file_path_m)
        #self.m_data = m_data.variables[self.target_var][range(1826, 1826+730), :, :]

        # read mete data as nc
        self.mete05_data = nc.Dataset(file_path_mete05)
        self.mete06_data = nc.Dataset(file_path_mete06)
        self.mete07_data = nc.Dataset(file_path_mete07)

        # processing data
        self._mete_temp_avg()
        self._train_test_split(test_season=test_season)

        # define lat&lon of MERRA, G5NR and mete
        self.M_lons = self.m_data.variables['lon'][:]
        #self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        self.M_lats = self.m_data.variables['lat'][:]
        # self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
        self.G_lons = g05_data.variables['lon'][:]
        # self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
        self.G_lats = g05_data.variables['lat'][:]
        # self.G_lats = (G_lats-G_lats.mean())/G_lats.std()
        self.Mete_lons = self.mete05_data.variables['longitude'][:]
        self.Mete_lats = self.mete05_data.variables['latitude'][:]

        # define model
        self.model = self.define_model()

    def _data_g5nr_to_array(self, nc_data, time_start=0, time_length=365):
        time_interval = range(time_start, time_start + time_length)
        out = nc_data.variables[self.target_var][:][time_interval, :, :]
        return out

    def _mete_temp_avg(self):
        # temporal convert mete data to match g5nr
        # 1. daily average
        # 2. take 2005/5/16 - 2007/5/15
        self.Mete_data = dict()
        # 2005 mete data, take 5/16/2005 - last, average over days
        t0 = datetime.datetime.strptime('2005-01-01 00:00', "%Y-%m-%d %H:%M")
        t1 = datetime.datetime.strptime('2005-05-16 00:00', "%Y-%m-%d %H:%M")
        start_day = t1.__sub__(t0).days
        start_hour = start_day*24
        for var in self.mete05_data.variables.keys():
            if var not in ['longitude', 'latitude', 'time']:
                time_len, lat_len, lon_len = self.mete05_data.variables[var].shape
                temp_data = np.zeros((int((time_len - start_hour) / 24), lat_len, lon_len))
                for i in range(int((time_len - start_hour) / 24)):
                    temp_data[i, :, :] = np.mean(self.mete05_data.variables[var][start_hour + i * 24:(i * 24 + 24 + start_hour)])
                self.Mete_data[var] = temp_data

        # 2006 take full year
        for var in self.mete06_data.variables.keys():
            if var not in ['longitude', 'latitude', 'time']:
                time_len, lat_len, lon_len = self.mete06_data.variables[var].shape
                temp_data = np.zeros((int(time_len/ 24), lat_len, lon_len))
                for i in range(int(time_len / 24)):
                    temp_data[i, :, :] = np.mean(self.mete06_data.variables[var][i * 24:(i * 24 + 24)])
                self.Mete_data[var] = np.concatenate((self.Mete_data[var], temp_data), axis=0)
        # 2007 take to 5/15/2007
        t0 = datetime.datetime.strptime('2007-01-01 00:00', "%Y-%m-%d %H:%M")
        t1 = datetime.datetime.strptime('2007-05-16 01:00', "%Y-%m-%d %H:%M")
        end_day = t1.__sub__(t0).days
        for var in self.mete07_data.variables.keys():
            if var not in ['longitude', 'latitude', 'time']:
                time_len, lat_len, lon_len = self.mete07_data.variables[var].shape
                temp_data = np.zeros((end_day, lat_len, lon_len))
                for i in range(end_day):
                    temp_data[i, :, :] = np.mean(self.mete07_data.variables[var][i * 24:(i * 24 + 24)])
                self.Mete_data[var] = np.concatenate((self.Mete_data[var], temp_data), axis=0)

    def _train_test_split(self, test_season):
        '''
        self.g_data: ndarray
        self.mete_data: dict
        self.m_data: nc
        '''
        self.test_season=test_season
        test_index = np.array(range((1+(test_season-1)*3)*73-2, (1+test_season*3) * 73-1))
        self.test_index = test_index
        train_x_index = np.array(list(set(range(729)) - set(test_index)))
        self.xday_list = train_x_index
        self.yday_list = test_index[:-1]
        train_y_index = train_x_index + 1
        # train raw data
        self.train_xg_data = self.g_data[train_x_index, :, :]

        train_xm_data = dict()
        for var in self.m_data.variables.keys():
            if var not in ['lat', 'lon', 'time']:
                train_xm_data[var] = self.m_data.variables[var][train_x_index, :, :]
        self.train_xm_data = train_xm_data

        train_xme_data = dict()
        for var, data in self.Mete_data.items():
            train_xme_data[var] = data[train_x_index, :, :]
        self.train_xme_data = train_xme_data
        self.train_y_data = self.g_data[train_y_index, :, :]

        # test raw data
        self.test_xg_data = self.g_data[test_index[:-1], :, :]

        test_xm_data = dict()
        for var in self.m_data.variables.keys():
            if var not in ['lat', 'lon', 'time']:
                test_xm_data[var] = self.m_data.variables[var][test_index[:-1], :, :]
        self.test_xm_data = test_xm_data

        test_xme_data = dict()
        for var, data in self.Mete_data.items():
            test_xme_data[var] = data[test_index[:-1], :, :]
        self.test_xme_data = test_xme_data
        self.test_y_data = self.g_data[test_index[1:], :, :]


    def define_model(self):
        input = Input(shape=(26))
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

    def _generator(self, xg_data, xm_data, xme_data, yg_data, day_list):
        
        while True:
            for i, day in enumerate(day_list):
                permutation = np.random.permutation(np.prod(yg_data[i].shape))
                outx_table = data_processing.image_to_table(xg_data[i, :, :],
                                                            (self.G_lats-self.G_lats.mean())/self.G_lats.std(),
                                                            (self.G_lons - self.G_lons.mean())/self.G_lons.std(),
                                                            (day%365)/365)
                for var, data in xm_data.items():
                    xm_img = data_processing.resolution_downward(data[i, :, :], self.M_lats, self.M_lons,
                                                                 self.G_lats, self.G_lons)
                    outx_table = np.concatenate((outx_table, xm_img.reshape((np.prod(xm_img.shape), 1))), 1)

                for var, data in xme_data.items():
                    xme_img = data_processing.resolution_downward(data[i, :, :], self.Mete_lats, self.Mete_lons,
                                                                  self.G_lats, self.G_lons)
                    outx_table = np.concatenate((outx_table, xme_img.reshape((np.prod(xme_img.shape), 1))), 1)
                outy = yg_data[i].reshape((np.prod(yg_data[i].shape), 1))
                yield outx_table[permutation], outy[permutation]

    def train(self, epoch=50):
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='mm_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5
            )
        ]
        history = self.model.fit_generator(generator=self._generator(self.train_xg_data, self.train_xm_data, self.train_xme_data,
                                                           self.train_y_data, self.xday_list),
                                           epochs=epoch,
                                           validation_data=self._generator(self.test_xg_data, self.test_xm_data,
                                                                           self.test_xme_data,
                                                                           self.test_y_data, self.yday_list),
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
        pred_seq_y = np.zeros_like(self.test_y_data)
        # initialization
        outx_table = data_processing.image_to_table(self.test_y_data[0, :, :],
                                                    (self.G_lats - self.G_lats.mean()) / self.G_lats.std(),
                                                    (self.G_lons - self.G_lons.mean()) / self.G_lons.std(),
                                                    (((self.yday_list[0])% 365) / 365))
        for var, data in self.test_xm_data.items():
            xm_img = data_processing.resolution_downward(data[0, :, :], self.M_lats, self.M_lons,
                                                         self.G_lats, self.G_lons)
            outx_table = np.concatenate((outx_table, xm_img.reshape(np.prod(xm_img.shape))), 1)

        for var, data in self.test_xme_data.items():
            xme_img = data_processing.resolution_downward(data[0, :, :], self.Mete_lats, self.Mete_lons,
                                                          self.G_lats, self.G_lons)
            outx_table = np.concatenate((outx_table, xme_img.reshape(np.prod(xme_img.shape))), 1)
        # predict step by step
        for i, day in enumerate(self.yday_list):
            pred_y_flat = self.model.predict_on_batch(outx_table)
            prev_g_image = pred_y_flat.reshape(self.test_y_data.shape[1:])
            if i != len(self.yday_list)-1:
                outx_table = data_processing.image_to_table(prev_g_image,
                                                        (self.G_lats - self.G_lats.mean()) / self.G_lats.std(),
                                                        (self.G_lons - self.G_lons.mean()) / self.G_lons.std(),
                                                        ((day + 1 % 365) / 365))
                for var, data in self.test_xm_data.items():
                    xm_img = data_processing.resolution_downward(data[i+1 , :, :], self.M_lats, self.M_lons,
                                                             self.G_lats, self.G_lons)
                    outx_table = np.concatenate((outx_table, xm_img.reshape(np.prod(xm_img.shape))), 1)

                for var, data in self.test_xme_data.items():
                    xme_img = data_processing.resolution_downward(data[i+ 1, :, :], self.Mete_lats, self.Mete_lons,
                                                              self.G_lats, self.G_lons)
                    outx_table = np.concatenate((outx_table, xme_img.reshape(np.prod(xme_img.shape))), 1)
            pred_seq_y[i, :, :] = prev_g_image
        seq_RMSE_mat, seq_R2_mat = self._evaluate(pred_seq_y, self.test_y_data)
        np.save('seq_RMSE_mat', seq_RMSE_mat)
        np.save('seq_R2_mat', seq_R2_mat)
        pred_seq_y.dump('seq_pred_y')
        self.test_y_data.dump('true_y')

if __name__ == "__main__":
    #file_path_g_06 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_06 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    #file_path_g_05 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    file_path_g_05 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    # file path of MERRA-2 data
    # file_path_m = '/scratch/menglinw/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_m = r'C:\Users\96349\Documents\Downscale_data\MERRA2\MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    #file_path_mete05 = '/scratch/menglinw/Downscale_data/METE/2005_mete_data.nc'
    #file_path_mete06 = '/scratch/menglinw/Downscale_data/METE/2006_mete_data.nc'
    #file_path_mete07 = '/scratch/menglinw/Downscale_data/METE/2007_mete_data.nc'

    file_path_mete05 = r'C:\Users\96349\Documents\Downscale_data\meteology_data/2005_mete_data.nc'
    file_path_mete06 = r'C:\Users\96349\Documents\Downscale_data\meteology_data/2006_mete_data.nc'
    file_path_mete07 = r'C:\Users\96349\Documents\Downscale_data\meteology_data/2007_mete_data.nc'
    model = mete_merra_to_g5nr_model(file_path_g_05, file_path_g_06, file_path_m, file_path_mete05, file_path_mete06,
                                     file_path_mete07, test_season=1)
    model.train(epoch=1)
    model.predict()



