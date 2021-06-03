import numpy as np
import netCDF4 as nc
import random
import data_processing
import keras
import keras.layers as layers
from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
import matplotlib.pyplot as plt

class within_merra_model():
    def __init__(self, file_path_m, file_path_elev, merra_var, test_days=365, target_variable='TOTEXTTAU', perm=True):
        self.file_path_elev = file_path_elev
        self.test_days = test_days
        self.file_path_m = file_path_m
        self.target_var = target_variable
        self.merra_var = merra_var
        self.perm = perm

        # load MERRA-2 data
        self.m_data = nc.Dataset(file_path_m)
        self.m_target_data = self.m_data.variables[self.target_var][:, :, :]
        M_lons = self.m_data.variables['lon'][:]
        self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        M_lats = self.m_data.variables['lat'][:]
        self.M_lats = (M_lats-M_lats.mean())/M_lats.std()

        # load elevation data
        self.elev = self._elev_upscale()

        # split train/test set
        train_x_index, train_y_index, test_x_index, test_y_index = self._random_split()

        self.train_x, self.train_y = self._flatten(train_x_index, train_y_index)
        self.test_x, self.test_y = self._flatten(test_x_index, test_y_index)

        # define model
        self.model = self.define_model()

    def _random_split(self, seed=2021):
        random.seed(seed)
        test_days = self.test_days
        test_index_start = random.sample(range(2, (self.m_target_data.shape[0]-test_days-1)), 1)[0]
        test_index = list(range(test_index_start-1, test_index_start+test_days))
        test_x = np.array(test_index[:-1])
        test_y = np.array(test_index[1:])
        train_x = np.array(list(range(test_index_start)) +
                           list(range(test_index_start+test_days, self.m_target_data.shape[0]-1)))
        train_y = train_x + 1
        return train_x, train_y, test_x, test_y

    def _elev_upscale(self):
        elev_org_data = np.load(self.file_path_elev)
        lat_step = int(elev_org_data.shape[0]/len(self.M_lats))
        lon_step = int(elev_org_data.shape[1]/len(self.M_lons))
        out_elev = np.zeros((len(self.M_lats), len(self.M_lons)))
        for i in range(len(self.M_lats)):
            for j in range(len(self.M_lons)):
                lat_start = i*lat_step
                lat_end = (i+1)*lat_step if i != len(self.M_lats)-1 else elev_org_data.shape[0]
                lon_start = j*lon_step
                lon_end = (j+1)*lon_step if j != len(self.M_lons)-1 else elev_org_data.shape[1]
                out_elev[i, j] = np.mean(elev_org_data[lat_start:lat_end, lon_start:lon_end])
        return out_elev

    def _flatten(self, x_index, y_index):
        '''
        :return:
        a flatten table, not permuted the variables are:
        latitude, longitude, day, elevation, mera_vars
        '''
        single_img_size = np.prod(self.m_target_data.shape[1:])
        x = np.zeros((len(x_index)*single_img_size, 4 + len(self.merra_var)))
        y = np.zeros((len(y_index)*single_img_size, 1))
        for i, day in enumerate(x_index):

            outx_table = data_processing.image_to_table(self.m_target_data[day, :, :], self.M_lats, self.M_lons,
                                                        (day % 365) / 365, self.elev)[:, 1:]
            # print(outx_table.shape)
            for var, data in self.m_data.variables.items():
                if var in self.merra_var:
                    # print(var)
                    outx_table = np.concatenate((outx_table, data[day].reshape((single_img_size, 1))), 1)

            outy = self.m_target_data[y_index[i]].reshape((single_img_size, 1))
            # print(outx_table.shape)
            x[i*single_img_size:(i+1)*single_img_size] = outx_table
            y[i*single_img_size:(i+1)*single_img_size] = outy
        for j in range(4 + len(self.merra_var)):
            cur_col = x[:, j]
            x[:, j] = (cur_col - cur_col.min())/(cur_col.max()-cur_col.min())

        if self.perm:
            perm = np.random.permutation(len(x_index)*single_img_size)
            return x[perm], y[perm]
        else:
            return x, y

    def train(self, epochs=100):
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='within_merra_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=2
            )
        ]
        history = self.model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                                 batch_size=256, epochs=epochs, callbacks=callbacks_list)
        fig = plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label="train loss")
        plt.plot(history.history['val_loss'], label='test loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss_curve.jpg')

    def define_model(self):
        input = Input(shape=(4 + len(self.merra_var)))
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


if __name__ == '__main__':
    #file_path_elev = r'C:\Users\96349\Documents\Downscale_data\elevation\elevation_data.npy'
    #file_path_m = r'C:\Users\96349\Documents\Downscale_data\MERRA2\MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    #'''
    file_path_elev = '/scratch/menglinw/Downscale_data/ELEV/elevation_data.npy'
    file_path_m = '/scratch/menglinw/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    #'''
    merra_var = ['BCEXTTAU', 'DUEXTTAU', 'OCEXTTAU', 'SUEXTTAU', 'TOTEXTTAU', 'BCSMASS', 'DUSMASS25', 'DUSMASS',
                 'OCSMASS', 'SO4SMASS', 'SSSMASS']

    model = within_merra_model(file_path_m, file_path_elev, merra_var=merra_var, test_days=4*365, target_variable='TOTEXTTAU')
    model.train(100)