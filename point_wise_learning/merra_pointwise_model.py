import numpy as np
import nc_process
import netCDF4 as nc
import random
from data_prepare import data_processing
import keras.layers as layers
from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
import matplotlib.pyplot as plt

"""
use MERRA-2 data to train a pointwise model, which will be fed into final model to stablize the spatialtemporal association
input data: flatten MERRA-2 data (random split parameter: test size)
output: a loss-curve plot, final model
"""
class merra_pointwise():
    def __init__(self, file_path_m, test_days):
        self.test_days = test_days
        self.file_path_m = file_path_m
        self.target_var = 'TOTEXTTAU'
        m_data = nc.Dataset(file_path_m)
        self.m_data = m_data.variables[self.target_var][:,:,:]
        M_lons = m_data.variables['lon'][:]
        self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        M_lats = m_data.variables['lat'][:]
        self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
        # define model
        self.model = self.define_model()

    def random_split(self, test_days=365, seed=2021):
        random.seed(seed)
        test_index = random.sample(range(1, (self.m_data.shape[0]-1)), int(test_days))
        train_index = list(set(list(range(self.m_data.shape[0]-1))) - set(test_index))
        return np.array(train_index), np.array(test_index)

    def raw_to_table(self, x_data, y_data):
        if x_data.shape != y_data.shape:
            print('Please check the data consistency!')
            raise ValueError
        train_x_table = np.zeros((np.prod(x_data.shape), 4))
        for i in range(x_data.shape[0]):
            train_x_table[i*np.prod(x_data.shape[1:]):(i+1)*np.prod(x_data.shape[1:])] = data_processing.\
                image_to_table(x_data[i], self.M_lats, self.M_lons, (i%365)/365)
        train_y_table = y_data.reshape(np.prod(y_data.shape))
        return train_x_table, train_y_table

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

    def define_model(self):
        input = Input(shape=(4))
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

    def train(self):
        train_x_index, test_y_index = self.random_split(test_days=self.test_days)
        train_y_index = train_x_index+1
        test_x_index = test_y_index-1
        #print(train_y_index)
        # train raw data
        train_xm_data = self.m_data[train_x_index,:,:]
        train_y_data_3d = self.m_data[train_y_index,:,:]
        train_x_data, train_y_data = self.raw_to_table(train_xm_data,train_y_data_3d)
        # test raw data
        '''
        test_xm_data = self.m_data[test_x_index,:,:]
        self.test_y_data_3d = self.m_data[test_y_index,:,:]
        self.test_x_data, test_y_data = self.raw_to_table(test_xm_data, self.test_y_data_3d)
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
        '''
        history = self.model.fit(train_x_data, train_y_data, batch_size=499, epochs=50)
        self.model.save('pointwise_1_randomsplit')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label="train loss")
        # plt.plot(history.history['val_loss'], label='test loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss_curve.jpg')

if __name__=="__main__":
    # file path of MERRA-2 data
    file_path_m = '/scratch/menglinw/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    #file_path_m = r'C:\Users\96349\Documents\Downscale_data\MERRA2\MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    model = merra_pointwise(file_path_m, 0)
    model.train()
