import numpy as np
import netCDF4 as nc
import data_processing
import datetime
import time
import h5py


class raw_to_h5():
    def __init__(self, file_path_g_05, file_path_g_06, file_path_m, file_path_mete05, file_path_mete06,
                 file_path_mete07, file_path_ele, target_variable='TOTEXTTAU' ,merra_var=['TOTEXTTAU'], mete_var=['u10', 'msl'], test_season=1):
        # define path and target variable
        self.test_season = test_season
        self.file_path_g_05 = file_path_g_05
        self.file_path_g_06 = file_path_g_06
        self.file_path_m = file_path_m
        self.file_path_mete05 = file_path_mete05
        self.file_path_mete06 = file_path_mete06
        self.file_path_mete07 = file_path_mete07
        self.file_path_ele = file_path_ele
        self.target_var = target_variable
        self.merra_var = merra_var
        self.mete_var = mete_var

        # read g5nr data
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        self.g_data = np.concatenate((self._data_g5nr_to_array(g05_data), self._data_g5nr_to_array(g06_data)), axis=0)

        # read merra data as nc
        self.m_data = nc.Dataset(file_path_m)
        # self.m_data = m_data.variables[self.target_var][range(1826, 1826+730), :, :]

        # read mete data as nc
        self.mete05_data = nc.Dataset(file_path_mete05)
        self.mete06_data = nc.Dataset(file_path_mete06)
        self.mete07_data = nc.Dataset(file_path_mete07)

        # read elevation data as array
        self.ele_data = np.load(self.file_path_ele)

        # define lat&lon of MERRA, G5NR and mete
        self.M_lons = self.m_data.variables['lon'][:]
        # self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        self.M_lats = self.m_data.variables['lat'][:]
        # self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
        self.G_lons = g05_data.variables['lon'][:]
        # self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
        self.G_lats = g05_data.variables['lat'][:]
        # self.G_lats = (G_lats-G_lats.mean())/G_lats.std()
        self.Mete_lons = self.mete05_data.variables['longitude'][:]
        self.Mete_lats = self.mete05_data.variables['latitude'][:]


    def process(self):
        # processing data
        self._mete_temp_avg()
        self._train_test_split()
        self._flatten(self.train_xg_data, self.train_xm_data, self.train_xme_data,
                      self.train_y_data, self.xday_list, h5_name='Train_data.h5')
        self._flatten(self.test_xg_data, self.test_xm_data, self.test_xme_data,
                      self.test_y_data, self.yday_list, h5_name='Test_data.h5')



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
        start_hour = start_day * 24
        for var in self.mete05_data.variables.keys():
            if var not in ['longitude', 'latitude', 'time']:
                time_len, lat_len, lon_len = self.mete05_data.variables[var].shape
                temp_data = np.zeros((int((time_len - start_hour) / 24), lat_len, lon_len))
                for i in range(int((time_len - start_hour) / 24)):
                    temp_data[i, :, :] = np.mean(
                        self.mete05_data.variables[var][start_hour + i * 24:(i * 24 + 24 + start_hour)])
                self.Mete_data[var] = temp_data

        # 2006 take full year
        for var in self.mete06_data.variables.keys():
            if var not in ['longitude', 'latitude', 'time']:
                time_len, lat_len, lon_len = self.mete06_data.variables[var].shape
                temp_data = np.zeros((int(time_len / 24), lat_len, lon_len))
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

    def _train_test_split(self):
        '''
        self.g_data: ndarray
        self.mete_data: dict
        self.m_data: nc
        '''
        test_season = self.test_season
        test_index = np.array(range((1 + (test_season - 1) * 3) * 73 - 2, (1 + test_season * 3) * 73 - 1))
        self.test_index = test_index
        train_x_index = np.array(list(set(range(729)) - set(test_index)))
        self.xday_list = train_x_index
        self.yday_list = test_index[:-1]
        train_y_index = train_x_index + 1
        # train raw data
        self.train_xg_data = self.g_data[train_x_index, :, :]

        self.train_xm_data = dict()
        for var in self.m_data.variables.keys():
            if var not in ['lat', 'lon', 'time']:
                self.train_xm_data[var] = self.m_data.variables[var][train_x_index, :, :]

        self.train_xme_data = dict()
        for var, data in self.Mete_data.items():
            self.train_xme_data[var] = data[train_x_index, :, :]
        self.train_y_data = self.g_data[train_y_index, :, :]

        # test raw data
        self.test_xg_data = self.g_data[test_index[:-1], :, :]

        self.test_xm_data = dict()
        for var in self.m_data.variables.keys():
            if var not in ['lat', 'lon', 'time']:
                self.test_xm_data[var] = self.m_data.variables[var][test_index[:-1], :, :]

        self.test_xme_data = dict()
        for var, data in self.Mete_data.items():
            self.test_xme_data[var] = data[test_index[:-1], :, :]
        self.test_y_data = self.g_data[test_index[1:], :, :]


    def _generator(self, xg_data, xm_data, xme_data, yg_data, day_list):
        while True:
            for i, day in enumerate(day_list):
                permutation = np.random.permutation(np.prod(yg_data[i].shape))
                outx_table = data_processing.image_to_table(xg_data[i, :, :],
                                                            (self.G_lats - self.G_lats.mean()) / self.G_lats.std(),
                                                            (self.G_lons - self.G_lons.mean()) / self.G_lons.std(),
                                                            (day % 365) / 365)
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

    def _flatten(self, xg_data, xm_data, xme_data, yg_data, day_list, h5_name):
        '''

        :param xg_data: G5NR data X
        :param xm_data: merra2 data
        :param xme_data: meteological data
        :param yg_data: G5NR data Y
        :param day_list: X day index (not Y)
        :param permut: permutation of the output
        :return:
        a flatten table, not permuted the variables are:
        AOD, latitude, longitude, day, elevation, mera_vars, mete_vars
        '''

        #x = np.zeros((np.prod(xg_data.shape), 5 + len(self.merra_var) + len(self.mete_var)))
        #y = np.zeros((np.prod(xg_data.shape), 1))
        f = h5py.File(h5_name, 'a')
        for i, day in enumerate(day_list):

            outx_table = data_processing.image_to_table(xg_data[i, :, :], self.ele_data,
                                                        (self.G_lats - self.G_lats.mean()) / self.G_lats.std(),
                                                        (self.G_lons - self.G_lons.mean()) / self.G_lons.std(),
                                                        (day % 365) / 365)
            for var, data in xm_data.items():
                if var in self.merra_var:
                    xm_img = data_processing.resolution_downward(data[i, :, :], self.M_lats, self.M_lons,
                                                                 self.G_lats, self.G_lons)
                    outx_table = np.concatenate((outx_table, xm_img.reshape((np.prod(xm_img.shape), 1))), 1)

            for var, data in xme_data.items():
                if var in self.mete_var:
                    xme_img = data_processing.resolution_downward(data[i, :, :], self.Mete_lats, self.Mete_lons,
                                                                  self.G_lats, self.G_lons)
                    outx_table = np.concatenate((outx_table, xme_img.reshape((np.prod(xme_img.shape), 1))), 1)
            outy = yg_data[i].reshape((np.prod(yg_data[i].shape), 1))
            if i == 0:
                f.create_dataset('X', data=outx_table, chunks=True,
                                 maxshape=(None, 5 + len(self.merra_var) + len(self.mete_var)))
                f.create_dataset('Y', data=outy, chunks=True, maxshape=(None, 1))
            else:
                f['X'].resize((f['X'].shape[0] + outx_table.shape[0]), axis=0)
                f['X'][-outx_table.shape[0]:] = outx_table

                f['Y'].resize((f['Y'].shape[0] + outy.shape[0]), axis=0)
                f['Y'][-outy.shape[0]:] = outy
            print('Processing day: ', i, '/', len(day_list), 'Shape of X and Y', outx_table.shape, outy.shape)
        f.close()






if __name__ == "__main__":
    start = time.time()
    #'''
    file_path_g_06 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/scratch/menglinw/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    file_path_m = '/scratch/menglinw/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_mete05 = '/scratch/menglinw/Downscale_data/METE/2005_mete_data.nc'
    file_path_mete06 = '/scratch/menglinw/Downscale_data/METE/2006_mete_data.nc'
    file_path_mete07 = '/scratch/menglinw/Downscale_data/METE/2007_mete_data.nc'
    file_path_ele = '/scratch/menglinw/Downscale_data/ELEV/elevation_data.npy'
    #'''


    #file_path_g_06 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    #file_path_g_05 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    #file_path_m = r'C:\Users\96349\Documents\Downscale_data\MERRA2\MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    #file_path_mete05 = r'C:\Users\96349\Documents\Downscale_data\meteology_data\2005_mete_data.nc'
    #file_path_mete06 = r'C:\Users\96349\Documents\Downscale_data\meteology_data\2006_mete_data.nc'
    #file_path_mete07 = r'C:\Users\96349\Documents\Downscale_data\meteology_data\2007_mete_data.nc'
    #file_path_ele = r'C:\Users\96349\Documents\Downscale_data\elevation\elevation_data.npy'

    merra_var = ['BCEXTTAU', 'DUEXTTAU', 'OCEXTTAU', 'SUEXTTAU', 'TOTEXTTAU', 'BCSMASS', 'DUSMASS25', 'DUSMASS',
                 'OCSMASS', 'SO4SMASS', 'SSSMASS']
    mete_var = ['u10', 'v10', 'd2m', 't2m', 'blh', 'uvb', 'msl', 'tp']

    data_processor = raw_to_h5(file_path_g_05, file_path_g_06, file_path_m, file_path_mete05, file_path_mete06,
                 file_path_mete07, file_path_ele, merra_var=merra_var, mete_var=mete_var, test_season=1)
    data_processor.process()
    print('Duriation:', time.time() - start)