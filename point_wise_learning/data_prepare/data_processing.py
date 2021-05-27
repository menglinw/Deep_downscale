import os
#os.environ['PROJ_LIB'] = r"C:\Users\96349\anaconda3\Lib\site-packages\mpl_toolkits\basemap"
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import sys
#sys.path.append(r'C:\Users\96349\OneDrive - University of Southern California\Desktop\Downscaling_Project\AERONET')
import numpy as np
import netCDF4 as nc



'''
def plot_image(single_image_data, title, MERRA_data, G5NR_data, is_merra = False):
    if is_merra:
        lons = MERRA_data.variables['lon'][:]
        lats = MERRA_data.variables['lat'][:]
    else:
        lons = G5NR_data.variables['lon'][:]
        lats = G5NR_data.variables['lat'][:]
    m = Basemap(projection='merc',llcrnrlon=25.,llcrnrlat=10.,urcrnrlon=75.,urcrnrlat=45. , resolution='h', epsg = 4326)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    parallels = np.arange(10,43,5.) # make latitude lines ever 5 degrees from 30N-50N
    meridians = np.arange(25,75,5.) # make longitude lines every 5 degrees from 95W to 70W
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    #m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)
    m.shadedrelief(scale=0.5)
    m.pcolormesh(lons, lats, single_image_data, latlon=True)
    plt.clim(0, single_image_data.max())
    m.drawcoastlines(color='lightgray')
    plt.title(title)
    plt.colorbar()
    plt.savefig(title+'.jpg')
'''

def resolution_downward(image, M_lats, M_lons, G_lats, G_lons):
    '''

    :param image:
    :param M_lats:
    :param M_lons:
    :param G_lats:
    :param G_lons:
    :return:
    '''
    if image.shape != (len(M_lats), len(M_lons)):
        print('Please check your input image')
        raise ValueError
    lat_gap = abs((M_lats[1]-M_lats[0])/2)
    lon_gap = abs((M_lons[1]-M_lons[0])/2)
    M_high_image = np.zeros((499, 788))
    for i in range(len(M_lats)):
        for j in range(len(M_lons)):
            aod = image[i, j]
            min_lat = np.argmin( np.abs( G_lats - M_lats[i] + lat_gap ) ) if i != 0 else 0
            max_lat = np.argmin( np.abs( G_lats - M_lats[i] - lat_gap ) ) if i != len(M_lats)-1 else len(G_lats)
            min_lon = np.argmin( np.abs( G_lons - M_lons[j] + lon_gap ) ) if j != 0 else 0
            max_lon = np.argmin( np.abs( G_lons - M_lons[j] - lon_gap) ) if j != len(M_lons)-1 else len(G_lons)
            # print('lat:', min_lat, max_lat, '  lon:', min_lon, max_lon)
            # print(M_high_image[min_lat:max_lat, min_lon:max_lon].shape)
            M_high_image[min_lat:max_lat, min_lon:max_lon] = aod
    return M_high_image


def image_to_table(image, elev, lats, lons, day):
    '''

    :param image: AOD data
    :param elev: elevation data
    :param lats: latitude
    :param lons: longitude
    :param day: day
    :return:
    (np.prod(image.shape), 5)
    5 columns: AOD, latitude, longitude, day, elevation,
    '''
    if image.shape != (len(lats), len(lons)):
        print('please check data consistency!')
        raise ValueError
    out_array = np.zeros((len(lats)*len(lons), 5))
    out_array[:, 0] = image.reshape(len(lats)*len(lons))
    lat_in = []
    for lat in lats:
        lat_in += [lat]*len(lons)
    out_array[:, 1] = lat_in
    out_array[:, 2] = list(lons)*len(lats)
    out_array[:, 3] = float(day)
    out_array[:, 4] = elev.reshape(len(lats)*len(lons))
    return out_array

if __name__=='__main__':
    # define some universal  variables
    # file path of G5NR 06-07, and 05-06
    file_path_g_06 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    # file path of MERRA-2 data
    file_path_m = r'C:\Users\96349\Documents\Downscale_data\MERRA2\MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_ele = r'C:\Users\96349\Documents\Downscale_data\elevation\elevation_data.npy'
    # target variable
    target_var = 'TOTEXTTAU'
    # take a sample image from G5NR and MERRA-2 respectively
    # 2005/05/16
    g05_data = nc.Dataset(file_path_g_05)
    sample_G_image = g05_data.variables[target_var][0]
    m_data = nc.Dataset(file_path_m)
    sample_M_image = m_data.variables[target_var][1825]
    elev_data = np.load(file_path_ele)
    M_lons = m_data.variables['lon'][:]
    M_lats = m_data.variables['lat'][:]

    G_lons = g05_data.variables['lon'][:]
    G_lats = g05_data.variables['lat'][:]

    d_image = resolution_downward(sample_M_image, M_lats, M_lons, G_lats, G_lons)
    table_test = image_to_table(sample_G_image, elev_data, G_lats, G_lons, 0)
    print('the shape of table is:', table_test.shape)
    print(table_test[499*788-10:])