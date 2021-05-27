from PIL import Image
import numpy as np
import os


def read_elevation(path):
    im = Image.open(path)
    pix = np.array(im.getdata()).reshape(im.height, im.width)
    return pix


def match_to_low_resolution(ele_data, target_shape=(499, 788)):
    out_ele = np.zeros(target_shape)
    lat_step = int(ele_data.shape[0]/target_shape[0])
    lon_step = int(ele_data.shape[1]/target_shape[1])
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            lat_start = i*lat_step
            if i == target_shape[0] - 1:
                lat_end = ele_data.shape[0]
            else:
                lat_end = (i+1)*lat_step

            lon_start = j*lon_step
            if j == target_shape[1] - 1:
                lon_end = ele_data.shape[1]
            else:
                lon_end = (j+1)*lon_step
            out_ele[i,j] = np.mean(ele_data[lat_start:lat_end, lon_start:lon_end])
    return out_ele

def normalize(ele_data):
    min_val = np.min(ele_data)
    max_val = np.max(ele_data)
    return (ele_data - min_val)/(max_val - min_val)

if __name__ == '__main__':
    file_path_ele = r'C:\Users\96349\Documents\Downscale_data\elevation'
    '''
    # the final image are construct by 6 image of elevation
    # GMTED2010N10E000, lat: (30, 10), lon: (0, 30)
    subimg_4 = read_elevation(os.path.join(file_path_ele,'GMTED2010N10E000'+'_300', '10n000e_20101117_gmted_mea300.tif'))
    # we need lon [25,30]
    # shape is (3600, 400)
    subimg_4 = subimg_4[:, 2000:]

    # GMTED2010N10E030, lat: (30, 10), lon: (30, 60)
    # we need all, shape is (3600, 2400)
    subimg_5 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N10E030' + '_300', '10n030e_20101117_gmted_mea300.tif'))

    # GMTED2010N10E060, lat: (30, 10), lon: (60, 90)
    subimg_6 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N10E060' + '_300', '10n060e_20101117_gmted_mea300.tif'))
    # we need lon [60, 75]
    # shape is (3600, 1200)
    subimg_6 = subimg_6[:, :1200]

    # GMTED2010N30E000, lat: (50, 30), lon: (0, 30)
    subimg_1 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N30E000' + '_300', '30n000e_20101117_gmted_mea300.tif'))
    # we need lat [43, 30], lon [25:30]
    # shape: (2340, 400)
    subimg_1 = subimg_1[1260:, 2000:]

    # GMTED2010N30E030, lat: (50, 30), lon: (30, 60)
    subimg_2 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N30E030' + '_300', '30n030e_20101117_gmted_mea300.tif'))
    # we need lat [43, 30], lon [30, 60]
    # shape: (2340, 2400)
    subimg_2 = subimg_2[1260:, :]

    # GMTED2010N30E060, lat: (50, 30), lon: (60, 90)
    subimg_3 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N30E060' + '_300', '30n060e_20101117_gmted_mea300.tif'))
    # we need lat [43, 30], lon [60, 75]
    # shape: (2340, 1200)
    subimg_3 = subimg_3[1260:, :1200]
    '''
    ### (2400 (height), 3600(width))
    # the final image are construct by 6 image of elevation
    # GMTED2010N10E000, lat: (30, 10), lon: (0, 30)
    subimg_4 = read_elevation(os.path.join(file_path_ele,'GMTED2010N10E000'+'_300', '10n000e_20101117_gmted_mea300.tif'))
    # we need lon [25,30]
    # shape is (3600, 400)
    subimg_4 = subimg_4[:, 3000:]

    # GMTED2010N10E030, lat: (30, 10), lon: (30, 60)
    # we need all, shape is (3600, 2400)
    subimg_5 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N10E030' + '_300', '10n030e_20101117_gmted_mea300.tif'))

    # GMTED2010N10E060, lat: (30, 10), lon: (60, 90)
    subimg_6 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N10E060' + '_300', '10n060e_20101117_gmted_mea300.tif'))
    # we need lon [60, 75]
    # shape is (3600, 1200)
    subimg_6 = subimg_6[:, :1800]

    # GMTED2010N30E000, lat: (50, 30), lon: (0, 30)
    subimg_1 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N30E000' + '_300', '30n000e_20101117_gmted_mea300.tif'))
    # we need lat [43, 30], lon [25:30]
    # shape: (2340, 400)
    subimg_1 = subimg_1[840:, 3000:]

    # GMTED2010N30E030, lat: (50, 30), lon: (30, 60)
    subimg_2 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N30E030' + '_300', '30n030e_20101117_gmted_mea300.tif'))
    # we need lat [43, 30], lon [30, 60]
    # shape: (2340, 2400)
    subimg_2 = subimg_2[840:, :]

    # GMTED2010N30E060, lat: (50, 30), lon: (60, 90)
    subimg_3 = read_elevation(
        os.path.join(file_path_ele, 'GMTED2010N30E060' + '_300', '30n060e_20101117_gmted_mea300.tif'))
    # we need lat [43, 30], lon [60, 75]
    # shape: (2340, 1200)
    subimg_3 = subimg_3[840:, :1800]
    upper_part = np.concatenate((subimg_1, subimg_2, subimg_3), axis=1)
    lower_part = np.concatenate((subimg_4, subimg_5, subimg_6), axis=1)
    ele_data = np.concatenate((upper_part, lower_part), axis=0)
    # plot final elevation plot
    #im = Image.fromarray(ele_data)
    #im.show()
    low_resl_data = match_to_low_resolution(ele_data)
    low_resl_data = normalize(low_resl_data)
    np.save(r'C:\Users\96349\Documents\Downscale_data\elevation\elevation_data.npy', low_resl_data)


