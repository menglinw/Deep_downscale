import h5py
import numpy as np
import os
import time


def normalize_h5(file_path_h5):
    train_file = h5py.File(os.path.join(file_path_h5, 'Train_data.h5'), 'r')
    norm_train_file = h5py.File(os.path.join(file_path_h5, 'Norm_train_data.h5'), 'w')
    norm_train_file.create_dataset('Y', data=train_file['Y'][:], maxshape=(train_file['Y'].shape))
    # f.create_dataset('Y', data=outy, chunks=True, maxshape=(None, 1))
    for i in range(24):
        if i == 0:
            norm_train_file.create_dataset('X', data=train_file['X'][:, i], chunks=True,
                             maxshape=(train_file['X'].shape[0], None))
        else:
            norm_train_file['X'].resize((norm_train_file['X'].shape[1] + 1), axis=1)
            cur_col = train_file['X'][:, i]
            norm_train_file['X'][:,-1] = (cur_col - cur_col.min())/(cur_col.max() - cur_col.min())
    print('Finished', file_path_h5)



if __name__ == '__main__':
    start = time.time()
    file_path_list_h5 = ['/scratch/menglinw/Downscale_data/FLAT_DATA/data1',
                         '/scratch/menglinw/Downscale_data/FLAT_DATA/data2',
                         '/scratch/menglinw/Downscale_data/FLAT_DATA/data3']
    for file_path in file_path_list_h5:
        normalize_h5(file_path)
    print('Duriation', (time.time() - start)/60, 'mins')