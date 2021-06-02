import h5py
import numpy as np
import os
import time


def normalize_h5(file_path_h5):
    start = time.time()
    train_file = h5py.File(os.path.join(file_path_h5, 'Train_data.h5'), 'r')
    norm_train_file = h5py.File(os.path.join(file_path_h5, 'Norm_train_data.h5'), 'w')
    norm_train_file.create_dataset('Y', data=train_file['Y'][:], maxshape=(train_file['Y'].shape))

    test_file = h5py.File(os.path.join(file_path_h5, 'Test_data.h5'), 'r')
    norm_test_file = h5py.File(os.path.join(file_path_h5, 'Norm_test_data.h5'), 'w')
    norm_test_file.create_dataset('Y', data=test_file['Y'][:], maxshape=(test_file['Y'].shape))
    # f.create_dataset('Y', data=outy, chunks=True, maxshape=(None, 1))
    print('Start doing', file_path_h5)
    for i in range(24):
        print('processing the', i+1, 'column')
        if i == 0:
            norm_train_file.create_dataset('X', data=train_file['X'][:, i].reshape((train_file['X'].shape[0],1)),
                                           chunks=True, maxshape=(train_file['X'].shape[0], None))

            norm_test_file.create_dataset('X', data=test_file['X'][:, i].reshape((test_file['X'].shape[0], 1)),
                                           chunks=True, maxshape=(test_file['X'].shape[0], None))
        else:
            norm_train_file['X'].resize((norm_train_file['X'].shape[1] + 1), axis=1)
            cur_col = train_file['X'][:, i]
            norm_train_file['X'][:,-1] = (cur_col - cur_col.min())/(cur_col.max() - cur_col.min())

            norm_test_file['X'].resize((norm_test_file['X'].shape[1] + 1), axis=1)
            cur_col = test_file['X'][:, i]
            norm_test_file['X'][:, -1] = (cur_col - cur_col.min())/(cur_col.max() - cur_col.min())
    train_file.close()
    norm_train_file.close()
    test_file.close()
    norm_test_file.close()
    print('Finished', file_path_h5)
    print('Duriation', (time.time() - start) / 60, 'mins')



if __name__ == '__main__':

    file_path_list_h5 = ['/scratch/menglinw/Downscale_data/FLAT_DATA/data1',
                         '/scratch/menglinw/Downscale_data/FLAT_DATA/data2',
                         '/scratch/menglinw/Downscale_data/FLAT_DATA/data3']
    for file_path in file_path_list_h5:
        normalize_h5(file_path)
