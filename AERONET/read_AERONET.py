# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:08:03 2020

@author: 96349
"""


import pandas as pd
import numpy as np
import os

def read_aeronet(filename):
    """
    Read a given AERONET AOT data file, and return it as a dataframe.
    
    This returns a DataFrame containing the AERONET data, with the index
    set to the timestamp of the AERONET observations. Rows or columns
    consisting entirely of missing data are removed. All other columns
    are left as-is.
    """
    dateparse = lambda x: pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
    aeronet = pd.read_csv(filename, skiprows=6, na_values=['N/A'],
                          parse_dates={'times':[0,1]},
                          date_parser=dateparse)

    aeronet = aeronet.set_index('times')
    
    # Drop any rows that are all NaN and any cols that are all NaN
    # & then sort by the index
    an = (aeronet.dropna(axis=1, how='all')
                .dropna(axis=0, how='all')
                .rename(columns={'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date'})
                .sort_index())
    return an

def aeronet_daily_avg(aeronet_path, aeronet_file_name, target_year, target_var):
    '''

    Parameters
    ----------
    aeronet_path : string
        path to aeronet data
    aeronet_file_name : string
        aeronet data file name
    target_year : int
        the year you want to investigate
    target_var : string
        the variable you want to average

    Returns
    -------
    a series of daily averaged target_var on target year

    '''
    aeronet_dat = read_aeronet(os.path.join(aeronet_path, aeronet_file_name))
    # subset the target year data
    sub_aeronet_dat = aeronet_dat[(aeronet_dat.index >= '%d-05-16' % target_year)& (aeronet_dat.index < '%d-05-16' % (target_year + 1))]
    if sub_aeronet_dat.shape[0] != 0:
        sub_aeronet_dat = sub_aeronet_dat[target_var][sub_aeronet_dat[target_var] != -999]
        daily_sub_aeronet_dat = sub_aeronet_dat.groupby([sub_aeronet_dat.index.date]).mean()
        return(daily_sub_aeronet_dat)
    
def site_missing_check(aeronet_path, aeronet_file_name, target_year):
   
    '''
    Parameters
    ----------
    aeronet_path : string
        path of aeronet data
    aeronet_file_name : string
        file name of aeronte data
    target_year : TYPE
        the year you want to exam

    Returns
    -------
    A dataframe of count of dates that have valid data and percentage correspondingly

    '''
    aeronet_dat = read_aeronet(os.path.join(aeronet_path, aeronet_file_name))
    # subset the target year data
    sub_aeronet_dat = aeronet_dat[((aeronet_dat.index.year == target_year)&(aeronet_dat['Day_of_Year'] > 136)) |
                              ((aeronet_dat.index.year == target_year + 1)&(aeronet_dat['Day_of_Year'] <= 136))]
    missing_dic = {}
    for col in sub_aeronet_dat.columns[2:]:
        non_empt_dat = sub_aeronet_dat[col][sub_aeronet_dat[col]!= -999]
        _ , counts = np.unique(non_empt_dat.index.date, return_counts=True)
        counts = counts.shape[0]
        if counts != 0:
            missing_dic[col] = [counts, counts/365]
    return(pd.DataFrame(missing_dic, index= ['Date counts','Date %']).T)

def aeronet_filename_to_latlon( aeronet_path, file_name):
    '''
    take aeronet file name and return it lat long information

    Parameters
    ----------
    aeronet_path : string
        path to aeronet data
    file_name : string
        aeronet data file name

    Returns
    -------
    latitude and longitude

    '''
    site_info = pd.read_csv(os.path.join(aeronet_path, 'AERONET_SITE_latlon.csv'))
    site_name = file_name[18:22]
    lat, lon = site_info.loc[site_info['Site_name'] == site_name, ['lat','lon']].iloc[0,:]
    return((lat, lon))

def aeronet_missing_check(aeronet_path, target_year):
    '''
    

    Parameters
    ----------
    aeronet_path : TYPE
        DESCRIPTION.
    target_year : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    aeronet_files = os.listdir(aeronet_path)
    del aeronet_files[aeronet_files.index('AERONET_SITE.xlsx')]
    del aeronet_files[aeronet_files.index('AERONET_SITE_latlon.csv')]
    aeronet_sum_dic =[]
    for aeronet_file in aeronet_files:
        print('processing', aeronet_file)
        sum_tab = site_missing_check(aeronet_path, aeronet_file, target_year)
        aeronet_sum_dic.append([aeronet_file, sum_tab['Date %'].max()])
    aeronet_sum = pd.DataFrame(aeronet_sum_dic, columns=['Site', 'Max date %'])
    return(aeronet_sum)