#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import random
import datetime
from datetime import datetime, timedelta
from .calculate_correlation import prepare_date_range


# In[3]:


def produce_output_name(target_name):
    output_name = target_name.replace('tested_positive_', "")
    output_name = output_name.replace('_', '.')
    return output_name


# In[4]:


def get_feature_list(target_name, corr_start, corr_end):
    
    # read correlation list
    read_name = target_name.replace('tested_positive_', "")
    df_corr = pd.read_csv(f'./correlation/correlation_{read_name}_{corr_start}_{corr_end}.csv')
    feat_list = df_corr['features'].tolist()

    # add target
    feat_list.append(target_name)
    # print('feature number:', len(feat_list))
    
    return feat_list


# In[5]:


def merge_data(feat_list, target_name):
    
    # blank dataframe
    data = pd.DataFrame()

    # merge data
    for i in range(len(feat_list)):

        data_name = feat_list[i]

        # read data
        if data_name == target_name:
            dt = pd.read_csv(f'./data/official/smoothed/{data_name}.csv')
            dt['survey_date'] = pd.to_datetime(dt['survey_date'], format="%Y-%m-%d")  ## convert survey_date datatype to datatime
        else:
            dt = pd.read_csv(f'./data/UMD/smoothed/{data_name}.csv')
            dt.columns = [data_name] + list(dt.columns[1:])                            ## rename needed columns
            dt['survey_date'] = pd.to_datetime(dt['survey_date'], format="%Y%m%d")     ## convert survey_date datatype to datatime
            dt = dt[['survey_date', data_name]]                                        ## drop useless columns and rearrange

        # merge data into dataframe
        if i == 0:
            data = dt
        else:
            data = pd.merge(data, dt, on=['survey_date'])

    # convert survey_date datatype to datatime
    data['survey_date'] = pd.to_datetime(data['survey_date'], format="%Y%m%d")
    
    return data


# In[6]:


def save_data(data, data_final, day, output_name, train_num=20):
    
    # Produce index to drop
    index_list = list(range(0, train_num))
    pop_list = list(range(train_num, len(data_final)))

    # covid.train
    train = data_final
    index = pop_list
    drop = train.drop(index, inplace = False)   # delete rows out of boundary
    drop.index = range(len(drop))
    drop.reset_index(drop=True, inplace=True)   # reindex
    train_time = drop
    train_notime = drop.drop(columns = ['survey_date'])

    # covid.test
    test = data_final
    index = index_list
    drop = test.drop(index, inplace = False)    # delete rows out of boundary
    drop.index = range(len(drop))
    drop.reset_index(drop=True, inplace=True)   # reindex
    test_time = drop
    test_notime = drop.drop(columns = ['survey_date'])

    # download training data
    csv_path_training = './data/training'
    csv_path_withdate = './data/training/withdate'
    csv_path_all = './data/all'
    os.makedirs(csv_path_training, exist_ok = True)
    os.makedirs(csv_path_withdate, exist_ok = True)
    os.makedirs(csv_path_all, exist_ok = True)
    
    train_notime.to_csv(f'{csv_path_training}/covid.train.{day}day.{output_name}.csv')
    test_notime.to_csv(f'{csv_path_training}/covid.test.{day}day.{output_name}.csv')
    train_time.to_csv(f'{csv_path_withdate}/covid.train.{day}day.withdate.{output_name}.csv')
    test_time.to_csv(f'{csv_path_withdate}/covid.test.{day}day.withdate.{output_name}.csv')
    data_final.to_csv(f'{csv_path_all}/covid.{day}day.{output_name}.csv')
    data.to_csv(f'{csv_path_all}/all.{output_name}.csv')


# In[7]:


def build_df_of_day_num(data, day_num, output_name, train_num=20):
    
    # build blank dataframe
    data_final = pd.DataFrame()

    for k in range(len(day_num)):

        '''Seperate days'''
        for i in range(day_num[k]):

            # build data_day
            data_day = data

            # add suffix to columns
            data_day = data_day.add_suffix(f'_{i+1}')

            # rename survey_date and shifted_date
            data_day = data_day.rename(columns={f'survey_date_{i+1}': 'survey_date'})

            # shift datetime
            data_day['shifted_date'] = data_day.survey_date - timedelta(days=i)

            # name data_day
            globals()[f'data_{i+1}'] = data_day
        
        '''Merge data_day'''
        # build data_final
        data_final = data_1
        data_final = data_final.rename(columns={'survey_date_1': 'survey_date'})    # with survey_date

        for i in range(day_num[k]-1):
            data_day = globals()[f'data_{i+2}']
            data_day = data_day.drop(columns = ['survey_date'])                 # drop survey_date for other days
            data_final = pd.merge(data_final, data_day, on=['shifted_date'])    # merge by shifted date

        # drop shifted_date
        data_final = data_final.drop(columns = ['shifted_date'])

        # build dataframe with survey_date
        data_final_time = data_final
        globals()[f'data_{day_num[k]}_day_time'] = data_final_time

        # build dataframe without survey_date
        data_final = data_final.drop(columns = ['survey_date'])
        # locals()[f'data_{day_num[k]}_day'] = data_final    # rename data_day_num
        
        '''Save data'''
        save_data(data, data_final_time, day_num[k], output_name, train_num)


# In[8]:


def data_sort(date_start, train_num, mv_day, day_num):
    
    corr_start, corr_end = prepare_date_range(date_start, train_num)
    
    for i in range(len(mv_day)):
        target_name = f'official_tested_positive_num_smoothed_{mv_day[i]}d'        
        output_name = produce_output_name(target_name)
        feat_list = get_feature_list(target_name, corr_start, corr_end)
        data = merge_data(feat_list, target_name)
        build_df_of_day_num(data, day_num, output_name)

