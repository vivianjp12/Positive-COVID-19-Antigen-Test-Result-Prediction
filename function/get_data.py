#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import requests
import json
import math
import os


# In[2]:


def get_official_data(save_file = True):
    
    # fetch data from official website
    response = requests.get("https://covid-19.nchc.org.tw/api/covid19?CK=covid-19@nchc.org.tw&querydata=4001&limited=TWN").text
    
    # transfer json to pandas
    jsonData = json.loads(response)
    df = pd.DataFrame.from_dict(jsonData)
    
    # rename columns
    df.rename(columns = {"id":"ID","a01":"iso_code","a02":"洲名","a03":"國家","a04":"日期","a05":"總確診數","a06":"新增確診數","a07":"七天移動平均新增確診數","a08":"總死亡數","a09":"新增死亡數","a10":"七天移動平均新增死亡數","a11":"每百萬人確診數","a12":"每百萬人死亡數","a13":"傳染率","a14":"新增檢驗件數","a15":"總檢驗件數","a16":"每千人檢驗件數","a17":"七天移動平均新增檢驗件數","a18":"陽性率","a19":"每確診案例相對檢驗數量","a20":"疫苗總接種總劑數","a21":"疫苗總接種人數","a22":"疫苗新增接種劑數","a23":"七天移動平均疫苗新增接種劑數","a24":"每百人接種疫苗劑數","a25":"每百人接種疫苗人數","a26":"疫情控管指數","a27":"總人口數","a28":"中位數年紀","a29":"70歲以上人口比例","a30":"平均壽命","a31":"解除隔離數","a32":"解封指數"}, inplace = True)
    
    # keep useable columns and rename into English
    df = df[["日期", "新增確診數"]]
    df = df.rename(columns = {"日期":"survey_date", "新增確診數":"official_tested_positive_num"})
    
    # sort by survey date
    df["survey_date"] = pd.to_datetime(df["survey_date"], format = "%Y-%m-%d")
    df = df.sort_values(by = "survey_date")
    df = df.reset_index(drop = True)
    
    # save file
    if save_file == True:

        # save as csv
        csv_path = 'data/official/raw'
        os.makedirs(csv_path, exist_ok = True)        
        df.to_csv(f'{csv_path}/official_tested_positive_num_raw.csv', index = False)
    
    return df


# In[ ]:


def smooth_data(df, target, mv_day=[1,7,14], save_file = True, scale = True, official = True):
    
    date = df['survey_date'].tolist()
    data = df[target].tolist()

    for i in range(len(mv_day)):

        day_num = mv_day[i]
        mv_date = date[day_num-1:]
        mv_data = []
        
        # calculate moving average
        for j in range(len(data)-day_num+1):
            mv_data_value = np.mean(list(map(int, data[j:j+day_num])))
            mv_data.append(mv_data_value)
        
        # scale value
        if scale == True:
            if max(mv_data) >= 1:
                d = int(math.log10(max(mv_data)))+1
                mv_data = [i * (0.1 ** d) for i in mv_data]
                mv_data = [round(i, d) for i in mv_data]
        
        # build dataframe
        locals()[f'data_smoothed_{day_num}d'] = pd.DataFrame({'survey_date': mv_date, f'{target}_smoothed_{day_num}d': mv_data})
        
        # save file
        if save_file == True:
            csv_path = 'data/official/smoothed' if official == True else 'data/smoothed'
            os.makedirs(csv_path, exist_ok = True)
            locals()[f'data_smoothed_{day_num}d'].to_csv(f'{csv_path}/{target}_smoothed_{day_num}d.csv', index = False)


# In[ ]:


def get_UMD_data(date_start, date_end, dtype = 'smoothed', save_file = True):
    
    # get feature list
    feat_df = pd.read_csv('features.csv')
    feat = feat_df['features'].tolist()
    
    data_ext = []
    data_err = []
    data_else = []
    
    for i in range(len(feat)):

        # request data from api
        response = requests.get(f"https://covidmap.umd.edu/api/resources?indicator={feat[i]}&type={dtype}&country=Taiwan&daterange={date_start}-{date_end}").text

        # convert json data to dic data for use!
        jsonData = json.loads(response)

        # convert to pandas dataframe
        if 'data' in jsonData:     # if data exists
            df = pd.DataFrame.from_dict(jsonData['data'])
            if df.empty:
                data_else.append(feat[i])     # check if df is empty
            else:
                data_ext.append(feat[i])
                
                # save file
                if save_file == True:
                    csv_path = f'data/UMD/{dtype}'
                    os.makedirs(csv_path, exist_ok = True)
                    df.to_csv(f'{csv_path}/{feat[i]}.csv', index = False)

            # print(feat[i], ':\n', df, '\n')

        elif 'error' in jsonData:     # if error exists
            data_err.append(feat[i])
            # print(feat[i], ': error\n')

        else:
            data_else.append(feat[i])
            # print(feat[i], ': else\n')
    
    # print('How many data exists: ', len(data_ext))
    # print('How many data error: ', len(data_err))
    # print('How many data left: ', len(data_else))
    # print('Data exists: ', data_ext)
    # print('Data error: ', data_err)
    # print('Data left: ', data_else)


# In[ ]:


def get_data(date_start, date_end, mv_day, smoothed=True):
    
    # get official data
    official_data = get_official_data()
    if smoothed == True:
        smooth_data(official_data, 'official_tested_positive_num', mv_day = mv_day)
    
    # get UMD data
    get_UMD_data(date_start, date_end)

