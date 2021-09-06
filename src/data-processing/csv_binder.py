import json_to_csv
import pandas as pd
import glob
#import tweepy
import csv
import json
import pandas as pd
import sys

import os
src_path = '/home/paperspace/research/twitter/tw_covid_wfh/tweets_collectors/premium'
first_half_files = glob.glob(os.path.join(src_path, "*_2020-01-01_2020-06-30.jsonl"))
second_half_files = glob.glob(os.path.join(src_path, "*_2020-07-01_2020-12-31.jsonl"))
all_json_files = first_half_files + second_half_files
all_base_files = [os.path.basename(f) for f in all_json_files]
all_base_names = [fn.split('.')[0] for fn in all_base_files]

# TODO: create two functions: all_json_to_csv(), bind_csv() 
# all_json_to_csv use a for loop
# binder function use old notebook code
#src_json = 'chinese_epidemic_2020-01-01_2020-06-30.jsonl'
#dst_csv = 'chinese_epidemic_2020-01-01_2020-06-30.csv'

def all_json_to_csv(base_names, src_path, dst_path):
    for bn in base_names:
        
        src_json = bn + '.jsonl'
        dst_csv = bn + '.csv'

        #src_file = os.path.join(src_path, src_json)
        #dst_file = os.path.join(dst_path, dst_csv)

        json_to_csv.json_to_csv(src_dir = src_path
                , dst_dir = dst_path
                , json_file_name = src_json
                , csv_file_name = dst_csv
                )
    return None


def bind_all_csv(base_names, src_file_path, dest_file_name):
    li = []

    for fn in base_names:
        filename = os.path.join(src_file_path, fn) + '.csv'
        df = pd.read_csv(filename, index_col=None, header=0,lineterminator='\n')
        li.append(df)

    df_pool = pd.concat(li, axis=0, ignore_index=True)

    df_pool.to_csv(dest_file_name)
    print('combined {} files with total number or rows {}!'.format(len(base_names), df_pool.shape[0]))
    
    return None


if __name__ == "__main__":
    dst_path = '../../data'
    all_json_to_csv(base_names = all_base_names, src_path = src_path, dst_path = dst_path)
    dst_file_name = os.path.join(dst_path, 'asian_hate_raw_tweets_df.csv')
    bind_all_csv(base_names = all_base_names, src_file_path = dst_path, dest_file_name = dst_file_name)


