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
all_files = first_half_files + second_half_files
all_base_files = [os.path.basename(f) for f in all_files]
all_base_names = []


# use a for loop
src_json = 'chinese_epidemic_2020-01-01_2020-06-30.jsonl'
dst_path = '../../data'
dst_csv = 'chinese_epidemic_2020-01-01_2020-06-30.csv'

json_to_csv.json_to_csv(src_dir = src_path
        , dst_dir = dst_path
        , json_file_name = src_json
        , csv_file_name = dst_csv
        )

