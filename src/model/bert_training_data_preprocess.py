# ETL to construct task specific training dataset to fine-tune BERT
# Goal: collect all labled data from SemEval2018 in the text-label format
import numpy as np
import pandas as pd
import os
from utils import process_text, get_data_wo_urls
import sys

label_name = sys.argv[1]
label_names = ['anger', 'anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']
label_index = label_names.index(label_name) + 2
print(label_index)

input_dir = "~/Documents/twitter/data/affect_data/SemEval2018-Task1-all-data/English/E-c"

train_name = "2018-E-c-En-train.txt"
dev_name = "2018-E-c-En-dev.txt"
test_name = "2018-E-c-En-test-gold.txt"

train_file = os.path.join(input_dir, train_name)
dev_file = os.path.join(input_dir, dev_name)
test_file = os.path.join(input_dir, test_name)
data_output_file = "~/Documents/twitter/data/affect_data/e_c_{}.csv".format(label_name)

train = pd.read_csv(train_file, header=None, sep="\t",encoding='latin')
dev = pd.read_csv(dev_file, header=None, sep="\t",encoding='latin')
test = pd.read_csv(test_file, header=None, sep="\t",encoding='latin')

train = train.loc[1:]
dev = dev.loc[1:]
test = test.loc[1:]

df = train.append([dev, test])
print(df.head(5))

label_text = df[[1, label_index]]
#label_text[3] = label_text[3].apply(lambda x: x[0])

# Convert labels to range 0-1
#label_text[0] = label_text[0].apply(lambda x: 0 if x == 0 else 1)

# Assign proper column names to labels
label_text.columns = ['text','label']

# Assign proper column names to labels
label_text.head()

# import re
#
# hashtags = re.compile(r"^#\S+|\s#\S+")
# mentions = re.compile(r"^@\S+|\s@\S+")
# urls = re.compile(r"https?://\S+")

# def process_text(text):
#   text = hashtags.sub(' hashtag', text)
#   text = mentions.sub(' entity', text)
#   return text.strip().lower()
#
# def match_expr(pattern, string):
#   return not pattern.search(string) == None

# def get_data_wo_urls(dataset):
#     link_with_urls = dataset.text.apply(lambda x: match_expr(urls, x))
#     return dataset[[not e for e in link_with_urls]]


label_text.text = label_text.text.apply(process_text)
label_text.text = label_text.text.apply(get_data_wo_urls)


print(label_text.head())
print(label_text.shape)


label_text.to_csv(data_output_file, index = False)
