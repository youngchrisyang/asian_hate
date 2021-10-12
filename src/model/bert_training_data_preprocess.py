# ETL to construct task specific training dataset to fine-tune BERT
# Goal: collect all labled data from SemEval2018 in the text-label format
import pandas as pd
import os
from utils import process_text, get_data_wo_urls
import pickle
import model_config as config
# src_dir = '../../data/raw_training_data'
# src_filename = 'annotations.csv'
#
# dst_dir = '../../data'
# dst_filename = 'processed_annotations.csv'


def get_dir(label):
    src_filename = ''
    dst_filename = ''
    if label == 'asian_hate':
        src_filename = config.ASIAN_HATE_SOURCE_RAW_TRAINING_FILE
        dst_filename = config.SOURCE_TRAINING_FILE
    if label == 'sentiment':
        src_filename = config.SENTIMENT_SOURCE_RAW_TRAINING_FILE
        dst_filename = config.SENTIMENT_SOURCE_PROCESSED_TRAINING_FILE
    return src_filename, dst_filename


if __name__ == "__main__":
    import sys
    label_type = sys.argv[1]

    src_filename, dst_filename = get_dir(label=label_type)

    if label_type == 'asian_hate':
        train = pd.read_csv(src_filename, header=None, sep=",", encoding='latin')
        train.columns = ['text', 'label']
        train = train[1:]
        train = train[train['label']!='Non-Asian Aggression']
        train.text = train.text.apply(process_text)
        train['num_label'] = train.label.map({'Neutral':0, 'Hate':1, 'Counterhate':2, 'Non-Asian Aggression':1})

        train['num_label'] = train['num_label'].astype('int')

    if label_type == 'sentiment':
        train = pd.read_csv(src_filename, header=None, sep=",", encoding='latin')
        train.columns = ['text', 'label']
        train = train[1:]
        train.text = train.text.apply(process_text)
        train.rename(columns={'label': 'num_label'}, inplace=True)
        train['num_label'] = train['num_label'].astype('int')

    with open(dst_filename, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train, filehandle)

    #train.to_csv(os.path.join(dst_dir, dst_filename), index = False)
    print('Finished processing raw training data with {} rows'.format(train.shape[0]))

