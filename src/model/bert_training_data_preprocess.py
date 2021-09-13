# ETL to construct task specific training dataset to fine-tune BERT
# Goal: collect all labled data from SemEval2018 in the text-label format
import pandas as pd
import os
from utils import process_text, get_data_wo_urls
import pickle

src_dir = '../../data/raw_training_data'
src_filename = 'annotations.csv'

dst_dir = '../../data'
dst_filename = 'processed_annotations.csv'


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(src_dir, src_filename), header=None, sep=",", encoding='latin')
    train.columns = ['text','label']
    train = train[1:]
    train.text = train.text.apply(process_text)
    train['num_label'] = train.label.map({'Neutral':0, 'Hate':1, 'Counterhate':2, 'Non-Asian Aggression':1})
    
    train['num_label'] = train['num_label'].astype('int')

    with open(os.path.join(dst_dir, dst_filename), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train, filehandle)

    #train.to_csv(os.path.join(dst_dir, dst_filename), index = False)
    print('Finished processing raw training data with {} rows'.format(train.shape[0]))

