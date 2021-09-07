## Preporcess tweets data before labeling. Keep distinct idstr and only qualified tweets
import csv
import pandas as pd
import sys
import os

# TODO: 1. keep distinct idstr; 2. keep only US tweets where full_text is not null
def keep_qualified(data):
    if 'retweeted' in data.columns:
        output = data[data.retweeted.eq(False)]
    if 'Unnamed: 0' in data.columns:
        output = output.drop('Unnamed: 0', 1)
    return output

def remove_duplicates(data, key):
    return data.drop_duplicates(subset=key, keep="first")

def full_text_imputing(data):
    if 'text' in data.columns and 'full_text' in data.columns:
        texts = data.text.values
        full_texts = data.full_text.values
        for i in range(len(texts)):
            if full_texts[i] == 'na':
                full_texts[i] = texts[i]
        data['full_text'] == full_texts
    return data

if __name__ == "__main__":
    raw_tweets_path = '../../data'
    raw_tweets_path = 'asian_hate_raw_tweets_df.csv'
    raw_tweets = pd.read_csv(os.path.join(raw_tweets_path, raw_tweets_path))
    print('Start preprocessing...')
    print('original number of tweets: {}').format(raw_tweets.shape[0])
    output = keep_qualified(raw_tweets)
    print('filtering qualified tweets: {}').format(output.shape[0])
    output = remove_duplicates(output)
    print('After removing duplicated idstrs, {} tweets were kept').format(output.shape[0])
    output = full_text_imputing(output)
    output_filename = 'asian_hate_processe_tweets_dataframe.csv'
    output.to_csv(os.path.join(raw_tweets_path, output_filename))



