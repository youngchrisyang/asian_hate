# model
ASIAN_HATE_SOURCE_RAW_TRAINING_FILE = '../../data/raw_training_data/annotations.csv'
SOURCE_TRAINING_FILE = '../../data/processed_annotations.csv'
MODEL_OUTPUT_DIR = 'pretrained_model_output/'

SENTIMENT_SOURCE_RAW_TRAINING_FILE = '../../data/processed_sentiment140_60k_sample.csv'
SENTIMENT_SOURCE_PROCESSED_TRAINING_FILE = '../../data/raw_training_data/sentiment140_60k_sample.csv'
SENTIMENT_MODEL_OUTPUT_DIR = 'pretrained_sentiment_model_output/'

# inference
SOURCE_INFERENCE_ASIAN_HATE_FILE = '../../data/inference_data/asian_hate_processe_tweets_dataframe.csv'
LABELED_DATA_OUTPUT_FILE = '../../data/inference_data/asian_hate_predicted_hate.csv'
LABELED_SENTIMENT_DATA_OUTPUT_FILE = '../../data/inference_data/asian_hate_predicted_binary_sentiment.csv'
