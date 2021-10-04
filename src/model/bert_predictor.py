# TODO: need to preprocess texts exactly the same way as you process training data

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
import pandas as pd
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import model_config as config

# start GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)


def inference_data_preprocessing(data_source_file):
    id_text = pd.read_csv(data_source_file)
    print('Read raw data: ' + str(id_text.shape))
    id_text = id_text[['id_str', 'full_text']]
    id_text.rename(columns={'full_text': 'text'}, inplace=True)
    id_text['fake_id'] = np.arange(len(id_text))
    print(id_text.head(10))
    return id_text


def divide_chunks(l, n):
    # Due to memory limit, divide and conqure
    # looping till length l 
    for i in range(0, len(l), n):
        yield l[i:i + n]


def inference(data, data_output_file, pretrained_model_dir, num_classes, chunk_size=20000):
    chunk_indices = list(divide_chunks(list(range(data.shape[0])), chunk_size))
    print('Splitted into ' + str(len(chunk_indices)) + ' chunks with chunck length: ' + str(len(chunk_indices[0])))

    chk_cnt = 1
    final_output_predictions = []
    for chk_idx in chunk_indices:
        chk_id_text = id_text.iloc[chk_idx]
        print('Using Data from Chunk ' + str(chk_cnt) + " with nrows: " + str(chk_id_text.shape[0]))
        chk_cnt += 1

        fake_ids = chk_id_text.fake_id.values
        fake_ids = [int(i) for i in fake_ids]
        # print(fake_ids)

        sentences = chk_id_text.text.values
        sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

        # Reload pretrained models
        # model_to_save.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(output_dir)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir,
                                                  do_lower_case=True)  # Add specific options if needed

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        print ("Tokenize the first sentence:")
        print (tokenized_texts[0])

        MAX_LEN = 256
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        # Pad the sequences
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        print(input_ids[0])
        print("finished sequence padding")

        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        test_inputs = torch.tensor(input_ids)
        test_masks = torch.tensor(attention_masks)
        fake_ids = torch.Tensor(fake_ids)
        print("set batch size")
        batch_size = 24

        test_data = TensorDataset(test_inputs, test_masks, fake_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        print("finished setting batch size")

        model = BertForSequenceClassification.from_pretrained(pretrained_model_dir, num_labels=num_classes)
        model.cuda()

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Predict data by minibatch
        output_predictions = []
        batch_cnt = 0
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_idstrs = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            b_fake_ids = b_idstrs.to('cpu').numpy()
            b_fake_ids = [int(id) for id in b_fake_ids]
            print(b_fake_ids)
            preds = np.argmax(logits, axis=1).flatten()
            d = {'fake_id': b_fake_ids, 'logit': preds}
            df = pd.DataFrame(d)
            convert_dict = {'fake_id': int, 'logit': int}
            df = df.astype(convert_dict)
            output_predictions.append(df)
            batch_cnt += 1
            if batch_cnt % 100 == 0:
                print('Processed number of batches:' + str(batch_cnt))

        output_predictions = pd.concat(output_predictions)

        output_predictions = pd.merge(chk_id_text, output_predictions, on='fake_id')

        print(output_predictions.shape)
        print(output_predictions.head(10))

        final_output_predictions.append(output_predictions)

    # After DnC
    final_output_predictions = pd.concat(final_output_predictions)
    final_output_predictions.to_csv(data_output_file, index=False)

    print('Finshed processing ' + str(final_output_predictions.shape[0]) + ' tweets!')

    return


if __name__ == "__main__":
    inference_id_text = inference_data_preprocessing(config.SOURCE_INFERENCE_ASIAN_HATE_FILE)
    inference( data = inference_id_text
               , data_output_file = config.LABELED_DATA_OUTPUT_FILE
               , pretrained_model_dir = config.MODEL_OUTPUT_DIR
               , num_classes = 3
               , chunk_size=20000)
