# reference: https://www.tensorflow.org/official_models/fine_tuning_bert
# reference: https://github.com/sidmahurkar/BERT-Twitter-US-Airline-Sentiment-Analysis/blob/master/Bert_airline_senti_analysis.ipynb
# reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/#5-performance-on-test-set
# ENVIRONMENT SETUP
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from utils import get_auc, flat_accuracy, get_eval_report, compute_metrics, get_f1_score
import model_config as config
# SETUP GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# # READING PARAMETERS FROM COMMAND LINE
# import sys
#
# label_name = sys.argv[1]

#
# print("label name: " + str(label_name))
# print("learning rate: " + str(learning_rate))
# print("number of epoches: " + str(n_epoch))

def model_fine_tuning(src_train_file, model_output_dir, n_epoch = 5, rd_seed = 15213, save_model = False,):
    # READING TRAINING DATA FOR MODEL FINE-TUNING
    # Note: please use data after "upsampling" so that the binary labels have 50/50 of 1/0s.
    #src_train_file = '../../data/processed_annotations.csv'

    with open(src_train_file, 'rb') as filehandle:
        # store the data as binary data stream
        label_text = pickle.load(filehandle)
    print(label_text.head(10))

    # READING TRAINING DATA FOR MODEL FINE-TUNING
    # Pad sentences with start and end tokens used for BERT
    sentences = label_text.text.values
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

    print("enter tokenization")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print ("Tokenize the first sentence:")
    print (tokenized_texts[0])

    labels = label_text.num_label.values
    type(labels)
    type(labels[0])
    num_classes = label_text.num_label.nunique()

    # Maximum sentence length set to 256
    MAX_LEN = 256
    # Pad input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Use the BERT tokenizer to convert the tokens to corresponding index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    #Pad the sequences
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    print(input_ids[1])

    print("finished sequence padding")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                random_state=rd_seed, test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=rd_seed, test_size=0.2)

    print("Convert all of our data into torch tensors, the required datatype for our model")

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Select a batch size for training. For fine-tuning BERT on a specific task - based on the GPU environment
    print("set batch size")
    batch_size = 32

    # Create an iterator with torch DataLoader
    torch.manual_seed(15213)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    print("finished batch setup")

    # MODEL TRAINING
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
    model.cuda()

    print("finished loading bert-base-uncased")

    # Setup Adam optimizer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    print("finished setting different weight decays for different layers of the model")

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=.1)

    print("finished setting BertAdam hyperparameters")
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (recommend between 2 and 5)
    epochs = n_epoch

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        preds_epoch = []
        labels_epoch = []
        logits_epoch = []
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
              # Forward pass, calculate logit predictions
              logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            preds_flat = np.argmax(logits, axis=1).flatten()
            preds_epoch.extend(preds_flat)
            labels_epoch.extend(label_ids)
            logits_epoch.extend(logits)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        accuracy = eval_accuracy/nb_eval_steps
        print("Validation Accuracy: {}".format(accuracy))
        print("Validation Accuracy: {}".format(accuracy))
        micro_f1, sep_f1s = get_f1_score(preds_epoch, labels_epoch)
        print("Micro F1 Score: {}".format(micro_f1))
        print(sep_f1s)
        print(len(logits_epoch))
        print(len(logits_epoch[1]))
        print(len(labels_epoch))
        roc = get_auc(logits_epoch, labels_epoch, classes = [0,1,2])
        print("ROC: {}".format(roc))

    if save_model:
        from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
        #output_dir = "model_output"

        # Step 1: Save a model, configuration and vocabulary that you have fine-tuned
        # If we have a distributed model, save only the encapsulated model
        # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = model.module if hasattr(model, 'module') else model

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(model_output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(model_output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(model_output_dir)
    return micro_f1, accuracy, roc


if __name__ == "__main__":
    learning_rate = float(sys.argv[1])
    n_epoch = int(sys.argv[2])
    split_random_int = 1111

    f1, acc, roc = model_fine_tuning(src_train_file = config.SOURCE_TRAINING_FILE
                                     , model_output_dir = config.MODEL_OUTPUT_DIR
                                     , n_epoch=5, rd_seed=15213, save_model=True
                                     )
    print(f1)
    print(acc)
    print(roc)
