import numpy as np
import re
import math
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, confusion_matrix, f1_score, precision_recall_curve, average_precision_score, recall_score
from sklearn.preprocessing import label_binarize

hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")

def process_text(text):
  text = hashtags.sub(' hashtag', text)
  text = mentions.sub(' entity', text)
  text = urls.sub(' url', text)
  return text.strip().lower()

def match_expr(pattern, string):
  return not pattern.search(string) == None

def get_data_wo_urls(dataset):
  link_with_urls = dataset.text.apply(lambda x: match_expr(urls, x))
  return dataset[[not e for e in link_with_urls]]

def flat_accuracy(logits, labels):
  pred_flat = np.argmax(logits, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_f1_score(preds, labels):
    micro_f1 = f1_score(labels, preds, average='micro')
    sep_f1s = f1_score(labels, preds, average=None)
    return micro_f1, sep_f1s

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_auc(logits, y_label, classes = [0,1,2]):
    y_pred = [list(map(sigmoid, tup)) for tup in logits]
    y = label_binarize(y_label, classes=classes)
    n_classes = len(classes)
    y = np.array(y)
    y_pred = np.array(y_pred)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    recall_score = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y[:, i], y_pred[:, i])
        recall_score[i] = recall_score(y[:, i], y_pred[:, i])

    return roc_auc, precision, recall, average_precision, recall_score

def get_eval_report(labels, preds):
  mcc = matthews_corrcoef(labels, preds)
  tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
  return {
    "mcc": mcc,
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn
  }

def compute_metrics(labels, preds):
  assert len(preds) == len(labels)
  return get_eval_report(labels, preds)
