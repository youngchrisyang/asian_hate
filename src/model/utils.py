import numpy as np
import re
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score

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
