from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from profile_in_json import get_model_inference_profile
import wget
import os
import pandas as pd
import zipfile

# The URL for the dataset zip file.
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
# download and unzip the dataset
if not os.path.exists('./cola_public/'):
    wget.download(url, './cola_public_1.1.zip')
    with zipfile.ZipFile('./cola_public_1.1.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

def bert_input_constructor(seq_len, tokenizer):
    #Loading the test data and applying the same preprocessing techniques which we performed on the train data

    # Load the dataset into a pandas dataframe.
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    # Create sentence and label lists
    sentences = list(df.sentence.values)
    labels = list(df.label.values)

    # For every sentence...
    for sent in sentences:
        for _ in range(seq_len - (2 + len(sent))):
            sent += tokenizer.pad_token

    inputs = tokenizer(sentences,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    inputs = dict(inputs)
    labels = torch.tensor(labels)
    inputs.update({"labels": labels})
    return inputs

with get_accelerator().device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4
    seq_len = 256
    enable_profile = True
    dataset = bert_input_constructor(seq_len, tokenizer)
    print(dataset)
    print(len(dataset["labels"]))
    if enable_profile:
      flops, macs, params = get_model_inference_profile(
          model,
          kwargs=dataset,
          print_profile=True,
          detailed=True,
          output_file="./bert_profile.json"
      )
    else:
      inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
      outputs = model(inputs)
