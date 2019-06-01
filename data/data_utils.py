# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import re

UNK = '<unk>'
PAD = '<pad>'
TARGET = '<target>'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(data_file, sent_filter=[]):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(data_file, "r", encoding='utf-8').readlines())
    x_text = []
    labels = []
    for i, s in enumerate(examples):
      split = s.strip().split("|||")
      label = split[0]
      text = "".join(split[1:])
      cleaned_str = clean_str(text)
      # Only include words in sent_filter
      if len(sent_filter) > 0 and set(sent_filter).isdisjoint(cleaned_str.split()):
        continue
      x_text.append(clean_str(text))
      labels.append(int(label))
    y = np.array(labels)
    return x_text, y

def get_word_count(corpus):
  word_count = {}
  for sentence in corpus:
    words = sentence.split()
    for word in words:
      if word not in word_count:
        word_count[word] = 0
      word_count[word] += 1
  return word_count

def write_list_to_file(l, filename):
  with open(filename, 'w') as f:
    for item in l:
        f.write("%s\n" % item)

def build_vocab(data_path, min_freq, output_path, target_words=None):
  data = list(open(data_path, "r", encoding='utf-8').readlines())
  text = [clean_str(line.strip().split("|||")[1]) for line in data]
  word_count = get_word_count(text)
  idx2word = [UNK, PAD]
  word2idx = {UNK : 0, PAD : 1}
  if target_words:
    idx2word.append(TARGET)
    word2idx[TARGET] = 3
  for word in word_count:
    if target_words and word in target_words:
      continue
    if word_count[word] >= min_freq:
      idx2word.append(word)   
      word2idx[word] = len(idx2word) - 1

  write_list_to_file(idx2word, output_path)
  return idx2word, word2idx
 
def load_vocab(vocab_file):
  with open(vocab_file) as f:
    lines = f.readlines() 
  idx2word = []
  word2idx = {}
  for i, line in enumerate(lines):
    vocab = line.strip()
    idx2word.append(vocab)
    word2idx[vocab] = i
  return idx2word, word2idx

def text2idx(text, word2idx, max_len, target_words=[]):
  ret = []
  for sentence in text:
    words = sentence.split()
    idx = []
    for i, word in enumerate(words):
      if len(idx) >= max_len:
        break
      if word in word2idx:
        idx.append(word2idx[word])
      elif word in target_words:
        idx.append(word2idx[TARGET])
      else:
        idx.append(word2idx[UNK])
    for i in range(len(idx), max_len):
      idx.append(word2idx[PAD])
    ret.append(idx)
  return np.array(ret)

def get_all_pad(length, word2idx):
  return np.array([[word2idx[PAD] for i in range(length)]])
    
def batch_iter(inputs, outputs, batch_size, num_epochs, shuffle=False, seed=None):
    if seed is not None:
      np.random.seed(seed)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
      if shuffle:
        p = np.random.permutation(len(inputs))
        inputs = inputs[p]
        outputs = outputs[p]
      for batch_num in range(num_batches_per_epoch):
          start_index = batch_num * batch_size
          end_index = min((batch_num + 1) * batch_size, len(inputs))
          yield inputs[start_index:end_index], outputs[start_index:end_index]

def batch_iter_attr(inputs, outputs, attr_target, target_words_mask, batch_size, num_epochs, shuffle=False, seed=None):
    if seed is not None:
      np.random.seed(seed)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    attr_target = np.array(attr_target)
    target_words_mask = np.array(target_words_mask)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
      if shuffle:
        p = np.random.permutation(len(inputs))
        inputs = inputs[p]
        outputs = outputs[p]
        attr_target = attr_target[p]
        target_words_mask = target_words_mask[p]
      for batch_num in range(num_batches_per_epoch):
          start_index = batch_num * batch_size
          end_index = min((batch_num + 1) * batch_size, len(inputs))
          yield inputs[start_index:end_index], outputs[start_index:end_index], attr_target[start_index:end_index], target_words_mask[start_index:end_index]

def preprocess(data_file, max_len, word2idx, target_words=None, return_attr_target=False, sent_filter=[], target_words_to_token=False):
  x_text, y = load_data_and_labels(data_file, sent_filter)
  if target_words_to_token and target_words:
    x = text2idx(x_text, word2idx, max_len, target_words)
  else:
    x = text2idx(x_text, word2idx, max_len)
  if target_words is not None and return_attr_target:
    target_idx = [word2idx[word] for word in target_words if word in word2idx]
    attr_target = np.zeros(x.shape)
    attr_target[np.isin(x, target_idx)] = 1.0
    attr_target = -1 * (attr_target - 1)
    return x, y, attr_target
  return x, y

