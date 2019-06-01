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
import argparse
import data.data_utils as utils
import json
from models.word_cnn_attr import WordCNNPIG
import tensorflow as tf
import os


def parse_arguments():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps',
                      type=int,
                      default=50,
                      help='Interpolation steps in pig.')
  parser.add_argument('--target_label_index',
                      type=int,
                      default=1,
                      help='Class to explain with pig attributions.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=100,
                      help='Mini batch size for evaluating test set.')
  parser.add_argument('--model_dir',
                      type=str,
                      required=True,
                      help='Directory of train output.')
  parser.add_argument('--sent_filter', nargs='+',
                      default=[],
                      help='Only exaplin on sentences with these words.')
  parser.add_argument('--eval_data', 
                      type=str,
                      default='./data/wiki/wiki_dev.txt',
                      help='Dataset to evaluate/expalin on.')
  parser.add_argument('--pred_output',
                      type=str,
                      default='pred.txt',
                      help='Prediction filename.')
  parser.add_argument('--target_words_file',
                      type=str,
                      default='',
                      help='Words to minimize attribution/replace with special token.')
  parser.add_argument('--target_words_to_token',
                      action='store_true',
                      help='If specified, replace target words with special token.')
  return parser.parse_known_args()

def write_list_to_file(l, path):
  with open(path, 'w') as f:
    for item in l:
        f.write("%s\n" % item)

def add_attributions(total_attr, tok_count, attr, x, idx2word):
  num_instance = x.shape[0]
  seq_len = x.shape[1]
  for i in range(num_instance):
    seen_tok = set()
    for j in range(seq_len):
      tok = idx2word[x[i][j]]
      if tok == '<pad>':
        break
      if tok not in total_attr:
        total_attr[tok] = 0.0
      total_attr[tok] += attr[i][j]
      if tok in seen_tok:
        continue
      seen_tok.add(tok)
      if tok not in tok_count:
        tok_count[tok] = 0
      tok_count[tok] += 1
     

def explain(x_test, y_test, idx2word, word2idx, train_args, args):
  seq_len = train_args.seq_len
  embedding_size = train_args.embedding_size
  num_filters = train_args.num_filters  
  num_classes = train_args.num_classes
  filter_sizes = train_args.filter_sizes
  checkpoint = os.path.join(args.model_dir, 'model_best.ckpt')
  steps = args.steps
  batch_size=args.batch_size
  target_label_index=args.target_label_index
  pred_path = os.path.join(args.model_dir, args.pred_output)
  baseline = utils.get_all_pad(seq_len, word2idx)

  graph = tf.Graph()
  with graph.as_default():
    model = WordCNNPIG(seq_len, num_classes, len(idx2word), embedding_size, filter_sizes, num_filters)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, checkpoint)
      # Calculate accuracy and attributions on test set
      sum_accuracy, cnt = 0, 0
      test_batches = utils.batch_iter(x_test, y_test, batch_size, 1)
      predictions = []
      total_attr = {}
      tok_count = {}
      for x_test_batch, y_test_batch in test_batches:
        dev_feed_dict = {
          model.x: x_test_batch,
          model.y: y_test_batch,
          model.label_index: target_label_index,
          model.keep_prob: 1,
          model.baseline: baseline,
        }
        pred, accuracy, attr = sess.run([model.softmax, model.accuracy, model.sum_intergrated_grad], feed_dict=dev_feed_dict)
        predictions.extend(pred[:,1].tolist())
        add_attributions(total_attr, tok_count, attr, x_test_batch, idx2word)
        sum_accuracy += accuracy
        cnt += 1
      
      # Test accuracy  
      test_accuracy = sum_accuracy / cnt
      write_list_to_file(predictions, pred_path)
      print("\nTest Accuracy = {0}\n".format(test_accuracy))
      
      # Calculate attributions
      print("Global Attributions:")
      global_attr = {}
      for tok in total_attr:
        global_attr[tok] = total_attr[tok] / tok_count[tok]
      sorted_by_value = sorted(global_attr.items(), key=lambda kv: -kv[1])
      for k in sorted_by_value:
        print(k[0], k[1])

def main(argv=None):
  args, _ = parse_arguments()    

  train_args_path = os.path.join(args.model_dir, 'args.json')
  with open(train_args_path, 'r') as f:
    train_args_dict = json.load(f)
  train_args = argparse.Namespace(**train_args_dict)

  vocab_path = os.path.join(args.model_dir, 'vocab.txt')
  idx2word, word2idx = utils.load_vocab(vocab_path)

  target_words = []
  if args.target_words_to_token and args.target_words_file != '':
    with open(args.target_words_file) as f:
      content = f.readlines()
      target_words = [x.strip() for x in content]

  x_dev, y_dev = utils.preprocess(args.eval_data, train_args.seq_len, word2idx, sent_filter=args.sent_filter, target_words=target_words, target_words_to_token=args.target_words_to_token)
  explain(x_dev, y_dev, idx2word, word2idx, train_args, args)

if __name__ == '__main__':
    tf.app.run()
