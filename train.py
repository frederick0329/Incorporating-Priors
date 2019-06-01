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

import argparse
import data.data_utils as utils
import datetime
import json
import numpy as np
import os
import tensorflow as tf
from models.word_cnn_attr import WordCNNPIG


def parse_arguments():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--min_freq',
                      type=int,
                      default=5,
                      help='Words below this frequency will be replaced with <unk> token.')
  parser.add_argument('--seq_len',
                      type=int,
                      default=100,
                      help='Length to pad ot truncate to.')
  parser.add_argument('--epochs',
                      type=int,
                      default=10,
                      help='Number of epochs to train.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=128,
                      help='Size of mini batch.')
  parser.add_argument('--random_seed',
                      type=int,
                      default=1234,
                      help='Random seed.')
  parser.add_argument('--num_classes',
                      type=int,
                      required=True,
                      help='Number of classed.')
  parser.add_argument('--num_filters',
                      type=int,
                      default=128,
                      help='Number of classed.')
  parser.add_argument('--embedding_size',
                      type=int,
                      default=128,
                      help='Size of word embedding.')
  parser.add_argument('--learning_rate',
                      type=float,
                      default=0.001,
                      help='Learning rate.')
  parser.add_argument('--dropout_keep',
                      type=float,
                      default=0.8,
                      help='Dropout (keep prob).')
  parser.add_argument('--output_dir',
                      type=str,
                      required=True,
                      help='Directory to keep the best model.')
  parser.add_argument('--optim',
                      type=str,
                      default="joint",
                      help='Loss to optimize. options: [joint, clf, attr, importance]')
  parser.add_argument('--attr_loss_weight',
                      type=float,
                      default=1000000.0,
                      help='Attribution Loss Weight.')
  parser.add_argument('--importance_weight',
                      type=float,
                      default=1.0,
                      help='Importance Loss Weight.')
  parser.add_argument('--filter_sizes', nargs='+',
                      default=[2,3,4], 
                      help='Conv filter sizes.')
  parser.add_argument('--target_words_file',
                      type=str,
                      default='',
                      help='Words to minimize attribution.')
  parser.add_argument('--train_data',
                      type=str,
                      default='./data/wiki/wiki_train.txt',
                      help='Dataset to train on.')
  parser.add_argument('--dev_data',
                      type=str,
                      default='./data/wiki/wiki_dev.txt',
                      help='Dataset to evaluate on.')
  parser.add_argument('--target_label_index',
                      type=int,
                      default=1,
                      help='Class to explain with pig attributions.')
  parser.add_argument('--load_dir',
                      type=str,
                      default='',
                      help='It provided, load model parameters from this directory.')
  parser.add_argument('--target_words_to_token',
                      action='store_true',
                      help='If specified, replace target words with special token.')
  parser.add_argument('--reward',
                      action='store_true',
                      help='If specified, encourage model attribute to the tokens.')
  return parser.parse_known_args()

def train(x_train, y_train, attr_target, x_dev, y_dev, idx2word, word2idx, target_words_mask, args):
  epochs = args.epochs
  batch_size = args.batch_size
  output_dir = args.output_dir
  seed = args.random_seed
  seq_len = args.seq_len
  num_classes = args.num_classes
  embedding_size = args.embedding_size
  num_filters = args.num_filters
  learning_rate = args.learning_rate
  filter_sizes = args.filter_sizes
  optim = args.optim
  attr_loss_weight = args.attr_loss_weight
  importance_weight = args.importance_weight
  label_index = args.target_label_index
  keep_prob = args.dropout_keep
  load_dir = args.load_dir
  reward = args.reward
  checkpoint = os.path.join(load_dir, 'model_best.ckpt')


  graph = tf.Graph()
  with graph.as_default():
    tf.set_random_seed(seed)
    print(datetime.datetime.now(), " Start building model...")
    model = WordCNNPIG(seq_len, num_classes, len(idx2word), embedding_size, 
                       filter_sizes, num_filters, learning_rate, 
                       attr_loss_weight=attr_loss_weight, 
                       reward=reward, importance_weight=importance_weight)
    
    saver = tf.train.Saver() 
    with tf.Session() as sess:
      if load_dir:
        saver.restore(sess, checkpoint)
      sess.run(tf.global_variables_initializer())
      best_accuracy = 0
      baseline = utils.get_all_pad(seq_len, word2idx)
      if optim == "joint":
        print("Training with joint loss...")
        optim_op = model.joint_optimizer
        loss_op = model.loss
      elif optim == "attr":
        print("Training with attribution loss...")
        optim_op = model.attr_optimizer
        loss_op = model.attr_loss
      elif optim == "clf":
        print("Training with classification loss...")
        optim_op = model.clf_optimizer
        loss_op = model.clf_loss
      elif optim == "importance":
        print("Training with importance loss...")
        optim_op = model.importance_optimizer
        loss_op = model.importance_loss
      else:
        raise ValueError("Optim arg not supported")
      
      for epoch in range(epochs):
        train_batches = utils.batch_iter_attr(x_train, y_train, attr_target,
                                              target_words_mask, 
                                              batch_size, 1, shuffle=True, 
                                              seed=seed)
        for x_batch, y_batch, attr, mask in train_batches:
          train_feed_dict = {
            model.baseline: baseline,
            model.label_index: label_index,
            model.attr_target: attr,
            model.x: x_batch,
            model.y: y_batch,
            model.mask: mask,
            model.keep_prob: keep_prob
          }
        
          _, _ = sess.run([optim_op, loss_op], feed_dict=train_feed_dict)
        # Test accuracy with validation data for each epoch.
        sum_accuracy, cnt = 0, 0
        dev_batches = utils.batch_iter(x_dev, y_dev, batch_size, 1)
        for x_dev_batch, y_dev_batch in dev_batches:
          dev_feed_dict = {
            model.x: x_dev_batch,
            model.y: y_dev_batch,
          }
          accuracy = sess.run(model.accuracy, feed_dict=dev_feed_dict)
          sum_accuracy += accuracy
          cnt += 1
        dev_accuracy = sum_accuracy / cnt
        print(datetime.datetime.now())
        print("\nEpoch {0}: Validation Accuracy = {1}\n".format(epoch + 1, 
                                                                dev_accuracy))
        if dev_accuracy > best_accuracy:
          best_accuracy = dev_accuracy
          model_path = os.path.join(output_dir, "model_best.ckpt")
          save_path = saver.save(sess, model_path)
      
def main(argv=None):
  args, _ = parse_arguments()
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  args_path = os.path.join(args.output_dir, 'args.json')
  with open(args_path, 'w') as f:
    json.dump(vars(args), f)
  
  target_words = [] 
  if args.target_words_file:
    with open(args.target_words_file) as f:
      content = f.readlines()
      target_words = [x.strip() for x in content] 
  
  vocab_path = os.path.join(args.output_dir, 'vocab.txt')

  if args.target_words_to_token:
    idx2word, word2idx = utils.build_vocab(args.train_data, args.min_freq, vocab_path, target_words)
  else:
    idx2word, word2idx = utils.build_vocab(args.train_data, args.min_freq, vocab_path)

  x_train, y_train, attr_target = utils.preprocess(args.train_data, args.seq_len, word2idx, target_words=target_words, return_attr_target=True, target_words_to_token=args.target_words_to_token)
  
  target_words_mask = np.sum(attr_target-1, axis=1)
  target_words_mask[np.where(target_words_mask != 0)] = 1.0

  x_dev, y_dev = utils.preprocess(args.dev_data, args.seq_len, word2idx, target_words=target_words)
  train(x_train, y_train, attr_target, x_dev, y_dev, idx2word, word2idx, target_words_mask, args)

if __name__ == '__main__':
    tf.app.run()
