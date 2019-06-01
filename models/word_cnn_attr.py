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

import tensorflow as tf

class WordCNNPIG():
  def __init__(self, sequence_len, num_classes, vocab_size,
               embedding_size, filter_sizes, num_filters, 
               learning_rate=0.001, steps=50, attr_loss_weight=100000000000, 
               reward=False, importance_weight=1):
    self.sequence_len = sequence_len
    self.num_classes = num_classes
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.filter_sizes = filter_sizes
    self.num_filters = num_filters
    self.learning_rate = learning_rate
    self.steps = steps
    self.attr_loss_weight = attr_loss_weight
    self.importance_weight = importance_weight
    self.reward = reward
    self._build_graph()

  def _build_graph(self):
    # CNN input
    self.x = tf.placeholder(tf.int32, [None, self.sequence_len], name="x")
    self.y = tf.placeholder(tf.int32, [None], name="y")
    self.keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep_prob")
    learning_rate = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
    self.global_step = tf.Variable(0, trainable=False, name='global_step')    

    # Pig input
    self.baseline = tf.placeholder(tf.int32, [None, self.sequence_len], name="baseline") 
    self.attr_target = tf.placeholder(tf.float32, [None, self.sequence_len], name="attr_target")
    self.label_index = tf.placeholder(tf.int32, [])

    # Importance Weight
    self.mask = tf.placeholder(tf.float32, [None], name="mask")

    # CNN Model Part
    with tf.name_scope("embedding"):
      init_embeddings = tf.random_uniform([self.vocab_size, 
                                           self.embedding_size])
      self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
      self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)

      x_emb_expand = tf.expand_dims(self.x_emb, -1)

      baseline_emb = tf.nn.embedding_lookup(self.embeddings, self.baseline)[0]

    # PIG Interpolation
    with tf.name_scope("pig_scale"):
      scaled_inp = []
      for i in range(1, self.steps+1):
        scaled_inp.append((float(i) / self.steps) * (self.x_emb-baseline_emb) + baseline_emb)
      inp_concat = tf.concat(scaled_inp, axis=0)
      inp_concat_stop = tf.stop_gradient(inp_concat)
      inp_concat_expand = tf.expand_dims(inp_concat_stop, -1)
        
    with tf.name_scope("conv_pool"):
      pooled_outputs = []
      pooled_outputs_scaled = []
      for i, filter_size in enumerate(self.filter_sizes):
        conv = tf.layers.conv2d(x_emb_expand,
                                filters=self.num_filters,
                                kernel_size=[filter_size, self.embedding_size],
                                strides=(1, 1),
                                padding="VALID",                                 
                                activation=tf.nn.relu,
                                name='conv_'+ str(i))
        pool = tf.layers.max_pooling2d(conv, 
                                       pool_size=[self.sequence_len - filter_size + 1, 1],
                                       strides=(1, 1),
                                       padding="VALID")
        pooled_outputs.append(pool)


        conv_scaled = tf.layers.conv2d(inp_concat_expand,
                                       filters=self.num_filters,
                                       kernel_size=[filter_size, self.embedding_size],
                                       strides=(1, 1),
                                       padding="VALID",
                                       activation=tf.nn.relu,
                                       reuse=True, name='conv_'+ str(i))

        pool_scaled = tf.layers.max_pooling2d(conv_scaled,
                                              pool_size=[self.sequence_len - filter_size + 1, 1],
                                              strides=(1, 1),
                                              padding="VALID")
        pooled_outputs_scaled.append(pool_scaled)


      h_pool = tf.concat(pooled_outputs, 3)
      h_pool_scaled = tf.concat(pooled_outputs_scaled, 3)
      self.h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])
      self.h_pool_flat_scaled = tf.reshape(h_pool_scaled, [-1, self.num_filters * len(self.filter_sizes)])

    with tf.name_scope("dropout"):
      mask = tf.nn.dropout(tf.ones_like(self.h_pool_flat), self.keep_prob)
      mask_scaled = tf.tile(mask, [self.steps, 1])
      h_drop = self.h_pool_flat * mask
      h_drop_scaled = self.h_pool_flat_scaled * mask_scaled

    with tf.name_scope("output"):
      self.logits = tf.layers.dense(h_drop, self.num_classes, activation=None, name='dense')
      self.logits_scaled = tf.layers.dense(h_drop_scaled, self.num_classes, activation=None, reuse=True, name='dense')
      self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

    with tf.name_scope("softmax"):
      self.softmax = tf.nn.softmax(self.logits)
      self.softmax_scaled = tf.nn.softmax(self.logits_scaled)
     
    # Pig part
    with tf.name_scope("pig"):
      self.grad = tf.gradients(self.softmax_scaled[:, self.label_index], inp_concat_stop)[0] 
      grad_reshape = tf.reshape(self.grad, [self.steps, -1, self.sequence_len, self.embedding_size])
      avg_grad = tf.reduce_mean(grad_reshape, axis=0)
      intergrated_grad = avg_grad * (self.x_emb - baseline_emb)
      self.tmp = intergrated_grad
      self.sum_intergrated_grad = tf.reduce_sum(intergrated_grad, axis=2) 
      # 0 for target token [a0, a1, 0, a3, 0, ...]
      target_attribution = tf.multiply(self.sum_intergrated_grad, self.attr_target)
      
      if self.reward:
        # 1 for target token [a1, a2, 1, a4, 1, ...]
        target_attribution = target_attribution + -1 * (self.attr_target - 1)

    with tf.name_scope("loss"):
      labels = tf.one_hot(self.y, depth=self.num_classes)
      cross_entropy = - tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(self.logits)), reduction_indices=1)
      self.indiv_loss = cross_entropy
      self.importance_loss = tf.reduce_mean(cross_entropy * (self.mask-1) * -1) + tf.reduce_mean(cross_entropy * (self.mask)) * self.importance_weight
      self.clf_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
      self.attr_loss = tf.losses.mean_squared_error(target_attribution, self.sum_intergrated_grad) * self.attr_loss_weight
      self.loss = self.clf_loss + self.attr_loss
      self.clf_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.clf_loss, global_step=self.global_step)
      self.attr_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.attr_loss, global_step=self.global_step)
      self.joint_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
      self.importance_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.importance_loss, global_step=self.global_step)

    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions, self.y)
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
