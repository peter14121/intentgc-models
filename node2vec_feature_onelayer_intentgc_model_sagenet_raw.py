import tensorflow as tf
import numpy as np
import random
import time

from sklearn.metrics import roc_auc_score

from tf_client import base_io
import tf_context as ctx
import graph_embedding.operations as geops

from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io.file_io import FileIO
import traceback
import time


class Node2VecFeatureOneLayerIntentGCSageNetRawModel(object):
  
  def __init__(self, config, task_name, task_id, worker_num, ps_num):
    self.config = config
    self.task_name = task_name
    self.task_id = task_id
    self.worker_num = worker_num
    self.ps_num = ps_num
    print('task id = %d, worker num = %d, ps_num = %d' % (self.task_id, self.worker_num, self.ps_num))
    
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self.weights_map = {}
    
  def build_graph_by_mode(self, mode):
    if mode == "train":
      self.build_train_graph()
    elif mode == "test":
      self.build_test_graph()
    elif mode == "predict_item":
      self.build_predict_item_graph()
    elif mode == "predict_user":
      self.build_predict_user_graph()
    else:
      print('fatal error! unsupported mode')
    

  def set_sess(self, sess):
    self.sess = sess
    
  # new implementation
  def build_train_graph(self):
    # 0. reader
    self.make_data()
    
    # 1. make item and user feature and embedding
    cur_item_whole_embedding, cur_user_whole_embedding, cur_neg_item_whole_embedding = self.make_features_and_embedding()

    # 2. build network
    self.build_network(cur_item_whole_embedding, cur_user_whole_embedding, cur_neg_item_whole_embedding)
    
    # 3. cal loss
    status = self.cal_loss()
    if status == False:
      print('fatal error! status of cal loss is error')
      
    # 4. make train op
    self.make_train_op()
    
  def build_test_graph(self):
    # 0. reader
    self.make_data()
    
    # 1. make item and user feature and embedding
    cur_item_whole_embedding, cur_user_whole_embedding, cur_neg_item_whole_embedding = self.make_features_and_embedding()
    
    # 2. build network
    self.build_network(cur_item_whole_embedding, cur_user_whole_embedding, cur_neg_item_whole_embedding)
    
    # 3. cal loss
    status = self.cal_loss()
    if status == False:
      print('fatal error! status of cal loss is error')
      
  def build_predict_item_graph(self):
    # 0. reader
    self.make_data()
    
    self.create_convolve_weights()
      
    # 1. make item feature and embedding
    with ctx.local():
      cur_item_ids = tf.reshape(self.datas.mapped_item_id.embed_key.values, [-1])
      self.item_ids = cur_item_ids
      cur_item_title_package, cur_item_brand_package, cur_item_cate_and_prop_package, cur_item_basic_package, cur_item_usual_user_level_package = self.make_item_fea(cur_item_ids)
    
    cur_item_whole_embedding = self.make_item_embedding(cur_item_title_package, cur_item_brand_package, cur_item_cate_and_prop_package, cur_item_basic_package, cur_item_usual_user_level_package)
   
    cur_item_final_embedding = self.one_layer_item_gcn(cur_item_ids, cur_item_whole_embedding, False)
    
    # 2. build item network
    self.build_network_for_item(cur_item_final_embedding)
    
  def build_predict_user_graph(self):
    # 0. reader
    self.make_data()
    
    self.create_convolve_weights()
      
    # 1. make user feature and embedding
    with ctx.local():
      cur_user_ids = tf.reshape(self.datas.mapped_user_id.embed_key.values, [-1])
      self.user_ids = cur_user_ids
      cur_user_like_brand_package, cur_user_like_cate_package, cur_user_like_term_package, cur_user_pre_item_basic_package, cur_user_info_package, cur_user_pre_item_score_package = self.make_user_fea(cur_user_ids)
    
    cur_user_whole_embedding = self.make_user_embedding(cur_user_like_brand_package, cur_user_like_cate_package, cur_user_like_term_package, cur_user_pre_item_basic_package, cur_user_info_package, cur_user_pre_item_score_package, share_embedding = False)
    
    cur_user_final_embedding = self.one_layer_user_gcn(cur_user_ids, cur_user_whole_embedding)
    
    # 2. build user network
    self.build_network_for_user(cur_user_final_embedding)
    
  def make_train_op(self):
    optimizer = None
    if self.config.opt_type == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
    elif self.config.opt_type == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
    elif self.config.opt_type == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=self.config.momentum)
    elif self.config.opt_type == 'ada_grad':
      optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate)
    else:
      optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate)
      
    if optimizer is None:
      print('fatal error! optimizer is None')
      
    sgd_op = None
    if self.config.grad_clip_threshold > 0:
      print('do clip gradient')
      grads_and_vars = optimizer.compute_gradients(self.loss)
      gradients, variables = zip(*grads_and_vars)
      clipped_gradients, glob_norm = tf.clip_by_global_norm(gradients, self.config.grad_clip_threshold)
      sgd_op, glob_norm = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step = self.global_step), glob_norm
    else:
      sgd_op = optimizer.minimize(self.loss, global_step = self.global_step)
      
    if sgd_op is None:
      print('fatal error! sgd op is None')
      
    self.train_op = sgd_op
    
  def cal_loss(self):
    # check batch size
    if (self.item_l3.get_shape()[0] != self.user_l3.get_shape()[0]):
      print('fatal error! batch size is not match')
      return False
    
    if (self.item_l3.get_shape()[0] != self.neg_item_l3.get_shape()[0]):
      print('fatal error! batch size is not match: neg')
      return False
    
    # check layer size
    if (self.item_l3.get_shape()[1] != self.user_l3.get_shape()[1]):
      print('fatal error! layer size is not match')
      return False
    
    if (self.item_l3.get_shape()[1] != self.neg_item_l3.get_shape()[1]):
      print('fatal error! layer size is not match: neg')
      return False
    
    mid_matrix = self.item_l3 * self.user_l3
    print('mid matrix shape:')
    print(mid_matrix.get_shape())
    
    neg_mid_matrix = self.neg_item_l3 * self.user_l3
    print('neg mid matrix shape:')
    print(neg_mid_matrix.get_shape())
    
    z_out = tf.reduce_sum(mid_matrix, axis=1, name="z_out")
    print('z_out shape:')
    print(z_out.get_shape())
    neg_z_out = tf.reduce_sum(neg_mid_matrix, axis=1, name="neg_z_out")
    print('neg z_out shape:')
    print(neg_z_out.get_shape())
    
    # cal predicts
    #self.predicts = 1. / (1. + tf.exp(-z_out))
    item_norm = tf.sqrt(tf.reduce_sum(tf.square(self.item_l3), axis=1))
    print('item norm shape:')
    print(item_norm.get_shape())
    user_norm = tf.sqrt(tf.reduce_sum(tf.square(self.user_l3), axis=1))
    print('user norm shape:')
    print(user_norm.get_shape())
    neg_item_norm = tf.sqrt(tf.reduce_sum(tf.square(self.neg_item_l3), axis=1))
    print('neg item norm shape:')
    print(neg_item_norm.get_shape())
    
    self.predicts = tf.divide(z_out, tf.multiply(item_norm, user_norm) + 0.000001)
    print('predicts shape:')
    print(self.predicts.get_shape())
    self.neg_predicts = tf.divide(neg_z_out, tf.multiply(neg_item_norm, user_norm) + 0.000001)
    print('neg predicts shape:')
    print(self.neg_predicts.get_shape())
    
    #print('label shape:')
    #print(self.labels.get_shape())
    
    # used for auc
    labels_real_tensor = tf.concat([(self.predicts * 0.) + 1., self.neg_predicts * 0.], 0)
    print('labels_real_tensor shape:')
    print(labels_real_tensor.get_shape())
    scores_tensor = tf.concat([(self.predicts + 1.) / 2., (self.neg_predicts + 1.) / 2.], 0)
    print('scores_tensor shape:')
    print(scores_tensor.get_shape())
    self.auc, self.auc_op = tf.metrics.auc(labels_real_tensor, scores_tensor, num_thresholds = 2000)
    self.test_auc, self.test_auc_op = tf.metrics.auc(labels_real_tensor, scores_tensor, name = 'test_auc', num_thresholds = 2000)
    
    # cal logit loss:  [batch_size, 1]
    #pre_loss = (self.labels * tf.log(self.predicts) + (1 - self.labels) * tf.log(1 - self.predicts) )
    #pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=z_out)
    #process_labels = -1 + (self.labels * 2)
    #process_labels = 1 + (self.labels * 0)
    pre_loss = tf.maximum(0.0, self.neg_predicts - self.predicts + self.config.threshold)
    print('pre_loss shape:')
    print(pre_loss.get_shape())
    
    #self.loss = tf.reduce_sum(tf.square(self.predicts - 1.), 0) + tf.reduce_sum(tf.square(self.neg_predicts - (-1.)), 0)
    #self.loss = tf.reduce_sum(tf.square(self.predicts - process_labels), 0)
    #self.loss = - tf.reduce_mean(pre_loss, 0)
    self.loss = tf.reduce_sum(pre_loss, 0)
    
    return True
 

  def build_network(self, cur_item_whole_embedding, cur_user_whole_embedding, cur_neg_item_whole_embedding):
    self.build_network_for_item(cur_item_whole_embedding)
    
    self.build_network_for_user(cur_user_whole_embedding)
    
    self.build_network_for_neg_item(cur_neg_item_whole_embedding)
    

  def build_network_for_item(self, cur_item_whole_embedding):
    #activation_fn = tf.nn.relu
    # build network for item
    #with tf.variable_scope("item_layer1"):
      #self.item_l1 = self.full_connect(cur_item_whole_embedding, [cur_item_whole_embedding.get_shape()[1], self.config.item_l1_size], [self.config.item_l1_size], activation_fn, "item_layer1")
      
    #with tf.variable_scope("item_layer2"):
      #self.item_l2 = self.full_connect(self.item_l1, [self.item_l1.get_shape()[1], self.config.item_l2_size], [self.config.item_l2_size], activation_fn, "item_layer2")
      
    #with tf.variable_scope("item_layer3"):
      #self.item_l3 = self.full_connect(self.item_l2, [self.item_l2.get_shape()[1], self.config.item_l3_size], [self.config.item_l3_size], None, "item_layer3")
    self.item_l3 = cur_item_whole_embedding


  def build_network_for_neg_item(self, cur_neg_item_whole_embedding):
    #activation_fn = tf.nn.relu
    # build network for item
    #self.neg_item_l1 = self.get_shared_full_connect(cur_neg_item_whole_embedding, activation_fn, "item_layer1")
    
    #self.neg_item_l2 = self.get_shared_full_connect(self.neg_item_l1, activation_fn, "item_layer2")
    
    #self.neg_item_l3 = self.get_shared_full_connect(self.neg_item_l2, None, "item_layer3")
    self.neg_item_l3 = cur_neg_item_whole_embedding

  
  def build_network_for_user(self, cur_user_whole_embedding):
    #activation_fn = tf.nn.relu
    # build network for user
    #with tf.variable_scope("user_layer1"):
      #self.user_l1 = self.full_connect(cur_user_whole_embedding, [cur_user_whole_embedding.get_shape()[1], self.config.user_l1_size], [self.config.user_l1_size], activation_fn, "user_layer1")
      
    #with tf.variable_scope("user_layer2"):
      #self.user_l2 = self.full_connect(self.user_l1, [self.user_l1.get_shape()[1], self.config.user_l2_size], [self.config.user_l2_size], activation_fn, "user_layer2")
      
    #with tf.variable_scope("user_layer3"):
      #self.user_l3 = self.full_connect(self.user_l2, [self.user_l2.get_shape()[1], self.config.user_l3_size], [self.config.user_l3_size], None, "user_layer3")
    self.user_l3 = cur_user_whole_embedding


  def full_connect(self, train_inputs, weights_shape, biases_shape, activation_fn, scope_name):
    # weights
    if self.ps_num > 1:
      weights = tf.get_variable("weights", weights_shape, initializer=self.get_initializer(stddev=self.config.network_stddev), regularizer=tf.nn.l2_loss, partitioner=tf.min_max_variable_partitioner(max_partitions=self.ps_num))
    else:
      weights = tf.get_variable("weights", weights_shape, initializer=self.get_initializer(stddev=self.config.network_stddev), regularizer=tf.nn.l2_loss)
      
    self.weights_map[scope_name] = weights
    
    # biases
    biases = tf.get_variable("biases", biases_shape, initializer=tf.constant_initializer(value=self.config.biases_init_value))
    self.weights_map['%s_biases' % scope_name] = biases
    
    out = tf.nn.bias_add(tf.matmul(train_inputs, weights), biases)
    
    if activation_fn != None:
      out = activation_fn(out)
    else:
      print("warning! no activation fn !")
      
    return out
  

  def get_shared_full_connect(self, train_inputs, activation_fn, scope_name):
    # weights
    weights = self.weights_map[scope_name] 
    
    # biases
    biases = self.weights_map['%s_biases' % scope_name] 
    
    out = tf.nn.bias_add(tf.matmul(train_inputs, weights), biases)
    
    if activation_fn != None:
      out = activation_fn(out)
    else:
      print("warning! no activation fn !")
      
    return out


  def one_layer_item_gcn(self, cur_item_ids, cur_item_whole_embedding, is_share): 
    with ctx.local():
      # 1-D neighbors
      cur_item_1s_neighbor_ids = self.get_neighbor_ids(cur_item_ids)
      cur_item_1s_neighbor_title_package, cur_item_1s_neighbor_brand_package, cur_item_1s_neighbor_cate_and_prop_package, cur_item_1s_neighbor_basic_package, cur_item_1s_neighbor_usual_user_level_package = self.make_item_fea(cur_item_1s_neighbor_ids)
    
    cur_item_1s_neighbor_whole_embedding = self.make_shared_item_embedding(cur_item_1s_neighbor_title_package, cur_item_1s_neighbor_brand_package, cur_item_1s_neighbor_cate_and_prop_package, cur_item_1s_neighbor_basic_package, cur_item_1s_neighbor_usual_user_level_package)

    # reshape to [batch, neighbor_cnt, emb_size]
    batch_size = tf.shape(cur_item_ids)[0]
    cur_item_1s_neighbor_format_embedding = tf.reshape(cur_item_1s_neighbor_whole_embedding, [batch_size, self.config.neighbor_cnt, -1])

    # [batch, emb_size]
    cur_item_1s_neighbor_pool_embedding = tf.reduce_mean(cur_item_1s_neighbor_format_embedding, axis = 1)
    
    # raw net
    cur_item_1s_concate_embedding = tf.concat((cur_item_1s_neighbor_pool_embedding, cur_item_whole_embedding), 1)
    
    with tf.variable_scope("one_layer_convolve_item"):
      if is_share == False:
        cur_item_final_embedding = self.full_connect(cur_item_1s_concate_embedding, [self.config.whole_emb_size*2, self.config.item_l3_size], [self.config.item_l3_size], None, "one_layer_convolve_item")
      else:
        cur_item_final_embedding = self.get_shared_full_connect(cur_item_1s_concate_embedding, None, "one_layer_convolve_item")

    return cur_item_final_embedding
  

  def one_layer_user_gcn(self, cur_user_ids, cur_user_whole_embedding): 
    with ctx.local():
      # 1-D neighbors
      cur_user_1s_neighbor_ids = self.get_neighbor_ids(cur_user_ids)
      cur_user_1s_neighbor_like_brand_package, cur_user_1s_neighbor_like_cate_package, cur_user_1s_neighbor_like_term_package, cur_user_1s_neighbor_pre_item_basic_package, cur_user_1s_neighbor_info_package, cur_user_1s_neighbor_pre_item_score_package = self.make_user_fea(cur_user_1s_neighbor_ids)
    
    cur_user_1s_neighbor_whole_embedding = self.make_user_embedding(cur_user_1s_neighbor_like_brand_package, cur_user_1s_neighbor_like_cate_package, cur_user_1s_neighbor_like_term_package, cur_user_1s_neighbor_pre_item_basic_package, cur_user_1s_neighbor_info_package, cur_user_1s_neighbor_pre_item_score_package, share_embedding = True) 

    # reshape to [batch, neighbor_cnt, emb_size]
    batch_size = tf.shape(cur_user_ids)[0]
    cur_user_1s_neighbor_format_embedding = tf.reshape(cur_user_1s_neighbor_whole_embedding, [batch_size, self.config.neighbor_cnt, -1])

    # [batch, emb_size]
    cur_user_1s_neighbor_pool_embedding = tf.reduce_mean(cur_user_1s_neighbor_format_embedding, axis = 1)
    
    # raw net
    cur_user_1s_concate_embedding = tf.concat((cur_user_1s_neighbor_pool_embedding, cur_user_whole_embedding), 1)
    with tf.variable_scope("one_layer_convolve_user"):
      cur_user_final_embedding = self.full_connect(cur_user_1s_concate_embedding, [self.config.whole_user_emb_size*2, self.config.user_l3_size], [self.config.user_l3_size], None, "one_layer_convolve_user")

    return cur_user_final_embedding
    

  def create_convolve_weights(self):
    # convolve, TODO can be more, notice name and scope
    if self.ps_num > 1:
      self.convolve_weights_item_neighbor = tf.get_variable("convolve_weights_item_neighbor", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev), partitioner=tf.min_max_variable_partitioner(max_partitions=self.ps_num))
      self.convolve_weights_item_self = tf.get_variable("convolve_weights_item_self", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev), partitioner=tf.min_max_variable_partitioner(max_partitions=self.ps_num))
      self.convolve_weights_user_neighbor = tf.get_variable("convolve_weights_user_neighbor", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev), partitioner=tf.min_max_variable_partitioner(max_partitions=self.ps_num))
      self.convolve_weights_user_self = tf.get_variable("convolve_weights_user_self", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev), partitioner=tf.min_max_variable_partitioner(max_partitions=self.ps_num))
    else:
      self.convolve_weights_item_neighbor = tf.get_variable("convolve_weights_item_neighbor", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev))
      self.convolve_weights_item_self = tf.get_variable("convolve_weights_item_self", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev))
      self.convolve_weights_user_neighbor = tf.get_variable("convolve_weights_user_neighbor", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev))
      self.convolve_weights_user_self = tf.get_variable("convolve_weights_user_self", (1,), initializer=self.get_initializer(stddev=self.config.convolve_stddev))


  def make_features_and_embedding(self):
    # 0. create convolve weights
    self.create_convolve_weights()
    
    # 1. make embedding for item
    with ctx.local():
      cur_item_ids = tf.reshape(self.datas.mapped_item_id.embed_key.values, [-1])
      cur_item_title_package, cur_item_brand_package, cur_item_cate_and_prop_package, cur_item_basic_package, cur_item_usual_user_level_package = self.make_item_fea(cur_item_ids)
    
    cur_item_whole_embedding = self.make_item_embedding(cur_item_title_package, cur_item_brand_package, cur_item_cate_and_prop_package, cur_item_basic_package, cur_item_usual_user_level_package)
   
    cur_item_final_embedding = self.one_layer_item_gcn(cur_item_ids, cur_item_whole_embedding, False)

    # 2. make embedding for user
    with ctx.local():
      cur_user_ids = tf.reshape(self.datas.mapped_user_id.embed_key.values, [-1])
      cur_user_like_brand_package, cur_user_like_cate_package, cur_user_like_term_package, cur_user_pre_item_basic_package, cur_user_info_package, cur_user_pre_item_score_package = self.make_user_fea(cur_user_ids)
    
    cur_user_whole_embedding = self.make_user_embedding(cur_user_like_brand_package, cur_user_like_cate_package, cur_user_like_term_package, cur_user_pre_item_basic_package, cur_user_info_package, cur_user_pre_item_score_package, share_embedding = False)
    
    cur_user_final_embedding = self.one_layer_user_gcn(cur_user_ids, cur_user_whole_embedding)
    
    # 3. make embedding for neg item, notice that the neg item share the same networks with the item
    with ctx.local():
      cur_neg_item_ids = tf.reshape(self.datas.mapped_neg_item_id.embed_key.values, [-1])
      cur_neg_item_title_package, cur_neg_item_brand_package, cur_neg_item_cate_and_prop_package, cur_neg_item_basic_package, cur_neg_item_usual_user_level_package = self.make_item_fea(cur_neg_item_ids)
    
    cur_neg_item_whole_embedding = self.make_shared_item_embedding(cur_neg_item_title_package, cur_neg_item_brand_package, cur_neg_item_cate_and_prop_package, cur_neg_item_basic_package, cur_neg_item_usual_user_level_package)
    
    cur_neg_item_final_embedding = self.one_layer_item_gcn(cur_neg_item_ids, cur_neg_item_whole_embedding, True)
 
    # 4. return
    return [cur_item_final_embedding, cur_user_final_embedding, cur_neg_item_final_embedding]
    
  
  
  def get_neighbor_ids(self, cur_nodes_ids):
    # fill neighbor ids from graph
    src_feature_idxs = [19]
    cur_item_neighbor_ids_sparse = self.assemble_graph(cur_nodes_ids, src_feature_idxs)
    #batch_size = tf.shape(cur_nodes_ids)[0]
    cur_item_neighbor_ids = cur_item_neighbor_ids_sparse[0].values
    return cur_item_neighbor_ids


  # new imple, this will be a basic api that be called by various function
  def make_item_fea(self, cur_nodes_ids):
   
    # fill feature from graph
    src_feature_idxs = range(10, 19)
    cur_item_fill_fea_nodes = self.assemble_graph(cur_nodes_ids, src_feature_idxs)
 
    # build abstract group data
    ret_item_title_term_data = [cur_item_fill_fea_nodes[0]]
    ret_item_title_term_data_dim = [self.config.i01_embed_dim]
    ret_item_title_term_data_name = ['i01']
    ret_item_title_package = [ret_item_title_term_data, ret_item_title_term_data_dim, ret_item_title_term_data_name]
    print('i01 embeded_dim: %d' % self.config.i01_embed_dim)
    
    ret_item_brand_data = [cur_item_fill_fea_nodes[1]]
    ret_item_brand_data_dim = [self.config.i02_embed_dim]
    ret_item_brand_data_name = ['i02']
    ret_item_brand_package = [ret_item_brand_data, ret_item_brand_data_dim, ret_item_brand_data_name]
    print('i02 embeded_dim: %d' % self.config.i02_embed_dim)

    ret_item_cate_and_prop_data = [cur_item_fill_fea_nodes[2], cur_item_fill_fea_nodes[3]]
    ret_item_cate_and_prop_data_dim = [self.config.i03_embed_dim, self.config.i04_embed_dim]
    ret_item_cate_and_prop_data_name = ['i03', 'i04']
    ret_item_cate_and_prop_package = [ret_item_cate_and_prop_data, ret_item_cate_and_prop_data_dim, ret_item_cate_and_prop_data_name]
    print('i03 embeded_dim: %d' % self.config.i03_embed_dim)
    print('i04 embeded_dim: %d' % self.config.i04_embed_dim)
    
    ret_item_basic_data = [cur_item_fill_fea_nodes[4], cur_item_fill_fea_nodes[5]]
    ret_item_basic_data_dim = [self.config.i05_embed_dim, self.config.i06_embed_dim]
    ret_item_basic_data_name = ['i05', 'i06']
    ret_item_basic_package = [ret_item_basic_data, ret_item_basic_data_dim, ret_item_basic_data_name]
    print('i05 embeded_dim: %d' % self.config.i05_embed_dim)
    print('i06 embeded_dim: %d' % self.config.i06_embed_dim)

    # new add feature
    ret_item_usual_user_level_data = [cur_item_fill_fea_nodes[6]]
    ret_item_usual_user_level_data_dim = [self.config.i07_embed_dim]
    ret_item_usual_user_level_data_name = ['i07']
    ret_item_usual_user_level_package = [ret_item_usual_user_level_data, ret_item_usual_user_level_data_dim, ret_item_usual_user_level_data_name]
    print('i07 embeded_dim: %d' % self.config.i07_embed_dim)

    return [ret_item_title_package, ret_item_brand_package, ret_item_cate_and_prop_package, ret_item_basic_package, ret_item_usual_user_level_package]
  
  
  # new imple, this will be a basic api that be called by various function
  def make_user_fea(self, cur_nodes_ids):
    
    # fill feature from graph
    src_feature_idxs = range(1, 10)
    cur_user_fill_fea_nodes = self.assemble_graph(cur_nodes_ids, src_feature_idxs)
 
    # build abstract group data
    cur_user_like_brand_data = [cur_user_fill_fea_nodes[0]]
    cur_user_like_brand_data_dim = [self.config.u01_embed_dim]
    cur_user_like_brand_data_name = ['u01']
    ret_user_like_brand_package = [cur_user_like_brand_data, cur_user_like_brand_data_dim, cur_user_like_brand_data_name]
    print('u01 embeded_dim: %d' % self.config.u01_embed_dim)
    
    cur_user_like_cate_data = [cur_user_fill_fea_nodes[1], cur_user_fill_fea_nodes[2]]
    cur_user_like_cate_data_dim = [self.config.u02_embed_dim, self.config.u03_embed_dim]
    cur_user_like_cate_data_name = ['u02', 'u03']
    ret_user_like_cate_package = [cur_user_like_cate_data, cur_user_like_cate_data_dim, cur_user_like_cate_data_name]
    print('u02 embeded_dim: %d' % self.config.u02_embed_dim)
    print('u03 embeded_dim: %d' % self.config.u03_embed_dim)

    cur_user_like_term_data = [cur_user_fill_fea_nodes[3]]
    cur_user_like_term_data_dim = [self.config.u04_embed_dim]
    cur_user_like_term_data_name = ['u04']
    ret_user_like_term_package = [cur_user_like_term_data, cur_user_like_term_data_dim, cur_user_like_term_data_name]
    print('u04 embeded_dim: %d' % self.config.u04_embed_dim)
    
    # new add feature
    cur_user_pre_item_basic_data = [cur_user_fill_fea_nodes[4]]
    cur_user_pre_item_basic_data_dim = [self.config.u05_embed_dim]
    cur_user_pre_item_basic_data_name = ['u05']
    ret_user_pre_item_basic_package = [cur_user_pre_item_basic_data, cur_user_pre_item_basic_data_dim, cur_user_pre_item_basic_data_name]
    print('u05 embeded_dim: %d' % self.config.u05_embed_dim)
    
    cur_user_info_data = [cur_user_fill_fea_nodes[5]]
    cur_user_info_data_dim = [self.config.u06_embed_dim]
    cur_user_info_data_name = ['u06']
    ret_user_info_package = [cur_user_info_data, cur_user_info_data_dim, cur_user_info_data_name]
    print('u06 embeded_dim: %d' % self.config.u06_embed_dim)
    
    cur_user_pre_item_score_data = [cur_user_fill_fea_nodes[6]]
    cur_user_pre_item_score_data_dim = [self.config.u07_embed_dim]
    cur_user_pre_item_score_data_name = ['u07']
    ret_user_pre_item_score_package = [cur_user_pre_item_score_data, cur_user_pre_item_score_data_dim, cur_user_pre_item_score_data_name]
    print('u07 embeded_dim: %d' % self.config.u07_embed_dim)

    return [ret_user_like_brand_package, ret_user_like_cate_package, ret_user_like_term_package, ret_user_pre_item_basic_package, ret_user_info_package, ret_user_pre_item_score_package]

    
  # new imple, this will be a basic api that be called by various function
  def make_item_embedding(self, cur_item_title_package, cur_item_brand_package, cur_item_cate_and_prop_package, cur_item_basic_package, cur_item_usual_user_level_package):
    # note here is no scope name, if there is several out api call, they will get the same weights
    
    # 2. make item embedding
    cur_item_title_term_data = cur_item_title_package[0]
    cur_item_title_term_data_dim = cur_item_title_package[1]
    cur_item_title_term_data_name = cur_item_title_package[2]
    cur_item_title_term_embedding = self.sparse_embedding(cur_item_title_term_data, cur_item_title_term_data_dim, self.config.term_embedding_dim, names=cur_item_title_term_data_name)
    cur_item_title_term_embedding = tf.concat(cur_item_title_term_embedding, 1)
    
    cur_item_brand_data = cur_item_brand_package[0]
    cur_item_brand_data_dim = cur_item_brand_package[1]
    cur_item_brand_data_name = cur_item_brand_package[2]
    cur_item_brand_embedding = self.sparse_embedding(cur_item_brand_data, cur_item_brand_data_dim, self.config.brand_embedding_dim, names=cur_item_brand_data_name)
    cur_item_brand_embedding = tf.concat(cur_item_brand_embedding, 1)
    
    cur_item_cate_and_prop_data = cur_item_cate_and_prop_package[0]
    cur_item_cate_and_prop_data_dim = cur_item_cate_and_prop_package[1]
    cur_item_cate_and_prop_data_name = cur_item_cate_and_prop_package[2]
    cur_item_cate_and_prop_embedding = self.sparse_embedding(cur_item_cate_and_prop_data, cur_item_cate_and_prop_data_dim, self.config.cate_prop_embedding_dim, names=cur_item_cate_and_prop_data_name)
    cur_item_cate_and_prop_embedding = tf.concat(cur_item_cate_and_prop_embedding, 1)
    
    # new add feature
    cur_item_basic_data = cur_item_basic_package[0]
    cur_item_basic_data_dim = cur_item_basic_package[1]
    cur_item_basic_data_name = cur_item_basic_package[2]
    cur_item_basic_embedding = self.sparse_embedding(cur_item_basic_data, cur_item_basic_data_dim, self.config.basic_info_dim, names=cur_item_basic_data_name)
    cur_item_basic_embedding = tf.concat(cur_item_basic_embedding, 1)
   
    cur_item_usual_user_level_data = cur_item_usual_user_level_package[0]
    cur_item_usual_user_level_data_dim = cur_item_usual_user_level_package[1]
    cur_item_usual_user_level_data_name = cur_item_usual_user_level_package[2]
    cur_item_usual_user_level_embedding = self.sparse_embedding(cur_item_usual_user_level_data, cur_item_usual_user_level_data_dim, self.config.basic_info_dim, names=cur_item_usual_user_level_data_name)
    cur_item_usual_user_level_embedding = tf.concat(cur_item_usual_user_level_embedding, 1)
    
    # 3. concate item all embeddings and dense
    #self.item_whole_embedding = tf.concat([self.item_title_term_embedding, self.item_cate_and_prop_embedding, self.item_brand_embedding, self.datas.i05.dense_val, self.datas.i06.dense_val, self.datas.i07.dense_val, self.datas.i08.dense_val], 1)
    cur_item_whole_embedding = tf.concat([cur_item_title_term_embedding, cur_item_brand_embedding, cur_item_cate_and_prop_embedding, cur_item_basic_embedding, cur_item_usual_user_level_embedding], 1)
    return cur_item_whole_embedding


  # new imple, this will be a basic api that be called by various function
  def make_shared_item_embedding(self, cur_item_title_package, cur_item_brand_package, cur_item_cate_and_prop_package, cur_item_basic_package, cur_item_usual_user_level_package):
    # note here is no scope name, if there is several out api call, they will get the same weights
    
    # 2. make  item embedding
    cur_item_title_term_data = cur_item_title_package[0]
    cur_item_title_term_data_name = cur_item_title_package[2]
    cur_item_title_term_embedding = self.get_shared_sparse_embedding(cur_item_title_term_data, names=cur_item_title_term_data_name)
    cur_item_title_term_embedding = tf.concat(cur_item_title_term_embedding, 1)
    
    cur_item_brand_data = cur_item_brand_package[0]
    cur_item_brand_data_name = cur_item_brand_package[2]
    cur_item_brand_embedding = self.get_shared_sparse_embedding(cur_item_brand_data, names=cur_item_brand_data_name)
    cur_item_brand_embedding = tf.concat(cur_item_brand_embedding, 1)
    
    cur_item_cate_and_prop_data = cur_item_cate_and_prop_package[0]
    cur_item_cate_and_prop_data_name = cur_item_cate_and_prop_package[2]
    cur_item_cate_and_prop_embedding = self.get_shared_sparse_embedding(cur_item_cate_and_prop_data, names=cur_item_cate_and_prop_data_name)
    cur_item_cate_and_prop_embedding = tf.concat(cur_item_cate_and_prop_embedding, 1)
    
    # new add feature
    cur_item_basic_data = cur_item_basic_package[0]
    cur_item_basic_data_name = cur_item_basic_package[2]
    cur_item_basic_embedding = self.get_shared_sparse_embedding(cur_item_basic_data, names=cur_item_basic_data_name)
    cur_item_basic_embedding = tf.concat(cur_item_basic_embedding, 1)
    
    cur_item_usual_user_level_data = cur_item_usual_user_level_package[0]
    cur_item_usual_user_level_data_name = cur_item_usual_user_level_package[2]
    cur_item_usual_user_level_embedding = self.get_shared_sparse_embedding(cur_item_usual_user_level_data, names=cur_item_usual_user_level_data_name)
    cur_item_usual_user_level_embedding = tf.concat(cur_item_usual_user_level_embedding, 1)
    
    # 3. concate item all embeddings and dense
    #self.item_whole_embedding = tf.concat([self.item_title_term_embedding, self.item_cate_and_prop_embedding, self.item_brand_embedding, self.datas.i05.dense_val, self.datas.i06.dense_val, self.datas.i07.dense_val, self.datas.i08.dense_val], 1)
    cur_item_whole_embedding = tf.concat([cur_item_title_term_embedding, cur_item_brand_embedding, cur_item_cate_and_prop_embedding, cur_item_basic_embedding, cur_item_usual_user_level_embedding], 1)
    return cur_item_whole_embedding
   

  # new imple, this will be a basic api that be called by various function
  def make_user_embedding(self, cur_user_like_brand_package, cur_user_like_cate_package, cur_user_like_term_package, cur_user_pre_item_basic_package, cur_user_info_package, cur_user_pre_item_score_package, share_embedding):
    # 4. make user embedding
    cur_user_like_brand_data = cur_user_like_brand_package[0]
    cur_user_like_brand_data_dim = cur_user_like_brand_package[1]
    cur_user_like_brand_data_name = cur_user_like_brand_package[2]
    if share_embedding == True:
      cur_user_like_brand_embedding = self.get_shared_sparse_embedding(cur_user_like_brand_data, names=cur_user_like_brand_data_name)
    else:
      cur_user_like_brand_embedding = self.sparse_embedding(cur_user_like_brand_data, cur_user_like_brand_data_dim, self.config.brand_embedding_dim, names=cur_user_like_brand_data_name)
    cur_user_like_brand_embedding = tf.concat(cur_user_like_brand_embedding, 1)
    
    cur_user_like_cate_data = cur_user_like_cate_package[0]
    cur_user_like_cate_data_dim = cur_user_like_cate_package[1]
    cur_user_like_cate_data_name = cur_user_like_cate_package[2]
    if share_embedding == True:
      cur_user_like_cate_embedding = self.get_shared_sparse_embedding(cur_user_like_cate_data, names=cur_user_like_cate_data_name)
    else:
      cur_user_like_cate_embedding = self.sparse_embedding(cur_user_like_cate_data, cur_user_like_cate_data_dim, self.config.cate_prop_embedding_dim, names=cur_user_like_cate_data_name)
    cur_user_like_cate_embedding = tf.concat(cur_user_like_cate_embedding, 1)
    
    cur_user_like_term_data = cur_user_like_term_package[0]
    cur_user_like_term_data_dim = cur_user_like_term_package[1]
    cur_user_like_term_data_name = cur_user_like_term_package[2]
    if share_embedding == True:
      cur_user_like_term_embedding = self.get_shared_sparse_embedding(cur_user_like_term_data, names=cur_user_like_term_data_name)
    else:
      cur_user_like_term_embedding = self.sparse_embedding(cur_user_like_term_data, cur_user_like_term_data_dim, self.config.term_embedding_dim, names=cur_user_like_term_data_name)
    cur_user_like_term_embedding = tf.concat(cur_user_like_term_embedding, 1)
    
    # add new feature
    cur_user_pre_item_basic_data = cur_user_pre_item_basic_package[0]
    cur_user_pre_item_basic_data_dim = cur_user_pre_item_basic_package[1]
    cur_user_pre_item_basic_data_name = cur_user_pre_item_basic_package[2]
    if share_embedding == True:
      cur_user_pre_item_basic_embedding = self.get_shared_sparse_embedding(cur_user_pre_item_basic_data, names=cur_user_pre_item_basic_data_name)
    else:
      cur_user_pre_item_basic_embedding = self.sparse_embedding(cur_user_pre_item_basic_data, cur_user_pre_item_basic_data_dim, self.config.basic_info_dim, names=cur_user_pre_item_basic_data_name)
    cur_user_pre_item_basic_embedding = tf.concat(cur_user_pre_item_basic_embedding, 1)
    
    cur_user_info_data = cur_user_info_package[0]
    cur_user_info_data_dim = cur_user_info_package[1]
    cur_user_info_data_name = cur_user_info_package[2]
    if share_embedding == True:
      cur_user_info_embedding = self.get_shared_sparse_embedding(cur_user_info_data, names=cur_user_info_data_name)
    else:
      cur_user_info_embedding = self.sparse_embedding(cur_user_info_data, cur_user_info_data_dim, self.config.basic_info_dim, names=cur_user_info_data_name)
    cur_user_info_embedding = tf.concat(cur_user_info_embedding, 1)
    
    cur_user_pre_item_score_data = cur_user_pre_item_score_package[0]
    cur_user_pre_item_score_data_dim = cur_user_pre_item_score_package[1]
    cur_user_pre_item_score_data_name = cur_user_pre_item_score_package[2]
    if share_embedding == True:
      cur_user_pre_item_score_embedding = self.get_shared_sparse_embedding(cur_user_pre_item_score_data, names=cur_user_pre_item_score_data_name)
    else:
      cur_user_pre_item_score_embedding = self.sparse_embedding(cur_user_pre_item_score_data, cur_user_pre_item_score_data_dim, self.config.basic_info_dim, names=cur_user_pre_item_score_data_name)
    cur_user_pre_item_score_embedding = tf.concat(cur_user_pre_item_score_embedding, 1)
    
    # 5. concate user all embeddings
    cur_user_whole_embedding = tf.concat([cur_user_like_brand_embedding, cur_user_like_cate_embedding, cur_user_like_term_embedding, cur_user_pre_item_basic_embedding, cur_user_info_embedding, cur_user_pre_item_score_embedding], 1)
    return cur_user_whole_embedding

    
  def make_data(self):
    with ctx.local():
      # build reader
      self.reader = base_io.DolphinReader(batch_size = self.config.mini_batch_size)
      self.tags, self.datas = self.reader.read(self.config.source_data_dir, self.task_id, self.worker_num)
      
      # get labels (click or not)
      #self.labels = tf.reshape(self.datas.label_click.dense_val, [-1])
  
  
  def assemble_graph(self, src_nodes, src_feature_idxs):
    
    #batch_size = tf.shape(src_nodes)[0]
    src_filled_nodes = geops.fill_sample(src_nodes, src_feature_idxs)
    
    #filled_nodes = tf.train.batch(src_filled_nodes,
                    #batch_size=self.config.mini_batch_size, # model batch_size
                    #num_threads=1,
                    #capacity=20000,
                    #enqueue_many=True,
                    #allow_smaller_final_batch=True)
    return src_filled_nodes
  
  
  def sparse_embedding(self, sp_tensors, input_dimensions, embedding_dimension, names):
    l = []
    for i in range(len(sp_tensors)):
      with tf.variable_scope(names[i]):
        embedding = self.full_connect_sparse(sp_tensors[i], [input_dimensions[i], embedding_dimension], None, names[i])
        l.append(embedding)
        
    return l
  

  def get_shared_sparse_embedding(self, sp_tensors, names):
    l = []
    for i in range(len(sp_tensors)):
      embedding = self.get_shared_full_connect_sparse(sp_tensors[i], None, names[i])
      l.append(embedding)
      
    return l


  def full_connect_sparse(self, train_inputs, weights_shape, sp_weights, scope_name):
    # weights
    #from tf_ps.ps_context import variable_info
    #with variable_info(batch_read=3000, var_type="hash"):
    if self.ps_num > 1:
      weights = tf.get_variable("weights", weights_shape, initializer=self.get_initializer(stddev=self.config.embedding_stddev), partitioner=tf.min_max_variable_partitioner(max_partitions=self.ps_num))
    else:
      weights = tf.get_variable("weights", weights_shape, initializer=self.get_initializer(stddev=self.config.embedding_stddev))
      
    self.weights_map[scope_name] = weights
    
    sample_embedding = tf.nn.embedding_lookup_sparse(weights, sp_ids=train_inputs, sp_weights=sp_weights, combiner="mean")
    
    # no bias here
    
    return sample_embedding


  def get_shared_full_connect_sparse(self, train_inputs, sp_weights, scope_name):
    # weights
    weights = self.weights_map[scope_name]
    
    sample_embedding = tf.nn.embedding_lookup_sparse(weights, sp_ids=train_inputs, sp_weights=sp_weights, combiner="mean")
    
    # no bias here
    
    return sample_embedding


  def get_initializer(self, dtype=tf.float32, stddev=1.0):
    if self.config.init_type == 1:
      return tf.initializers.truncated_normal(mean=0.0, stddev=stddev, seed=self.config.seed, dtype=dtype)
    else:
      # TODO, we can add more initializer here
      print('warning, init_type is wrong!')
      return tf.initializers.truncated_normal(mean=0.0, stddev=stddev, seed=self.config.seed, dtype=dtype)
  
  
  def train(self):
    cur_train_step = 0
    read_samples = 0
    start = time.time()
    while not self.sess.should_stop():
      #_, cur_loss, cur_auc_op, cur_auc = self.sess.run([self.train_op, self.loss, self.auc_op, self.auc])
      _, cur_loss = self.sess.run([self.train_op, self.loss])
	  #_, cur_loss, cur_auc_op, cur_auc, cur_predicts, cur_neg_predicts, sample_batches_i01, sample_batches_i02, sample_batches_i03, sample_batches_i04, sample_batches_u01, sample_batches_u02, sample_batches_u03, sample_batches_u04, sample_batches_u05 = self.sess.run([self.train_op, self.loss, self.auc_op, self.auc, self.predicts, self.neg_predicts, self.datas.i01.embed_key, self.datas.i02.embed_key, self.datas.i03.embed_key, self.datas.i04.embed_key, self.datas.u01.embed_key, self.datas.u02.embed_key, self.datas.u03.embed_key, self.datas.u04.embed_key, self.datas.u05.embed_key])
	
	  #if np.any(np.isnan(sample_batches_i01.values)):
		#print('nan exists in i01')
	  #if np.any(np.isnan(sample_batches_i02.values)):
		#print('nan exists in i02')
	  #if np.any(np.isnan(sample_batches_i03.values)):
		#print('nan exists in i03')
	  #if np.any(np.isnan(sample_batches_i04.values)):
		#print('nan exists in i04')
	  #if np.any(np.isnan(sample_batches_u01.values)):
		#print('nan exists in u01')
	  #if np.any(np.isnan(sample_batches_u02.values)):
		#print('nan exists in u02')
	  #if np.any(np.isnan(sample_batches_u03.values)):
		#print('nan exists in u03')
	  #if np.any(np.isnan(sample_batches_u04.values)):
		#print('nan exists in u04')
	  #if np.any(np.isnan(sample_batches_u05.values)):
		#print('nan exists in u05')

	  #read_samples = read_samples + len(sample_batches_i01)
      cur_train_step = cur_train_step + 1
	  #if cur_train_step > self.config.max_train_step:
		#break

	  #print('tf computs using '+str(tf_computs-start)+' s...')
	  
      if (cur_train_step >= self.config.test_step) and (cur_train_step % self.config.test_step == 0):
        #print('read samples num = [%d]' % read_samples)
        cur_auc_op, cur_auc = self.sess.run([self.auc_op, self.auc])
        print('cur_train_step=[%d], cur_loss=[%s], cur_auc=[%s]' % (cur_train_step, str(cur_loss), str(cur_auc_op)))
		#print('cur_train_step=[%d], cur_loss=[%s]' % (cur_train_step, str(cur_loss)))
		#print(sample_batches_i01)
		
		#label_real = np.concatenate([np.ones(len(cur_predicts),dtype=int), np.zeros(len(cur_neg_predicts),dtype=int)], axis = 0) 
		#scores = np.concatenate([(cur_predicts + 1.) / 2., (cur_neg_predicts + 1.) / 2.], axis = 0)
		#auc = roc_auc_score(label_real, scores)
		#print('sk auc = [%s]' % str(auc))

        tf_computs = time.time()
        print('tf computs using '+str(tf_computs-start)+' s...')
        start = time.time()
		
    print('train finish inner')
  
  
  def create_hdfs(self, hdfs_dir):
    if hdfs_dir is None:
      print('write error, the hdfs dir is none!')
      return  
    
    try:
      timestamp = time.strftime("%Y%m%d")
      new_hdfs_dir = hdfs_dir + '/' + timestamp
      file_io.recursive_create_dir(new_hdfs_dir)
      print('create dir succ')
      
      f = file_io.FileIO('%s/trained_vectors_%d' % (new_hdfs_dir, self.task_id), 'w')
      print('f create suc')
      return f
    
    except Exception, e:
      print('exception')
      print('str(Exception):\t%s' % str(Exception))
      print('str(e):\t\t%s' % str(e))
      print('repr(e):\t%s' % repr(e))
      print('e.message:\t%s' % e.message)
      print('traceback.print_exc():%s' % traceback.print_exc())
      print('traceback.format_exc():\n%s' % traceback.format_exc())
  
  
  # will be removed
  def write_to_hdfs(self, f, ids, trained_vectors, hdfs_dir):
    if len(ids) != len(trained_vectors):
      print('fatal error! len is not match for id and vectors')
      return

    try:
      cnt = 0
      for idx in range(0, len(trained_vectors)):
        content = '%s,%s\n' % (ids[idx], ';'.join(map(str, trained_vectors[idx])))
        f.write(content)
        cnt = cnt + 1
		#if (cnt % 1000000) == 0:
		#  print('%d write suc' % cnt)

	  #f.close()
	  #print('close suc')
    except Exception, e:
      print('exception')
      print('str(Exception):\t%s' % str(Exception))
      print('str(e):\t\t%s' % str(e))
      print('repr(e):\t%s' % repr(e))
      print('e.message:\t%s' % e.message)
      print('traceback.print_exc():%s' % traceback.print_exc())
      print('traceback.format_exc():\n%s' % traceback.format_exc())
  
  
  def test(self):
    cur_train_step = 0
    batch_all_predicts_list = []
    batch_real_label_list = []
    
    sum_auc = 0.
    global_auc = 0.
    while not self.sess.should_stop():
      ##cur_loss, cur_auc, cur_auc_op, cur_predicts, cur_neg_predicts  = self.sess.run([self.loss, self.test_auc, self.test_auc_op, self.predicts, self.neg_predicts])
      cur_loss, cur_auc, cur_auc_op = self.sess.run([self.loss, self.test_auc, self.test_auc_op])
      
      ##if (cur_train_step > 0) and (cur_train_step % 2000 == 0):
        ##batch_all_predicts_list = []
        ##batch_real_label_list = []
        ##print('cur_train_step = %d, avg_auc = [%s]' % (cur_train_step, sum_auc / (cur_train_step / self.config.test_step) ))
        
      ##batch_real_label_list.append(np.ones(len(cur_predicts), dtype=int))
      ##batch_real_label_list.append(np.zeros(len(cur_neg_predicts), dtype=int))
      ##batch_all_predicts_list.append(( (cur_predicts+1.) / 2. ))
      ##batch_all_predicts_list.append(( (cur_neg_predicts+1.) / 2. ))
      
      cur_train_step = cur_train_step + 1
      
      if (cur_train_step >= self.config.test_step) and (cur_train_step % self.config.test_step == 0):
        print('cur_train_step=[%d], cur_loss=[%s], cur_auc = [%s]' % (cur_train_step, str(cur_loss), str(cur_auc_op)))
        
        #print('predict:')
        #str_cur_predicts = ','.join(map(lambda x: str(x), cur_predicts))
        #print(str_cur_predicts)
        
        #print('neg predict:')
        #str_cur_neg_predicts = ','.join(map(lambda x: str(x), cur_neg_predicts))
        #print(str_cur_neg_predicts)
        
        # add more check
        #f_predicts = map(lambda x : 1 if x > 0.1 else 0, cur_predicts)
        #cur_right_num = np.sum(f_predicts)
        #recall = 1.0 * cur_right_num / len(cur_predicts)
        #neg_f_predicts = map(lambda x : 1 if x > 0.1 else 0, cur_neg_predicts)
        #cur_false_num = np.sum(neg_f_predicts)
        #precision = 1.0 * cur_right_num / (cur_right_num + cur_false_num + 0.000001)
        
        ##batch_labels = np.concatenate(batch_real_label_list, axis = 0)
        ##batch_scores = np.concatenate(batch_all_predicts_list, axis = 0)
        ##global_auc = roc_auc_score(batch_labels, batch_scores)
        ##sum_auc = sum_auc + global_auc
        
        #batch_all_predicts_list = []
        #batch_real_label_list = []
        
        ##label_real = np.concatenate([np.ones(len(cur_predicts),dtype=int), np.zeros(len(cur_neg_predicts),dtype=int)], axis = 0) 
        ##scores = np.concatenate([(cur_predicts + 1.) / 2., (cur_neg_predicts + 1.) / 2.], axis = 0)
        ##auc = roc_auc_score(label_real, scores)

        #f_predicts = map(lambda x : 1 if x > 0.1 else 0, cur_predicts)
        #cur_right_num = np.sum(f_predicts * cur_labels)
        #precision = 1.0 * cur_right_num / np.sum(f_predicts)
        #recall = 1.0 * cur_right_num / np.sum(cur_labels)
        #print('precision = [%s], recall = [%s], auc = [%s]' % (str(precision), str(recall), str(auc)))
        
        ##print('sk auc = [%s], global auc = [%s]' % (str(auc), str(global_auc)))


    print('test finish inner')
  

  def predict_user(self):
    f = self.create_hdfs(self.config.output_hdfs_dir)
    cnt = 0
    while not self.sess.should_stop():
      trained_vectors, cur_user_ids = self.sess.run([self.user_l3, self.user_ids])
      
      self.write_to_hdfs(f, cur_user_ids, trained_vectors, self.config.output_hdfs_dir)
      cnt = cnt + 1
      if (cnt % 10000) == 0:
        print('%d write suc' % cnt)
        
    f.close()
    print('close suc')
    
  
  def predict_item(self):
    f = self.create_hdfs(self.config.output_hdfs_dir)
    cnt = 0
    while not self.sess.should_stop():
      trained_vectors, cur_item_ids = self.sess.run([self.item_l3, self.item_ids])
      
      self.write_to_hdfs(f, cur_item_ids, trained_vectors, self.config.output_hdfs_dir)
      cnt = cnt + 1
      if (cnt % 1000) == 0:
        print('%d write suc' % cnt)
        
    f.close()
    print('close suc')



  def train_experiment(self):
    cur_train_step = 0
    read_samples = 0
    start = time.time()
    while not self.sess.should_stop():
      cur_item_ids, cur_items_fea = self.sess.run([self.item_ids, self.item_fill_fea_nodes])
      
      print('cur item ids:')
      print(cur_item_ids)
     
      for i in range(11, 27):
        print('cur items fea %d:' % i)
        print(cur_items_fea[i-11].values)
      
      break

