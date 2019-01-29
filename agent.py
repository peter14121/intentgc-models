import tensorflow as tf
import numpy as np
from node2vec_feature_baseline_model_by_graph_plat_with_api import Node2VecFeatureModelByGraphPlatWithAPI
from node2vec_feature_onelayer_intentgc_model import Node2VecFeatureOneLayerIntentGCModel 
from node2vec_feature_onelayer_intentgc_model_mulkernel import Node2VecFeatureOneLayerIntentGCMulKernelModel
from node2vec_feature_onelayer_intentgc_model_sagenet_raw import Node2VecFeatureOneLayerIntentGCSageNetRawModel
from node2vec_feature_twolayer_intentgc_model import Node2VecFeatureTwoLayerIntentGCModel 
from node2vec_feature_twolayer_intentgc_model_sagenet import Node2VecFeatureTwoLayerIntentGCSageNetModel
from node2vec_feature_bine_model import Node2VecFeatureBiNEModel 
from node2vec_feature_deepwalk_model import Node2VecFeatureDeepwalkModel 
from node2vec_feature_i2i_gcn_model import Node2VecFeatureI2IGCNModel
from node2vec_feature_metapath_model import Node2VecFeatureMetapathModel 
from node2vec_feature_metapath2_model import Node2VecFeatureMetapath2Model 
from node2vec_feature_onelayer_intentgc_plus_model import Node2VecFeatureOneLayerIntentGCPlusModel
from node2vec_feature_twolayer_intentgc_plus_model import Node2VecFeatureTwoLayerIntentGCPlusModel
from hyper_config import HyperConfig 
import sys
import random

import tf_context as ctx
import graph_embedding.operations as geops


class Agent(object):
  def __init__(self, task_name, task_id, worker_num, ps_num):
    self.config = HyperConfig()
    
    #self.config.seed = seed
    #print('seed = %d' % self.config.seed)
    
    #self.sess = tf.Session()
    
    np.random.seed(self.config.seed)
    random.seed(self.config.seed)
    tf.set_random_seed(self.config.tf_seed)
    
    # init graph
    partition_num = int(ctx.get_config('extend_role','graph_server','partition_num'))
    zk_addr = 'please enter zk addr here'
    zk_path = '/graph_embedding/gs_%s' % ctx.get_app_id()
    self.init_graph(zk_addr, zk_path, partition_num)
    
    # init model
    model_type = ctx.get_config('model_type')
    if model_type == 'baseline':
      self.model = Node2VecFeatureModelByGraphPlatWithAPI(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == "one_intentgc":
      self.model = Node2VecFeatureOneLayerIntentGCModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == "one_intentgc_mulkernel":
      self.model = Node2VecFeatureOneLayerIntentGCMulKernelModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == "one_intentgc_sage_net_raw":
      self.model = Node2VecFeatureOneLayerIntentGCSageNetRawModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == "two_intentgc":
      self.model = Node2VecFeatureTwoLayerIntentGCModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == "two_intentgc_sage_net_raw":
      self.model = Node2VecFeatureTwoLayerIntentGCSageNetModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == 'bine':
      self.model = Node2VecFeatureBiNEModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == 'deepwalk':
      self.model = Node2VecFeatureDeepwalkModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == 'i2i_gcn':
      self.model = Node2VecFeatureI2IGCNModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == 'metapath':
      self.model = Node2VecFeatureMetapathModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == 'metapath2':
      self.model = Node2VecFeatureMetapath2Model(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == "one_intentgc_plus":
      self.model = Node2VecFeatureOneLayerIntentGCPlusModel(self.config, task_name, task_id, worker_num, ps_num)
    elif model_type == "two_intentgc_plus":
      self.model = Node2VecFeatureTwoLayerIntentGCPlusModel(self.config, task_name, task_id, worker_num, ps_num)

    self.model.build_graph_by_mode(self.config.build_mode)
    
    #self.model.set_sess(self.sess)
    
    #self.saver = tf.train.Saver()
    
    
  def init_graph(self, zk_addr, zk_path, partition_num):
    geops.init_graph_client_distributed(partition_num, zk_addr, zk_path)


  def set_sess(self, sess):
    self.model.set_sess(sess)
    
    
    
  def run(self):
    if self.config.build_mode == "train":
      self.run_train()
    elif self.config.build_mode == "test":
      self.model.test()
    elif self.config.build_mode == "predict_item":
      print('before predict item')
      self.model.predict_item()
      print('predict item finish')
    elif self.config.build_mode == "predict_user":
      print('before predict user')
      self.model.predict_user()
      print('predict user finish')
    else:
      print('fatal error! unsupported build mode')
      
      
  def run_train(self):
    # load data
    # TODO
    
    # 1.0 train
    print('before train')
    #self.sess.run(tf.global_variables_initializer())
    
    #self.model.set_train_data(train_data)
    self.model.train()
    #save_path = self.saver.save(self.sess, self.config.model_save_path)
    
    print('train finish')
    
    #self.model.analyze()
    #print('analyze finish')
    
    # 2.0 predict on train data
    # self.h_predict_and_validate(train_data)
    
    # 3.0 predict on test data
    #self.h_predict_and_validate(test_data)
    
    
  def run_predict(self, real_test):
    self.saver.restore(self.sess, self.config.model_save_path)
    predicts = []
    for station_id in real_test:
      self.model.set_test_data(real_test[station_id])
      # predict : [decoder_max_time, 1, targets_size]
      predict = self.model.predict()
      predicts.append(predict)
    return np.concatenate(predicts, 1)


if __name__ == '__main__':
  client = Agent()
  client.run_train()


