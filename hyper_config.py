import argparse
import tf_context as ctx

class HyperConfig(object):

  def __init__(self):
    #node_count = 492425
    #node_count = 500743
    #node_count = 23197028
    #node_count = 48885
    #node_count = 54858
    #node_count = 13013332 
    #self.node_count = 26000000 
    
    #self.node_embedding_size = 10
    
    # 0. build mode
    self.build_mode = ctx.get_config("build_mode")
    
    # 1. opt config
    #learning_rate = 0.001
    self.learning_rate = 0.0001
    #self.grad_clip_threshold = 1.
    self.grad_clip_threshold = 0.
    #self.opt_type = 'adam'
    #self.opt_type = 'ada_grad'
    self.opt_type = 'momentum'
    self.momentum = 0.0001
    #self.inner_rate = 0.5
    
    # 2. net config
    self.item_l1_size = 800
    self.item_l2_size = 300
    self.item_l3_size = 100
    
    self.user_l1_size = 800
    self.user_l2_size = 300
    self.user_l3_size = 100
    
    
    # 3. initial config
    self.network_stddev = 0.8
    self.embedding_stddev = 0.4
    self.convolve_stddev = 0.9
    self.biases_init_value = 0.002
    self.init_type = 1
    
    self.term_embedding_dim = 15
    self.cate_prop_embedding_dim = 10
    self.brand_embedding_dim = 15
    #self.shop_embedding_dim = 15
    self.basic_info_dim = 20
    
    # this is changing
    # TODO
    self.whole_emb_size = 110
    self.whole_user_emb_size = 110

    self.seed = 133
    self.tf_seed = 123
    
    # 4. runtime config
    self.mini_batch_size = 200
    #self.mini_batch_size = 1000
    #self.train_epochs = 2
    #self.instance_read_samples = 10000000
    self.max_train_step = 21000
    self.test_step = 100

    # 5. algo config
    self.threshold = 0.3
    self.auxi_threshold = 0.3
    self.cvr_threshold = 0.2

    # 6. path config
    #self.split_v = ctx.get_config("split_v")
    
    self.source_data_dir = ctx.get_config("source_data_dir") 
    self.output_hdfs_dir = ctx.get_config("output_hdfs_dir")


    # 7. dict dim
    self.u01_embed_dim = 107500 
    self.u02_embed_dim = 17000
    self.u03_embed_dim = 50
    self.u04_embed_dim = 500010
    self.u05_embed_dim = 30
    self.u06_embed_dim = 30
    self.u07_embed_dim = 30
    self.u08_embed_dim = 30
    self.u09_embed_dim = 30
    
    self.i01_embed_dim = 1000010
    self.i02_embed_dim = 107500 
    self.i03_embed_dim = 17000
    self.i04_embed_dim = 50
    self.i05_embed_dim = 30
    self.i06_embed_dim = 30
    self.i07_embed_dim = 30
    self.i08_embed_dim = 30
    self.i09_embed_dim = 30

    # 8. gcn relate
    self.neighbor_cnt = 10


    # 9. laplass
    self.laplass_alpha = 0.6
    self.laplass_beta = 0.01


