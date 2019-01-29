import sys
import time
import ctypes
import os
import graph_embedding.graph_server as gs
import tf_context as ctx

data_path_config = str(ctx.get_config('extend_role','graph_server','data_dir'))
partition_num = int(ctx.get_config('extend_role','graph_server','partition_num'))
server_index = int(ctx.get_task_index())

partition = server_index % partition_num
app_id = ctx.get_app_id()
gs.server_init(partition = partition, data_path = data_path_config, zk_path='/graph_embedding/gs_%s' % app_id)


