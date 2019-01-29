import random
import tensorflow as tf
import tf_context as ctx
import numpy as np

from tf_client import base_io

from agent import Agent
from hyper_config import HyperConfig

	
with ctx.graph():
  print("define ops network")
	
  task_name = ctx.get_task_name()
  task_index = ctx.get_task_index()
  worker_num = ctx.get_config("worker", "instance_num")
  ps_num = ctx.get_config("ps", "instance_num")

  print('task_name = [%s], task_index = [%s], woker_num = [%s], ps_num = [%s]' % (task_name, str(task_index), str(worker_num), str(ps_num) ) )

  agent = Agent(task_name, task_index, worker_num, ps_num)
	  
  with ctx.session() as sess:
	print("run network")

	agent.set_sess(sess)
	agent.run()



