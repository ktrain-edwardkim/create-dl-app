import torch
print("cudnn version:{}".format(torch.backends.cudnn.version()))
print("cuda version: {}".format(torch.version.cuda))

import tensorflow as tf

sys_details = tf.sysconfig.get_build_info()
print(sys_details)