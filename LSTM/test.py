import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import torch
print(torch.cuda.is_available())