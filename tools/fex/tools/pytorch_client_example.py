# -*- coding: utf-8 -*-
'''
Created on Jan-20-21 18:46
pytorch_client_example.py
@author: liuzhen.nlp
Description: 访问线上laplace model的client实例, 方便参照.
'''

from absl import flags
from absl import app
try:
    import numpy as np
    import tensorflow as tf
    from laplace import Laplace
except Exception as e:
    print(str(e))

FLAGS = flags.FLAGS

def make_tensor_proto(value, dtype=None):
    proto = tf.make_tensor_proto(value, dtype=dtype)
    binary = proto.SerializeToString()
    return binary

def def_flags():
    flags.DEFINE_string("psm", "", "laplace serving psm")
    flags.DEFINE_string("ip_port", "", "laplace serving ip && port , example: 10.8.128.144:8992")


def make_request(_):
    if FLAGS.psm:
        laplace_client = Laplace(FLAGS.psm)
    if FLAGS.ip_port:
        laplace_client = Laplace(FLAGS.ip_port)
    
    batch_size = 1

    inputs = {
        "queries": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "titles": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "user_nicknames": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "challenges": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string)
    }

    custom_input = {
        "vis_embed": [make_tensor_proto(np.random.rand(8, 128), dtype=tf.float32)] * batch_size
    }

    result = laplace_client.inference(model_name="", input_data=inputs, custom_input_data=custom_input, caller="data.nlp.test")
    print(result)

if __name__ == "__main__":
    def_flags()
    app.run(make_request)

