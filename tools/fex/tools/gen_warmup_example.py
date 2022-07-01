# -*- coding: utf-8 -*-
'''
Created on Jan-20-21 13:55
gen_warmup_example.py
Description: 一个生成warmup data的示例，在使用Laplace服务的时候需要有一个warmup_data的配置.
'''

from absl import flags
from absl import app
try:
    import numpy as np
    import tensorflow as tf
    from laplace import WarmupGenerator
except Exception as e:
    print(str(e))

FLAGS = flags.FLAGS

def make_tensor_proto(value, dtype=None):
    proto = tf.make_tensor_proto(value, dtype=dtype)
    binary = proto.SerializeToString()
    return binary

def def_flags():
    flags.DEFINE_string("warmup_path", './tf_serving_warmup_requests', "laplace serving need warmup data")


def gen_warmup_data(_):
    batch_size = 1
    inputs = {
        "queries": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "titles": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "user_nicknames": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "challenges": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "ocr_asr_summary": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "music_title": make_tensor_proto(["hello world"] * batch_size, dtype=tf.string),
        "sparse": make_tensor_proto([[1]*11] * batch_size, dtype=tf.int64),
        "dense": make_tensor_proto([[0.1]*5] * batch_size, dtype=tf.float32)
    }

    custom_input = {
        "vis_embed": [make_tensor_proto(np.random.rand(8, 128), dtype=tf.float32)] * batch_size
    }

    WarmupGenerator.make_inference(path=FLAGS.warmup_path, input_data=inputs, custom_input_data=custom_input)


if __name__ == "__main__":
    def_flags()
    app.run(gen_warmup_data)

