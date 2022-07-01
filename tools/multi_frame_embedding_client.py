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
    from tensorflow.core.framework import tensor_pb2
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

def read_image():
    with open("./origin_image.jpeg", "rb") as img:
        img_bytes = img.read()
        return img_bytes

def make_request(_):
    if FLAGS.psm:
        laplace_client = Laplace(FLAGS.psm)
    if FLAGS.ip_port:
        laplace_client = Laplace(FLAGS.ip_port)
    

    batch_size = 1
    image_str = read_image()
    input_fields = {}
    input_fields["frame0"] = make_tensor_proto([image_str, ""], dtype=tf.string)
    input_fields["frame1"] = make_tensor_proto([image_str, ""], dtype=tf.string)
    input_fields["frame2"] = make_tensor_proto([image_str, image_str], dtype=tf.string)
    input_fields["frame3"] = make_tensor_proto([image_str, image_str], dtype=tf.string)
    input_fields["frame4"] = make_tensor_proto([image_str, image_str], dtype=tf.string)
    input_fields["frame5"] = make_tensor_proto([image_str, ""], dtype=tf.string)
    input_fields["frame6"] = make_tensor_proto(["", image_str], dtype=tf.string)
    input_fields["frame7"] = make_tensor_proto([image_str, image_str], dtype=tf.string)

    result = laplace_client.inference(model_name="search_cross_modal_multi_frames_resnet_mean8", input_data=input_fields, caller="data.nlp.test")
    values = result.custom_output["bbox_embedding"]
    print("result batch size: ", len(values))
    tensors = []
    for value in values:
        proto = tensor_pb2.TensorProto()
        proto.ParseFromString(value)
        tensor = tf.make_ndarray(proto)
        print('tensor shape: ', tensor.shape)
        tensors.append(tensor)



if __name__ == "__main__":
    def_flags()
    app.run(make_request)

