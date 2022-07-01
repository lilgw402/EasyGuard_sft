
import base64

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.core.framework import tensor_pb2
except Exception as e:
    print(str(e))

def proto_bytes_to_numpy(value):
    proto = tensor_pb2.TensorProto()
    proto.ParseFromString(value)
    try:
        emb_num = tf.make_ndarray(proto)
    except ValueError:
        emb_num = tf.io.parse_tensor(value, out_type=tf.dtypes.as_dtype(proto.dtype))
    return emb_num


def make_tensor_proto(value, dtype=None):
    proto = tf.make_tensor_proto(value, dtype=dtype)
    binary = proto.SerializeToString()
    return binary