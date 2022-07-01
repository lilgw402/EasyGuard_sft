# -*- coding: utf-8 -*-

from typing import List
import json
import base64
import matx
import matx_pytorch

from fex.trace.model_convert import ModelConvertToFTAndFP16
from fex.config import cfg
from fex.utils.load import load_from_pretrain
from fex.utils.hdfs_io import hopen

from example.image_classify.model import ResNetClassifier
from example.image_classify.vision_pipline import VisionGPUPipe


class Tracer:
    def __init__(self, model):
        # 初始化训练MatxPipeline
        self.vision_pipe = VisionGPUPipe(thread_num=6,
                                         device_id=0,
                                         image_dsr_height=224,
                                         image_dsr_width=224,
                                         resize_shorter=256,
                                         mean=[
                                             0.485 * 255, 0.456 * 255, 0.406 * 255],
                                         std=[
                                             0.229 * 255, 0.224 * 255, 0.225 * 255],
                                         scale=[0.1, 1.0],
                                         ratio=[0.8, 1.25],
                                         is_trace=True)
        # 初始化matx_pytorch中的InferenceOp
        self.model = matx_pytorch.InferenceOp(model=model, device=0)

    def process(self, frames: List[bytes]):
        frames_nd = self.vision_pipe(frames)
        frames_embs = self.model(frames_nd)
        return frames_embs


def trace_model(frames_data):
    config_path = "example/image_classify/resnet_json_matx.yaml"
    ckpt_path = "example/image_classify/model_state_epoch_1050.th"
    cfg.update_cfg(config_path)

    resnet_classifier = ResNetClassifier(config=cfg)
    load_from_pretrain(resnet_classifier, pretrain_paths=ckpt_path)

    model_converter = ModelConvertToFTAndFP16(use_fp16=True, use_ft=True)
    fp16_model = model_converter.convert_model(resnet_classifier)

    tracer = Tracer(model=fp16_model)

    jit_module = matx.pipeline.Trace(tracer.process, frames_data)

    jit_module.SetAttr("output_names", ["embs"])
    jit_module.Save('./resnet_classifier_embs')

    ret = jit_module.Run({"frames": frames_data})
    ret = ret.asnumpy()
    print(ret.shape)


if __name__ == "__main__":
    data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/tower/imagenet_nte_org_n_nz/part-00247.snappy"
    frames_data = []
    with hopen(data_path, 'r') as fr:
        for line in fr:
            if len(frames_data) == 8:
                break
            data_item = json.loads(line)
            b64_binary = data_item["b64_resized_binary"]
            if isinstance(b64_binary, str):
                b64_binary = b64_binary.encode()
            frames_data.append(base64.decodebytes(b64_binary))

    trace_model(frames_data)
