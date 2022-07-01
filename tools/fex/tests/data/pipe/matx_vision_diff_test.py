# -*- coding: utf-8 -*-
'''
Created on May-11-21 18:20
test_matx_diff.py
'''
import io
import json
import numpy as np
import torch
from base64 import b64decode
from fex.utils.hdfs_io import hopen
from fex.data.pipe.v_pipe import VisionPipe
from fex.data.pipe.matx_v_pipe import MatxVisionPipe
from fex.config import cfg, reset_cfg
import torchvision.transforms as transforms


def save_numpy_image(numpy_image, fn):
    def _save_image(image):
        with io.BytesIO() as output:
            image.save(output, format='jpeg', optimize=True, quality=85)
            return output.getvalue()

    trans_func = transforms.ToPILImage()
    with open(fn, 'wb') as f:
        f.write(_save_image(trans_func(numpy_image)))


def test_vision_pipeline_diff():
    hdfs_file = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/tusou/click_202057_filter_trn/part-00246.4mz"
    cfg.update_cfg('./ci/test_data/test_config/tuso_mlm.yaml')

    vision_pipe = VisionPipe(cfg)
    matx_vision_pipe = MatxVisionPipe(cfg)

    with hopen(hdfs_file, 'r') as fr:
        index = 0
        for line in fr:
            raw_data = json.loads(line.strip())

            example = vision_pipe(raw_data['transed_image_b64_binary'])
            example_matx = matx_vision_pipe(
                raw_data['transed_image_b64_binary'])

            vision_image = example["image"].numpy()
            matx_vision_image = example_matx["image"].numpy()

            save_numpy_image(torch.from_numpy(vision_image),
                             "./vision_image{}.jpeg".format(index))
            save_numpy_image(torch.from_numpy(matx_vision_image),
                             "./vision_matx_image{}.jpeg".format(index))


if __name__ == "__main__":
    # test_vision_pipeline_diff()
    pass
