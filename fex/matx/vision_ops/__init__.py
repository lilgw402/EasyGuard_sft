# -*- coding: utf-8 -*-
'''
Created on May-11-21 17:10
__init__.py
'''

from .recognize_image_format import image_rec
try:
    import matx
    import byted_vision
    if matx.__version__.startswith("1.5") and byted_vision.__version__.startswith("1.5"):
        from .vision_cpu_transform import NormalizeAndTranspose
        from .vision_cpu_transform import RateResizeAndRotate
        from .vision_cpu_transform import RandomResizedCrop
        from .vision_cpu_transform import Resize
        from .vision_cpu_transform import CenterCrop
        from .vision_cpu_transform import RandomHorizontalFlip
        from .vision_cpu_transform import Pad
        from .vision_cpu_transform import ImdecodeOp, CvtColorOp
except Exception:
    print("[NOTICE] Matx or BytedVision found in FEX/Matx Ops, please check !")


class TaskManager:
    def __init__(self, pool_size: int, use_lockfree_pool: bool) -> None:
        self.thread_pool: matx.NativeObject = matx.make_native_object(
            "ThreadPoolExecutor", pool_size, use_lockfree_pool)

    def get_thread_pool(self) -> object:
        return self.thread_pool
