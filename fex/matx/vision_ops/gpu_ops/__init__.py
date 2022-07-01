# -*- coding: utf-8 -*-

try:
    import matx
    import byted_vision
    if matx.__version__.startswith("1.5") and byted_vision.__version__.startswith("1.5"):
        from ..gpu_ops.cuda_cvtcolor import BatchCvtColor, CudaCvtColorOp
        from ..gpu_ops.cuda_imdecode import BatchCudaImdecodeOp, CudaImdecodeOp, BatchCudaImdecodeRandomCropOp
        from ..gpu_ops.cuda_normalize_with_stack import CudaKernelNormalizeWithStack, \
            CudaKernelNormalize, CudaKernelNormalizeFusionWithStack
        from ..gpu_ops.cuda_pad import BatchImagePad, ImagePad, NativeBatchGPUImagePad
        from ..gpu_ops.cuda_pad_and_stack import ImagesPadAndStack
        from ..gpu_ops.cuda_rate_resize import BatchGPURateResize, GPURateResize, NativeBatchGPURateResize
        from ..gpu_ops.cuda_rate_resize_and_rotate import BatchCudaRateResizeAndRotate, GPURateResizeAndRotate
        from ..gpu_ops.cuda_stream import CreateCudaStreams, BatchCudaStreamSync, CudaStreamSync
        from ..gpu_ops.cuda_random_resize_and_crop import BatchCudaRandomResizeCrop, NativeBatchCudaRandomResizeCrop
        from ..gpu_ops.cuda_center_crop import BatchGPUCenterCrop, NativeBatchGPUCenterCrop
except Exception:
    print("[NOTICE] Matx or BytedVision found in FEX/Matx Ops, please check !")
