# -*- coding: utf-8 -*-

import torch

from fex import _logger as logger

try:
    import matx
    if matx.__version__.startswith("1.5"):
        import matx_pytorch
        to_torch_type = {"float32": torch.float32,
                         "float64": torch.float64,
                         "uint8": torch.uint8,
                         "int8": torch.int8,
                         "int16": torch.int16,
                         "int32": torch.int32,
                         "int64": torch.int64}

        device_map = {"cpu": torch.device('cpu'),
                      "gpu:0": torch.device('cuda', 0),
                      "gpu:1": torch.device('cuda', 1),
                      "gpu:2": torch.device('cuda', 2),
                      "gpu:3": torch.device('cuda', 3),
                      "gpu:4": torch.device('cuda', 4),
                      "gpu:5": torch.device('cuda', 5),
                      "gpu:6": torch.device('cuda', 6),
                      "gpu:7": torch.device('cuda', 7)}

    def convert_matx_ndarry_to_torch_tensor(nd: matx.NDArray) -> torch.Tensor:
        nd_device = nd.device()
        tensor_device = device_map[nd_device]
        tensor_dtype = to_torch_type[nd.dtype()]
        ret_tensor = torch.empty(list(nd.shape()),
                                 dtype=tensor_dtype,
                                 device=tensor_device)
        matx_pytorch.DLTensorToTorchTensorBuffer(ret_tensor, nd)
        return ret_tensor

except Exception:
    logger.warn("[NOTICE] Matx or Matx_pytorch found in FEX/Matx Ops, please check !")
