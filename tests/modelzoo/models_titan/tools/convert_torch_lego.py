import os.path
import typing as T
from pathlib import Path

import lego
import torch
from torch import nn
from anyon.utils.argument import AnyonParamBase, list_default_value


class LegoConverter(AnyonParamBase):
    checkpoint: Path  # -c; checkpoint path
    output_ft_engine: str  # -o; output ft engine filename, will be placed into {workspace}  # need validation
    is_torchscript: bool = False  # whether checkpoint is torchscript
    input_shapes: T.List[T.List[int]] = list_default_value()  # validation
    # input shapes seperated by space， eg: [224,224] [1,768]
    input_dtypes: T.List[str] = list_default_value()  # validation input dtypes
    # seperated by space, need to be a valid torch.dtype type like float32, int32, etc.
    input_values: T.List[str] = list_default_value()  # validation input values
    is_int8: bool = False  # in int8 precision

    # seperated by space, possible values: number, float or 'random'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ensure_path()
        self.output_ft_engine = os.path.join(self.workspace,
                                             self.output_ft_engine)

    def _validate(self):
        if not self.checkpoint.exists():
            raise ValueError(f"checkpoint file {self.checkpoint} not exist")

        assert len(self.input_shapes) == len(self.input_dtypes)
        assert len(self.input_shapes) == len(self.input_values)
        for idx, dtype in enumerate(self.input_dtypes):
            if not isinstance(dtype, str):
                continue

            self.input_dtypes[idx] = getattr(torch, dtype)
            try:
                self.input_values[idx] = float(self.input_values[idx])
            except ValueError:
                if isinstance(self.input_values[idx], str):
                    if self.input_values[idx] == 'random':
                        continue
                    if self.input_values[idx] == 'None':
                        self.input_values[idx] = 0
                        continue

                raise ValueError("input value must be a number or `random`")

    def _ensure_path(self):
        self.workspace.mkdir(parents=True,
                             exist_ok=True)  # ensure workspace exists
        # create_path_if_not_exists(self.workspace.)
        self.ts_path = self.workspace.joinpath('fp16_ts')
        self.ts_path.mkdir(parents=True, exist_ok=True)
        self.tmp_ts = self.ts_path.joinpath(self.checkpoint.name)

    def get_sample_inputs(self) -> T.List[torch.Tensor]:
        arr = []
        for shape, value, dtype in zip(self.input_shapes, self.input_values,
                                       self.input_dtypes):
            if value == 'random':
                tensor = torch.rand(*shape, dtype=dtype)
            else:
                tensor = torch.empty(*shape, dtype=dtype)
                tensor[:] = value

            arr.append(tensor.cuda())
        return arr

    def trace_ts_model(self) -> nn.Module:
        p = str(self.checkpoint)
        arr = self.get_sample_inputs()

        wrapper = torch.load(p, map_location='cuda:0').eval()
        m_half = wrapper.half()
        m_traced = torch.jit.trace(m_half, tuple(arr))
        torch.jit.save(m_traced, self.tmp_ts)
        return m_traced

    def prepare_ts_model(self):
        if not self.is_torchscript:
            original_model = self.trace_ts_model()
            p = self.tmp_ts
        else:
            p = str(self.checkpoint)
            original_model = torch.jit.load(p)

        return original_model, p

    def _optimize(self, pth):
        arr = self.get_sample_inputs()
        for i in arr:
            print(i.shape)
        if self.is_int8:
            lego.set_lego_thresold(5.0)  # larger diff range tolerance in int8

        return lego.optimize(pth, arr, len(arr) > 0, is_quant=self.is_int8)

    def convert(self):
        if not self.is_torchscript:
            original_model, p = self.prepare_ts_model()
        else:
            p = self.checkpoint
            original_model = torch.jit.load(self.checkpoint)

        lego.torch_load_lego_library()
        lego_model = self._optimize(p)
        lego_model.save(self.output_ft_engine)
        arr = self.get_sample_inputs()
        lego.perf_model(original_model, lego_model, arr[0], True)


if __name__ == '__main__':
    LegoConverter().convert()
