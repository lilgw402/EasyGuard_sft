"""
accelerators: 包含各种训练加速module
"""
from fex.engine.accelerators.accelerator import Accelerator
from fex.engine.accelerators.ddp_accelerator import DDPAccelerator
from fex.engine.accelerators.apex_ddp_accelerator import ApexDDPAccelerator
from fex.engine.accelerators.xla_ddp_accelerator import XlaDDPAccelerator
from fex.engine.accelerators.torch_ddp_accelerator import TorchAMPDDPAccelerator
from fex.engine.accelerators.ptx_ddp_accelerator import PtxAMPDDPAccelerator

ACCELERATOR_MAP = {'ApexDDP': ApexDDPAccelerator,
                   'DDP': DDPAccelerator,
                   'XlaDDP': XlaDDPAccelerator,
                   'TorchAMPDDP': TorchAMPDDPAccelerator,
                   'PtxAMPDDP': PtxAMPDDPAccelerator
                   }
