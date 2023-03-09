r"""
Build BERT (Albert), mainly based on PTX.
"""

from typing import List, Optional, Tuple, Dict, Iterable, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import embedding
import torch.utils.checkpoint
import numpy as np

import math
from dataclasses import dataclass, field
from .xperf_training import FTLinear, FTLayerNorm, FTSoftmax


# MHAOutput = namedtuple('MHAOutput', ['attn_outputs', 'attn_probs', 'attn_weights', 'q_states', 'k_states', 'v_states', 'attn_bias'])
# TransformerEncoderLayerOutput = namedtuple('TransformerEncoderLayerOutput',
#                                            ['hidden_states', 'attn_probs', 'attn_weights', 'q_states', 'k_states', 'v_states', 'attn_bias'])
# TransformerEncoderOutput = namedtuple('TransformerEncoderOutput',
#                                       ['last_hidden_state', 'all_hidden_states', 'all_attn_probs', 'all_attn_weights',
#                                        'all_q_states', 'all_k_states', 'all_v_states'])


ColumnParallelLinear = None
RowParallelLinear = None
MockModule = None
veGiantModel = None


# define metric
class Metric:
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor]):
        """
        Args:
            predictions : ``torch.Tensor``, required.
                A tensor of predictions.
            gold_labels : ``torch.Tensor``, required.
                A tensor corresponding to some gold label to evaluate against.
            mask: ``torch.Tensor``, optional (default = None).
                A mask can be passed, in order to deal with metrics which are
                computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor, move_to_cpu=False):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly, and they can be on
        the CPU.
        """
        return ((x.detach().cpu() if move_to_cpu else x.detach()) if isinstance(x, torch.Tensor) else x for x in tensors)

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)

class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise Exception(
                "Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)"
            )
        if top_k <= 0:
            raise Exception("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0
        self.local_value = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            predictions : `torch.Tensor`, required.
                A tensor of predictions of shape (batch_size, ..., num_classes).
            gold_labels : `torch.Tensor`, required.
                A tensor of integer class label of shape (batch_size, ...). It must be the same
                shape as the `predictions` tensor without the `num_classes` dimension.
            mask : `torch.Tensor`, optional (default = None).
                A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        predictions_within_classes = True
        if predictions.dim() - gold_labels.dim() == 1:
            num_classes = predictions.size(-1)
            if (gold_labels >= num_classes).any():
                raise Exception(
                    "A gold label passed to Categorical Accuracy contains an id >= {}, "
                    "the number of classes.".format(num_classes)
                )
            predictions = predictions.view((-1, num_classes)).float()
        else:
            assert self._top_k == 1, "`top_k` should be 1 if `predictions` has no `num_classes` dimension."
            predictions_within_classes = False
            predictions = predictions.view(-1).float()

        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1) if predictions_within_classes else predictions.unsqueeze(-1)
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # TODO
            assert predictions_within_classes, "`tie_break` requires `predictions` with `num_classes` dimension."

            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel(), device=gold_labels.device).long(), gold_labels
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        total_count_add = 0
        if mask is not None:
            correct *= mask.view(-1, 1).float()
            total_count_add = mask.sum()
        else:
            total_count_add = gold_labels.numel()
        correct_count_add = correct.sum()
        if total_count_add > 1e-12:
            self.local_value = float(correct_count_add) / float(total_count_add)
        else:
            self.local_value = 0.0
        self.total_count += total_count_add
        self.correct_count += correct_count_add

    def get_metric(self, reset: bool = False, use_local: bool = False):
        """
        Returns:
            The accumulated accuracy.
        """
        if not use_local:
            if self.total_count > 1e-12:
                accuracy = float(self.correct_count) / float(self.total_count)
            else:
                accuracy = 0.0
        else:
            accuracy = self.local_value
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
        self.local_value = 0.0


@dataclass
class TransformerConfig:
    n_layers: int
    dim: int
    n_heads: int
    dim_ff: int
    act: str = 'gelu'
    layernorm_type: str = 'v0'  # See `ptx.ops.layernorm`
    use_pre_layernorm: bool = False
    use_realformer: bool = False
    p_drop_hidden: float = 0.1
    p_drop_hidden2: Optional[float] = None
    p_drop_attn: float = 0.1
    return_layers: List[int] = field(default_factory=list)
    clamp_inf_nan: bool = False
    layer_norm_eps: float = 1e-5
    max_batch_size: int = 128
    max_seq_length: int = 512
    fp16: bool = True
    layernorm_fp16: bool = False
    remove_padding: bool = False
    fuse_qkv_projs: bool = False
    omit_other_attn_output: bool = False
    layer_grad_checkpoint: bool = False
    decoder_layer_grad_checkpoint: bool = False
    use_ft_softmax: bool = False
    disable_ft_softmax_dropout: bool = False
    use_ft_layernorm: bool = False
    use_apex_mha_mask_additive: bool = False
    use_ft_linear_in_attn: bool = False
    use_mp_linear_in_attn: bool = False
    use_ft_transpose_in_attn: bool = False
    use_ft_mm_in_attn: bool = False
    use_ft_linear_in_attn_out: bool = False
    use_mp_linear_in_attn_out: bool = False
    use_ft_linear_in_ffn: bool = False
    use_mp_linear_in_ffn: bool = False
    mha_acts_unite_d01: bool = True
    dropout_in_ffn: bool = False
    use_ft_ffn_linear_fusion: bool = False
    use_ffn_output_dropout: bool = False
    use_ft_attn_out_proj_dropout_fusion: bool = False
    use_ft_linear_transpose_fusion_in_attn: bool = False
    use_ft_remove_pad: bool = False
    use_ft_fused_attn: bool = False
    n_decoder_layers: int = 0
    return_decoder_layers: List[int] = field(default_factory=list)
    pad_seq_len_even: bool = False
    use_moe: bool = False
    use_moe_type: str = "pwff"
    use_moe_transformer_layer: str = ""
    moe_k: int = 1
    moe_experts: int = 8
    moe_output_dropout_prob: float = 0.0
    moe_min_capacity: int = 1
    moe_capacity_factor: float = 1.0
    moe_l_aux_factor: float = 1.0
    moe_z_loss_factor: float = 0.0
    moe_noisy_gate_policy: Optional[str] = None
    moe_eval_capacity_factor: float = 1.0
    moe_experts_decay: bool = False
    moe_dim: int = 1024
    moe_dropout: bool = False
    moe_flexible_validate: bool = True
    moe_drop_token: bool = False
    moe_random_token_select: bool = False
    moe_load_balanced: bool = False
    moe_enable_token_drop: bool = False
    moe_warmup_stage: Optional[str] = None
    moe_warmup_steps: str = ''
    moe_expert_shape: str = 'abc->abd'
    use_moe_attn: bool = False
    use_moe_transformer_layer_attn: str = ""
    moe_k_attn: int = 2
    moe_experts_attn: int = 32
    moe_dropout_attn: bool = False
    moe_l_aux_factor_attn: float = 0.01
    moe_dim_attn: int = 1024
    moe_load_balanced_attn: bool = False
    moe_attn_expert_shape: str = 'abc->abd'
    pos_emb_type: str = ''
    use_ft_preset: str = ''
    n_t5_rel_pos_buckets: int = 32
    beit_rel_pos_window_size: List[int] = field(default_factory=list)
    use_deep_norm: bool = False
    deep_norm_enc_alpha: float = 1.0
    deep_norm_enc_beta: float = 1.0

    def __post_init__(self):
        if self.p_drop_hidden2 is None:
            self.p_drop_hidden2 = self.p_drop_hidden
        self.head_dim = self.dim // self.n_heads
        assert self.dim % self.n_heads == 0, f"`dim` must be divisible by `n_heads` (got {self.dim}/{self.n_heads})."

        if self.use_deep_norm:
            self.deep_norm_enc_alpha = (2 * self.n_layers) ** 0.25
            self.deep_norm_enc_beta = (8 * self.n_layers) ** -0.25

Config = TransformerConfig

def gather_positions(sequence: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        sequence: tensor (batch_size, seq_len, dim)
        positions: tensor (batch_size, number_of_tokens_to_be_masked)

    Returns:
        tensor (batch_size, number_of_tokens_to_be_masked, dim)
    """

    batch_size, seq_len, dim = sequence.shape
    position_shift = (seq_len * torch.arange(batch_size, device=sequence.device)).unsqueeze(-1)
    flat_positions = torch.reshape(positions + position_shift, [-1]).long()
    flat_sequence = torch.reshape(sequence, [batch_size * seq_len, dim])
    gathered = flat_sequence.index_select(0, flat_positions)
    return torch.reshape(gathered, [batch_size, -1, dim])



# define Activation
def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

class Swish:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return swish(tensor)

class Mish:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return mish(tensor)

def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

gelu = F.gelu

class GeluOld:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return _gelu_python(tensor)

class Gelu:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return gelu(tensor)

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GeluNew:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return gelu_new(tensor)

class SquaredReLU:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.relu(tensor))

Activations = {
    "linear": lambda: lambda x: x,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "elu": torch.nn.ELU,
    "prelu": torch.nn.PReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "threshold": torch.nn.Threshold,
    "hardtanh": torch.nn.Hardtanh,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "log_sigmoid": torch.nn.LogSigmoid,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanhshrink": torch.nn.Tanhshrink,
    "swish": Swish,
    "mish": Mish,
    "gelu_old": GeluOld,
    "gelu": Gelu,
    "gelu_new": GeluNew,
    "squared_relu": SquaredReLU,
}

def Activation(name: str, **kwargs):
    return Activations[name](**kwargs)



# define LayerNormTypes
class LayerNorm(nn.Module):
    """
    An implementation of `Layer Normalization
    <https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5>`_ .

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    Args:
        dimension : ``int``, required.
            The dimension of the layer output to normalize.

    Returns:
        The normalized layer output.
    """

    def __init__(self, dimension: int, eps: float = 1e-12, **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dimension))
        self.bias = torch.nn.Parameter(torch.zeros(dimension))
        self.eps: float = eps

    def forward(self, tensor: torch.Tensor):  # pylint: disable=arguments-differ
        # new_tensor = tensor.type_as(self.weight)
        new_tensor = tensor.to(dtype=self.weight.dtype)

        mean = new_tensor.mean(-1, keepdim=True)
        std = new_tensor.std(-1, unbiased=False, keepdim=True)
        res = self.weight * (new_tensor - mean) / (std + self.eps) + self.bias

        # res = res.type_as(tensor)
        res = res.to(dtype=tensor.dtype)
        return res

class TFLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, dimension: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

class T5LayerNorm(nn.Module):
    """ LayerNorm in T5 style, also named `rms_norm` officially. No bias and no subtraction of mean. """

    def __init__(self, dimension: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = None
        self.eps = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x

LayerNormTypes = {
    'default': nn.LayerNorm,
    'v0': LayerNorm,
    'tf': TFLayerNorm,
    't5': T5LayerNorm,
    'ft': FTLayerNorm,
}



# def transformer
class BeitRelPosBias(nn.Module):
    def __init__(self, window_size: Tuple[int, int], n_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.weight = nn.Parameter(
            torch.zeros(self.num_relative_distance, n_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("rel_pos_idx", relative_position_index)

    def forward(self):
        relative_position_bias = self.weight[self.rel_pos_idx.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


def rescue_inf_nan(x: torch.Tensor) -> torch.Tensor:
    if torch.isinf(x).any() or torch.isnan(x).any():
        clamp_value = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


@torch.jit.script
def hack_torch_trace(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    if b is None:
        return a
    return a + b


def _t5_rel_pos_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets


class AttentionExpert(nn.Module):

    def __init__(self, dim, dim2, LinearCls):
        super().__init__()
        self.proj_k = LinearCls(dim, dim2)
        self.proj_v = LinearCls(dim, dim2)

    def forward(self, x):
        k_states = self.proj_k(x)
        v_states = self.proj_v(x)
        return torch.concat([k_states, v_states], dim=2)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention from 'Attention Is All You Need' paper
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config: Config, is_decoder: bool = False, order: int = -1):
        super().__init__()
        if isinstance(config, dict):
            config = Config(**config)
        self.config = config
        self.is_decoder = is_decoder

        LinearCls = nn.Linear
        self.model_parallel_size = 1
        if self.config.use_mp_linear_in_attn:
            self.model_parallel_size = veGiantModel.distributed.get_model_parallel_world_size()
        self._use_ft_linear_transpose_fusion = self.config.use_ft_linear_transpose_fusion_in_attn and not self.config.use_ft_fused_attn
        self.faster_transpose = None
        self.faster_matmul = None

        self.l_aux = 0
        self.use_moe = False
        if not config.fuse_qkv_projs:
            if self.config.use_moe_attn and str(order) in self.config.use_moe_transformer_layer_attn.split(','):
                import janus.layer
                self.use_moe = True
                self.proj_moe = janus.layer.MoE(
                    hidden_size=config.moe_dim_attn,
                    expert=AttentionExpert(config.dim, config.dim, LinearCls),
                    num_experts=config.moe_experts_attn,
                    k=config.moe_k_attn,
                    noisy_gate_policy='None',
                    load_balanced=config.moe_load_balanced_attn,
                    enable_token_drop=False,
                    expert_shape=config.moe_attn_expert_shape)
            else:
                self.proj_k = LinearCls(config.dim, config.dim)
                self.proj_v = LinearCls(config.dim, config.dim)
            self.proj_q = LinearCls(config.dim, config.dim)
            self.proj_qkv = None
        else:
            assert not self.config.use_mp_linear_in_attn, "ColumnParallelLinear does not support `fuse_qkv_projs`"
            self.proj_qkv = LinearCls(config.dim, config.dim * 3)
            self.proj_k, self.proj_v, self.proj_q = None, None, None
        # in mp_linear mode, the num of heads & dim require adjustments
        self._use_mp_linear = self.config.use_mp_linear_in_attn and not issubclass(ColumnParallelLinear, MockModule)
        self.dropout = nn.Dropout(config.p_drop_attn)
        self.score_scale = config.head_dim ** -0.5


        self._use_ft_fused_attn = self.config.use_ft_fused_attn and not self.is_decoder  # TODO
        self.faster_attn = None

        if self.config.pos_emb_type == 'roformer':
            position_enc = np.array(
                [[pos / np.power(10000, 2 * (j // 2) / self.config.head_dim)
                  for j in range(self.config.head_dim)]
                 for pos in range(1024)])
            sin_pos = torch.repeat_interleave(torch.FloatTensor(np.sin(position_enc[:, 0::2])), 2, dim=-1)
            cos_pos = torch.repeat_interleave(torch.FloatTensor(np.cos(position_enc[:, 1::2])), 2, dim=-1)
            self.register_buffer('sin_pos', sin_pos[None, None, :, :])
            self.register_buffer('cos_pos', cos_pos[None, None, :, :])

    def _shape(self, tensor: torch.Tensor, bsz: int) -> torch.Tensor:
        """(bsz, seq_len, dim) -> (bsz, n_heads / model_parallel_size, seq_len, head_dim)"""
        n_heads = self.config.n_heads
        if self._use_mp_linear:
            n_heads = n_heads // self.model_parallel_size
        split_tensor = tensor.view(bsz, -1, n_heads, self.config.head_dim)
        return split_tensor.permute(0, 2, 1, 3).contiguous()

    def _shape2(self, tensor: torch.Tensor, bsz: int) -> torch.Tensor:
        """(bsz * n_heads / model_parallel_size, seq_len, head_dim) -> (bsz, seq_len, dim)
        """
        n_heads, dim = self.config.n_heads, self.config.dim
        if self._use_mp_linear:
            n_heads = n_heads // self.model_parallel_size
            dim = dim // self.model_parallel_size
        if self.config.mha_acts_unite_d01:
            tensor = tensor.view(bsz, n_heads, -1, self.config.head_dim)
        return tensor.permute(0, 2, 1, 3).reshape(bsz, -1, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_attn_weights: Optional[torch.Tensor] = None,
        gathered_mask_index: Optional[torch.Tensor] = None,
        gathered_hidden_states: Optional[torch.Tensor] = None,
        q_state: Optional[torch.Tensor] = None,  # Only for Deberta so far
        attn_bias: Optional[torch.Tensor] = None,  # e.g. T5-style rel pos attn bias
    ):
        """
        Args:
            hidden_states: shape (bsz, seq_len, dim)
            attn_mask: optional; shape (bsz, 1, tgt_len, src_len)
            key_value_states: optional; if provided, this layer is used as a cross-attention layer for decoder
            past_key_value: optional; tuple of (past key states, past value states)
            prev_attn_weights: optional; use realformer if provided
        """
        bsz, tgt_len, dim = hidden_states.size()
        if q_state is not None:  # TODO: ensure
            tgt_len = q_state.size(1)

        original_hidden_states = hidden_states
        remove_padding = gathered_hidden_states is not None
        if remove_padding:
            assert q_state is None, 'Remove padding must not accept special query states'
            hidden_states = gathered_hidden_states

        is_cross_attention = key_value_states is not None

        shaped_projected_q, shaped_projected_k, shaped_projected_v = None, None, None
        if self.config.fuse_qkv_projs:
            assert q_state is None, 'Fused QKV must not accept special query states'
            assert not is_cross_attention, 'Fused QKV must not be used with cross attention'
            assert not self.config.remove_padding, 'Fused QKV can not be used with `remove_padding` now'
            assert not self.config.use_mp_linear_in_attn, "Fused QKV with model parallelism is not implemented yet"
            projected_qkv = self.proj_qkv(hidden_states)  # (bsz, seq_len, dim * 3)
            shaped_projected_qkv = projected_qkv.view(bsz, -1, self.config.n_heads * 3, self.config.head_dim).view(bsz, -1, 3, self.config.n_heads, self.config.head_dim).permute(2, 0, 3, 1, 4).contiguous().view(3 * bsz * self.config.n_heads, -1, self.config.head_dim)
            shaped_projected_q, shaped_projected_k, shaped_projected_v = shaped_projected_qkv.chunk(3, dim=0)  # (bsz * n_heads, seq_len, head_dim)
            if not self.config.mha_acts_unite_d01:
                shaped_projected_q = shaped_projected_q.view(bsz, self.config.n_heads, -1, self.config.head_dim)
                shaped_projected_k = shaped_projected_k.view(bsz, self.config.n_heads, -1, self.config.head_dim)
                shaped_projected_v = shaped_projected_v.view(bsz, self.config.n_heads, -1, self.config.head_dim)

        if not remove_padding:
            q_states = self._shape(self.proj_q(hidden_states if q_state is None else q_state), bsz) if shaped_projected_q is None else shaped_projected_q  # (bsz, n_heads, seq_len, head_dim)
        else:
            q_states = torch.zeros_like(original_hidden_states)
            q_states.view(-1, dim)[gathered_mask_index] = self.proj_q(hidden_states)
            q_states = self._shape(q_states, bsz)  # (bsz, n_heads, seq_len, head_dim)

        # Get key, value proj
        if is_cross_attention:
            if remove_padding:
                raise NotImplementedError('Cross attention has not supported `remove_padding` yet')
            # Cross attention
            if past_key_value is not None:
                # Reuse k v
                k_states = past_key_value[0]
                v_states = past_key_value[1]
            else:
                k_states = self._shape(self.proj_k(key_value_states), bsz)  # (bsz, n_heads, seq_len, head_dim)
                v_states = self._shape(self.proj_v(key_value_states), bsz)  # (bsz, n_heads, seq_len, head_dim)
        else:
            # Self attention
            if not remove_padding:
                if shaped_projected_k is not None:
                    k_states = shaped_projected_k
                    v_states = shaped_projected_v
                else:
                    if self.use_moe:
                        states, self.l_aux, _ = self.proj_moe(hidden_states)
                        slice_line = states.shape[2] // 2
                        k_states_tmp, v_states_tmp = states[:, :, :slice_line].contiguous(), states[:, :, slice_line:].contiguous()
                    else:
                        k_states_tmp = self.proj_k(hidden_states)
                        v_states_tmp = self.proj_v(hidden_states)
                    k_states = self._shape(k_states_tmp, bsz)
                    v_states = self._shape(v_states_tmp, bsz)
            else:
                k_states = torch.zeros_like(original_hidden_states)
                k_states.view(-1, dim)[gathered_mask_index] = self.proj_k(hidden_states)
                k_states = self._shape(k_states, bsz)  # (bsz, n_heads, seq_len, head_dim)
                v_states = torch.zeros_like(original_hidden_states)
                v_states.view(-1, dim)[gathered_mask_index] = self.proj_v(hidden_states)
                v_states = self._shape(v_states, bsz)  # (bsz, n_heads, seq_len, head_dim)
            if past_key_value is not None:
                if remove_padding:
                    raise NotImplementedError('`past_key_value` has not supported `remove_padding` yet')
                # Reuse k v
                k_states = torch.cat([past_key_value[0], k_states], dim=2)
                v_states = torch.cat([past_key_value[1], v_states], dim=2)

        if self.config.pos_emb_type == 'roformer':
            ro_q = torch.stack([-q_states[..., 1::2], q_states[..., ::2]], dim=-1).reshape_as(q_states)
            ro_k = torch.stack([-k_states[..., 1::2], k_states[..., ::2]], dim=-1).reshape_as(k_states)
            q_states = q_states * self.cos_pos[:, :, :tgt_len] + ro_q * self.sin_pos[:, :, :tgt_len]
            k_states = k_states * self.cos_pos[:, :, :tgt_len] + ro_k * self.sin_pos[:, :, :tgt_len]

        src_len = k_states.size(-2)

        unite_d01 = self.config.mha_acts_unite_d01
        n_heads = self.config.n_heads
        if self._use_mp_linear:
            n_heads = n_heads // self.model_parallel_size
        proj_shape = (bsz * n_heads, -1, self.config.head_dim)

        if unite_d01:
            attn_weights = torch.bmm(q_states.view(proj_shape) * self.score_scale, k_states.view(proj_shape).transpose(-1, -2))  # (bsz * n_heads, seq_len, seq_len)
        else:
            attn_weights = torch.matmul(q_states * self.score_scale, k_states.transpose(-1, -2))  # (bsz, n_heads, seq_len, seq_len)

        if attn_bias is not None:
            if unite_d01:
                attn_weights = attn_weights.view(bsz, n_heads, tgt_len, src_len) + attn_bias
                attn_weights = attn_weights.view(bsz * n_heads, tgt_len, src_len)  # (bsz * n_heads, seq_len, seq_len)
            else:
                if self.training:
                    attn_weights = attn_weights + attn_bias
                else:
                    attn_weights = hack_torch_trace(attn_weights, attn_bias)  # HACK: walk-around for torch.jit.trace bug
        if attn_mask is not None:
            if unite_d01:
                attn_weights = attn_weights.view(bsz, n_heads, tgt_len, src_len) + attn_mask
                attn_weights = attn_weights.view(bsz * n_heads, tgt_len, src_len)  # (bsz * n_heads, seq_len, seq_len)
            else:
                if self.training:
                    attn_weights = attn_weights + attn_mask
                else:
                    attn_weights = hack_torch_trace(attn_weights, attn_mask)  # HACK: walk-around for torch.jit.trace bug

        if self.config.use_realformer and (prev_attn_weights is not None):
            if unite_d01:
                attn_weights = attn_weights + prev_attn_weights.view(bsz * n_heads, tgt_len, src_len)  # (bsz * n_heads, seq_len, seq_len)
            else:
                attn_weights = attn_weights + prev_attn_weights

        attn_probs = F.softmax(attn_weights, dim=-1)  # (bsz * n_heads, seq_len, seq_len)
        attn_probs = self.dropout(attn_probs)  # (bsz * n_heads, seq_len, seq_len)


        # (bsz * n_heads, tgt_seq_len, src_seq_len) * (bsz * n_heads, src_seq_len, head_dim) => (bsz * n_heads, tgt_seq_len, head_dim)
        if unite_d01:
            attn_outputs = torch.bmm(attn_probs, v_states.view(proj_shape))
        else:
            attn_outputs = torch.matmul(attn_probs, v_states)
        attn_outputs = self._shape2(attn_outputs, bsz)  # (bsz, seq_len, dim)
        if remove_padding:
            attn_outputs = attn_outputs.view(-1, dim)[gathered_mask_index]
        return (attn_outputs,
                None if self.config.omit_other_attn_output else (attn_probs if not unite_d01 else attn_probs.view(bsz, n_heads, tgt_len, src_len)),  # (bsz, n_heads, seq_len, seq_len)
                None if (self.config.omit_other_attn_output or self.config.use_ft_softmax) else (attn_weights if not unite_d01 else attn_weights.view(bsz, n_heads, tgt_len, src_len)),  # (bsz, n_heads, seq_len, seq_len)
                None if self.config.omit_other_attn_output else q_states,  # (bsz, n_heads, seq_len, head_dim)
                None if (self.config.omit_other_attn_output and not self.is_decoder) else k_states,  # (bsz, n_heads, seq_len, head_dim)
                None if (self.config.omit_other_attn_output and not self.is_decoder) else v_states,  # (bsz, n_heads, seq_len, head_dim)
                attn_bias)
        # return MHAOutput(
        #     attn_outputs=attn_outputs,
        #     attn_probs=None if self.config.omit_other_attn_output else (attn_probs if not unite_d01 else attn_probs.view(bsz, n_heads, tgt_len, src_len)),  # (bsz, n_heads, seq_len, seq_len)
        #     attn_weights=None if (self.config.omit_other_attn_output or self.config.use_ft_softmax) else (attn_weights if not unite_d01 else attn_weights.view(bsz, n_heads, tgt_len, src_len)),  # (bsz, n_heads, seq_len, seq_len)
        #     q_states=None if self.config.omit_other_attn_output else q_states,  # (bsz, n_heads, seq_len, head_dim)
        #     k_states=None if (self.config.omit_other_attn_output and not self.is_decoder) else k_states,  # (bsz, n_heads, seq_len, head_dim)
        #     v_states=None if (self.config.omit_other_attn_output and not self.is_decoder) else v_states,  # (bsz, n_heads, seq_len, head_dim)
        #     attn_bias=attn_bias,
        # )


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if self.config.use_mp_linear_in_ffn:
            assert ColumnParallelLinear is not None
            assert RowParallelLinear is not None
            self.fc1 = ColumnParallelLinear(config.dim, config.dim_ff)
            self.fc2 = RowParallelLinear(config.dim_ff, config.dim)
        else:
            self.fc1 = nn.Linear(config.dim, config.dim_ff)
            self.fc2 = nn.Linear(config.dim_ff, config.dim)
        self.act = Activation(config.act)
        self.dropout = nn.Dropout(config.p_drop_hidden)  # TODO: Is this dropout redundant?

    def forward(self, x) -> torch.Tensor:
        # (bsz, seq_len, dim) -> (bsz, seq_len, dim_ff / model_parallel_size) -> (bsz, seq_len, dim)
        fc1_out = self.act(self.fc1(x))
        if self.config.dropout_in_ffn:
            fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)
        if self.config.use_ffn_output_dropout:
            fc2_out = self.dropout(fc2_out)
        return fc2_out


class TransformerEncoderLayer(nn.Module):
    attention_class = MultiHeadAttention

    def __init__(self, config: Config, order: int = -1):
        super().__init__()
        if isinstance(config, dict):
            config = Config(**config)
        self.config = config
        self.dim = self.config.dim
        self.l_aux = -1
        self.l_aux_attn = -1

        self.attn = self.attention_class(config, order=order)
        self.fused_self_attn = None  # TODO: for ft inference

        if self.config.use_mp_linear_in_attn_out:
            assert RowParallelLinear is not None
            assert self.config.use_mp_linear_in_attn
        if self.config.use_mp_linear_in_attn_out:
            self.proj = RowParallelLinear(config.dim, config.dim)
        else:
            self.proj = nn.Linear(config.dim, config.dim)

        assert LayerNormTypes.get(config.layernorm_type, None) is not None, f"layernorm_type: {config.layernorm_type} Type returned None"
        self.norm1 = LayerNormTypes[config.layernorm_type](config.dim, config.layer_norm_eps)

        self.pwff = PositionWiseFeedForward(config)
        if self.config.use_moe and str(order) in self.config.use_moe_transformer_layer.split(','):
            import janus.layer
            self._use_moe = True
            if config.moe_warmup_stage:
                self.moe_warmup_k = list(map(int, config.moe_warmup_stage.split(',')))
                self.moe_warmup_steps = list(map(int, config.moe_warmup_steps.split(',')))
            else:
                self.moe_warmup_k = None
            self.pwff = janus.layer.MoE(
                hidden_size=config.moe_dim,
                expert=self.pwff,
                num_experts=config.moe_experts,
                k=config.moe_k,
                noisy_gate_policy=config.moe_noisy_gate_policy,
                load_balanced=config.moe_load_balanced,
                enable_token_drop=config.moe_enable_token_drop,
                expert_shape=config.moe_expert_shape)
        else:
            self._use_moe = False

        assert LayerNormTypes.get(config.layernorm_type, None) is not None, f"layernorm_type: {config.layernorm_type} Type returned None"
        self.norm2 = LayerNormTypes[config.layernorm_type](config.dim, config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.p_drop_hidden)
        self.dropout2 = nn.Dropout(config.p_drop_hidden2)

        if config.layernorm_fp16:
            self.norm1._simply_cast = True
            self.norm2._simply_cast = True

    def _hook_hidden_a_attn_out_b_res1(self, hidden_states, **kwargs):
        return hidden_states

    def _hook_hidden_a_ffn_b_res2(self, hidden_states, **kwargs):
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        prev_attn_weights: Optional[torch.Tensor] = None,
        gathered_mask_index: Optional[torch.Tensor] = None,
        gathered_hidden_states: Optional[torch.Tensor] = None,
        attn_mask_for_remove_pad: Optional[torch.Tensor] = None,
        word_index_for_remove_pad: Optional[torch.Tensor] = None,
        q_state: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        ffn_type_ids: Optional[torch.Tensor] = None,
    ):
        """ Transformer layer (encoder)

        Args:
            hidden_states (torch.Tensor): (bsz, seq_len, dim)
            attn_mask (torch.Tensor): (bsz, 1, tgt_len, src_len)
            prev_attn_weights (torch.Tensor, optional): (realformer) previous attention_weights. (bsz, n_heads, seq_len, seq_len)

        Returns:
            hidden_states (torch.Tensor): (bsz, seq_len, dim)
            attn_probs (torch.Tensor): (bsz, n_heads, seq_len, seq_len)
            attn_weights (torch.Tensor): (bsz, n_heads, seq_len, seq_len)
            (q_states, k_states, v_states): (bsz, n_heads, seq_len, head_dim)
        """
        bsz = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        seq_len_padded_even = False

        # Multi-Head Self-Attention
        original_hidden_states = hidden_states
        remove_padding = gathered_hidden_states is not None
        if remove_padding:
            assert not self.config.use_ft_remove_pad
            hidden_states = gathered_hidden_states

        if self.config.pad_seq_len_even:
            assert not remove_padding
            assert not self.config.use_ft_remove_pad
            if seq_len % 2 == 1:
                seq_len_padded_even = True
                # (bsz, seqlen, dim) -> (bsz, seqlen+1, dim)
                hidden_states = F.pad(hidden_states, (0, 0, 0, 1), 'constant', 0)
                # (bsz, 1, seqlen, seqlen) -> (bsz, 1, seqlen+1, seqlen+1)
                attn_mask = F.pad(attn_mask, (0, 1, 0, 1), 'constant', torch.finfo(attn_mask.dtype).min)
                if attn_bias is not None:
                    attn_bias = F.pad(attn_bias, (0, 1, 0, 1), 'constant', torch.finfo(attn_bias.dtype).min)

        word_idx = None

        residual = hidden_states if q_state is None else q_state  # TODO: ensure

        if self.config.use_pre_layernorm:
            hidden_states = self.norm1(hidden_states)

        (self_attn_output_hidden, self_attn_output_probs, self_attn_output_weights, self_attn_output_q_states,
            self_attn_output_k_states, self_attn_output_v_states, self_attn_output_bias) = self.attn(
            hidden_states if not remove_padding else original_hidden_states,
            attn_mask=attn_mask,
            prev_attn_weights=prev_attn_weights,
            gathered_mask_index=gathered_mask_index,
            gathered_hidden_states=gathered_hidden_states,
            q_state=q_state,
            attn_bias=attn_bias,
        )
        # self_attn_output: MHAOutput = self.attn(
        #     hidden_states if not remove_padding else original_hidden_states,
        #     attn_mask=attn_mask,
        #     prev_attn_weights=prev_attn_weights,
        #     gathered_mask_index=gathered_mask_index,
        #     gathered_hidden_states=gathered_hidden_states,
        #     q_state=q_state,
        #     attn_bias=attn_bias,
        # )
        if self.training:
            self.l_aux_attn = self.attn.l_aux
        # self_attn_output_hidden = self_attn_output.attn_outputs

        hidden_states = self.proj(self_attn_output_hidden)
        hidden_states = self.dropout1(hidden_states)

        # After out proj (including the last dropout), before the (first) residual addition
        hidden_states = self._hook_hidden_a_attn_out_b_res1(hidden_states)

        if self.config.use_deep_norm:
            residual = residual * self.config.deep_norm_enc_alpha

        hidden_states = residual + hidden_states
        if not self.config.use_pre_layernorm:
            hidden_states = self.norm1(hidden_states)

        # Position-wise Feed-Forward Networks
        residual = hidden_states
        if self.config.use_pre_layernorm:
            hidden_states = self.norm2(hidden_states)
        if self._use_moe:
            # hidden_states = self.pwff(hidden_states)
            # l_aux = self.pwff.l_aux
            import janus.groups
            if self.moe_warmup_k:
                now_step = janus.groups.get_step()
                k = -1
                for i in range(len(self.moe_warmup_steps)):
                    if now_step >= self.moe_warmup_steps[i]:
                        k = self.moe_warmup_k[i]
                self.pwff.set_topk(k)
            hidden_states, l_aux, _ = self.pwff(hidden_states)
            if self.training:
                self.l_aux = l_aux
                self.z_loss = self.pwff.get_z_loss()
        else:
            if (ffn_type_ids is None) or not getattr(self, '_use_n_ffn_types', False):  # TODO
                hidden_states = self.pwff(hidden_states)
            else:
                ffn_type_dim = ffn_type_ids.dim()
                # TODO: computation efficiency
                if ffn_type_dim == 2:  # shape of [bsz, seqlen], token-level moe
                    hidden_states_blank = torch.zeros_like(hidden_states)
                    for ffn_type_idx in range(len(self.pwff)):
                        # cur_hidden = torch.zeros_like(hidden_states)
                        # cur_ffn_type_index = ffn_type_ids == ffn_type_idx
                        # cur_ffn_type_state = hidden_states[cur_ffn_type_index]
                        # if torch.any(cur_ffn_type_index):
                        #     # cur_hidden[cur_ffn_type_index] = self.pwff[ffn_type_idx](cur_ffn_type_state.unsqueeze(0)).squeeze(0)
                        #     cur_hidden[cur_ffn_type_index] = self.pwff[ffn_type_idx](cur_ffn_type_state)
                        #     hidden_states_blank += cur_hidden
                        cur_hidden = self.pwff[ffn_type_idx](hidden_states)
                        cur_hidden[ffn_type_ids != ffn_type_idx] = 0
                        hidden_states_blank += cur_hidden
                    hidden_states = hidden_states_blank
                elif ffn_type_dim == 1:  # shape of [bsz], sequence-level moe
                    hidden_states_blank = torch.zeros_like(hidden_states)
                    for ffn_type_idx in range(len(self.pwff)):
                        cur_hidden = self.pwff[ffn_type_idx](hidden_states)
                        cur_hidden[ffn_type_ids != ffn_type_idx] = 0
                        hidden_states_blank += cur_hidden
                    hidden_states = hidden_states_blank
                elif ffn_type_dim == 0:  # shape of [], batch-level moe
                    ffn_type_idx = int(ffn_type_ids)
                    hidden_states = self.pwff[ffn_type_idx](hidden_states)
                else:
                    raise Exception(f'Unsupported dim of ffn_type_ids: {ffn_type_dim}')

        if not self.config.use_ffn_output_dropout:
            hidden_states = self.dropout2(hidden_states)

        # After ffn (including the last dropout), before the (second) residual addition
        hidden_states = self._hook_hidden_a_ffn_b_res2(hidden_states)

        if self.config.use_deep_norm:
            residual = residual * self.config.deep_norm_enc_alpha

        hidden_states = residual + hidden_states
        if not self.config.use_pre_layernorm:
            hidden_states = self.norm2(hidden_states)

        if self.config.clamp_inf_nan and (hidden_states.dtype == torch.float16):
            hidden_states = rescue_inf_nan(hidden_states)

        # attn_bias = self_attn_output.attn_bias
        attn_bias = self_attn_output_bias
        if seq_len_padded_even:
            hidden_states = hidden_states[:, :-1, :]
            if attn_bias is not None:
                attn_bias = attn_bias[:, :, :-1, :-1]
        return hidden_states, self_attn_output_probs, self_attn_output_weights, self_attn_output_q_states, \
            self_attn_output_k_states, self_attn_output_v_states, self_attn_output_bias


@torch.no_grad()
def _t5_rel_pos(qlen: int, klen: int, as_tensor: torch.Tensor, num_buckets: int, is_bidirectional: bool = True) -> torch.Tensor:
    """
    Args:
        is_bidirectional: seems true if encoder, false if decoder
    """
    context_position = torch.arange(
        qlen, dtype=torch.long, device=as_tensor.device
    )[:, None]
    memory_position = torch.arange(
        klen, dtype=torch.long, device=as_tensor.device
    )[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = _t5_rel_pos_bucket(
        relative_position,  # shape (qlen, klen)
        bidirectional=is_bidirectional,
        num_buckets=num_buckets,
    )
    return rp_bucket


class TransformerEncoder(nn.Module):
    layer_class = TransformerEncoderLayer

    def __init__(self, config: Config):
        super().__init__()
        if isinstance(config, dict):
            config = Config(**config)
        self.config = config
        self.dim = self.config.dim
        self.l_aux = list()
        self.z_loss = list()
        self.l_aux_attn = list()

        self.blocks = nn.ModuleList([self.layer_class(config, order=i) for i in range(config.n_layers)])
        self.final_norm = None
        if self.config.use_pre_layernorm:
            assert LayerNormTypes.get(self.config.layernorm_type, None) is not None, f"layernorm_type: {self.config.layernorm_type} Type returned None"
            self.final_norm = LayerNormTypes[self.config.layernorm_type](self.config.dim, self.config.layer_norm_eps)

            if config.layernorm_fp16:
                self.final_norm._simply_cast = True

        self._use_t5_rpe = self.config.pos_emb_type == 't5'
        if self._use_t5_rpe:
            self.t5_rpe = nn.Embedding(self.config.n_t5_rel_pos_buckets, self.config.n_heads)

        self._use_beit_rpe = self.config.pos_emb_type == 'beit'
        if self._use_beit_rpe:
            self.beit_rpe = BeitRelPosBias(tuple(self.config.beit_rel_pos_window_size), self.config.n_heads)

    def forward(self, hidden_states, attn_mask=None, use_namedtuple_output=True, ffn_type_ids=None):
        """
        Args:
            hidden_states: shape (bsz, seq_len, dim); embeddings
            attn_mask: shape (bsz, seq_len); Mask to avoid performing attention on padding token indices;
        """
        qlen = klen = hidden_states.size(1)
        original_hidden_states = hidden_states
        gathered_mask_index = None  # (num_masked,)
        gathered_hidden_states = None  # (num_masked, dim)
        if self.config.layer_grad_checkpoint:
            assert not self.config.remove_padding, '`layer_grad_checkpoint` does not support `remove_padding`'
            assert not self.config.use_realformer, '`layer_grad_checkpoint` does not support `use_realformer`'
        if self.config.remove_padding:
            assert attn_mask is not None, 'Transformer `remove_padding` should be with mask'
            assert attn_mask.dim() == 2, 'Transformer `remove_padding` should be with 2-D mask'
            # Use indexing
            flattened_mask_index = attn_mask.long().view(-1) * torch.arange(1, attn_mask.size(0) * attn_mask.size(1) + 1, device=hidden_states.device)
            gathered_mask_index = flattened_mask_index[flattened_mask_index != 0] - 1
            flattened_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            gathered_hidden_states = flattened_hidden_states[gathered_mask_index]
            # # Or use selecting
            # gathered_mask_index = torch.arange(0, attn_mask.size(0) * attn_mask.size(1), device=hidden_states.device).view(attn_mask.size(0), -1)
            # gathered_mask_index = torch.masked_select(gathered_mask_index, attn_mask.bool())
            # gathered_hidden_states = torch.index_select(hidden_states.view(-1, hidden_states.size(-1)), 0, gathered_mask_index)

        attn_mask_expanded = _expand_mask(attn_mask, hidden_states.dtype) if attn_mask.dim() != 4 else attn_mask

        all_hidden_states = []
        if 0 in self.config.return_layers:
            all_hidden_states.append(hidden_states)
        all_attn_weights = []
        all_attn_probs = []
        all_q_states = []
        all_k_states = []
        all_v_states = []
        prev_attn_weights = None

        attn_mask_for_remove_pad = None
        word_index_for_remove_pad = None

        l_aux = list()
        z_loss = list()
        l_aux_attn = list()
        attn_bias = None
        if self._use_t5_rpe:
            p = _t5_rel_pos(qlen, klen, self.t5_rpe.weight, self.t5_rpe.num_embeddings, is_bidirectional=True)
            attn_bias = self.t5_rpe(p)  # (qlen, klen, n_heads)
            attn_bias = attn_bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, qlen, klen)
        if self._use_beit_rpe:
            attn_bias = self.beit_rpe()

        for idx, block in enumerate(self.blocks):

            if self.config.layer_grad_checkpoint and self.training:

                # TODO: support ft options
                # TODO: support ffn_type_ids

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        n_inputs = len(inputs)
                        if n_inputs == 2:  # hidden_states attn_mask
                            return module(inputs[0], inputs[1]).hidden_states
                        if n_inputs == 3:  # hidden_states attn_mask attn_bias
                            return module(inputs[0], inputs[1], attn_bias=inputs[2]).hidden_states
                        if n_inputs == 4:  # hidden_states attn_mask attn_bias ffn_type_ids
                            return module(inputs[0], inputs[1], attn_bias=inputs[2], ffn_type_ids=inputs[3]).hidden_states
                        else:
                            raise Exception(f'Unsupported number ({n_inputs}) of args, for layer checkpointing function')

                    return custom_forward

                args_for_ckpt_func = (
                    hidden_states if not self.config.remove_padding else original_hidden_states,
                    attn_mask_expanded,
                    attn_bias,
                    ffn_type_ids,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    *args_for_ckpt_func,
                )
                transformer_outputs_hidden = hidden_states
                transformer_outputs_probs = None
                transformer_outputs_weights = None
                transformer_outputs_q_states = None
                transformer_outputs_k_states = None
                transformer_outputs_v_states = None
                transformer_outputs_bias = attn_bias
            else:
                if ffn_type_ids is None:  # For compat
                    (transformer_outputs_hidden, transformer_outputs_probs,
                     transformer_outputs_weights, transformer_outputs_q_states,
                     transformer_outputs_k_states, transformer_outputs_v_states,
                     transformer_outputs_bias) = block(
                        hidden_states if not self.config.remove_padding else original_hidden_states,
                        attn_mask_expanded,
                        prev_attn_weights,
                        gathered_mask_index=gathered_mask_index,
                        gathered_hidden_states=gathered_hidden_states,
                        attn_mask_for_remove_pad=attn_mask_for_remove_pad,
                        word_index_for_remove_pad=word_index_for_remove_pad,
                        attn_bias=attn_bias,
                    )
                else:
                    (transformer_outputs_hidden, transformer_outputs_probs,
                     transformer_outputs_weights, transformer_outputs_q_states,
                     transformer_outputs_k_states, transformer_outputs_v_states,
                     transformer_outputs_bias) = block(
                        hidden_states if not self.config.remove_padding else original_hidden_states,
                        attn_mask_expanded,
                        prev_attn_weights,
                        gathered_mask_index=gathered_mask_index,
                        gathered_hidden_states=gathered_hidden_states,
                        attn_mask_for_remove_pad=attn_mask_for_remove_pad,
                        word_index_for_remove_pad=word_index_for_remove_pad,
                        attn_bias=attn_bias,
                        ffn_type_ids=ffn_type_ids,
                    )

            attn_bias = transformer_outputs_bias

            hidden_states = transformer_outputs_hidden
            if self.config.use_pre_layernorm and (idx == self.config.n_layers - 1):
                hidden_states = self.final_norm(hidden_states)
            if self.config.remove_padding:
                gathered_hidden_states = hidden_states

            if self.config.use_realformer:
                prev_attn_weights = transformer_outputs_weights

            if (idx + 1) in self.config.return_layers:
                all_hidden_states.append(hidden_states)
                all_attn_weights.append(transformer_outputs_weights)
                all_attn_probs.append(transformer_outputs_probs)
                all_q_states.append(transformer_outputs_q_states)
                all_k_states.append(transformer_outputs_k_states)
                all_v_states.append(transformer_outputs_v_states)

            if self.training:
                if getattr(block, '_use_moe', False):
                    l_aux.append(block.l_aux)
                    z_loss.append(block.z_loss)
                if getattr(block.attn, 'use_moe', False):
                    l_aux_attn.append(block.l_aux_attn)

        if self.config.remove_padding:
            last_hidden_state = torch.zeros_like(original_hidden_states)
            last_hidden_state.view(-1, last_hidden_state.size(-1))[gathered_mask_index] = gathered_hidden_states
        else:
            last_hidden_state = hidden_states
        if self.training:
            self.l_aux = l_aux
            self.z_loss = z_loss
            self.l_aux_attn = l_aux_attn

        # if not use_namedtuple_output:
        #     raise NotImplementedError('TransformerEncoder should use TransformerEncoderOutput')
        return last_hidden_state, all_hidden_states, all_attn_probs, all_attn_weights, all_q_states, all_k_states, all_v_states


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None, mask_value: Optional[int] = None) -> torch.Tensor:
    """
    Expands attn_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if mask.dim() == 4:
        return mask
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    if mask_value is None:
        mask_value = torch.finfo(dtype).min
    return inverted_mask.masked_fill(inverted_mask.bool(), mask_value)



# def bert_moe
class Embedding(torch.nn.Module):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).

    Note that if you are using our data API and are trying to embed a TextField, you should use a
    TextFieldEmbedder instead of using this directly.

    Args:
        num_embeddings : int:
            Size of the dictionary of embeddings (vocabulary size).
        embedding_dim : int
            The size of each embedding vector.
        weight : torch.FloatTensor, (optional, default=None)
            A pre-initialised weight matrix for the embedding lookup, allowing the use of
            pretrained vectors.
        padding_index : int, (optional, default=None)
            If given, pads the output with zeros whenever it encounters the index.
        trainable : bool, (optional, default=True)
            Whether or not to optimize the embedding parameters.
        max_norm : float, (optional, default=None)
            If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type : float, (optional, default=2):
            The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq : boolean, (optional, default=False):
            If given, this will scale gradients by the frequency of the words in the mini-batch.
        sparse : bool, (optional, default=False):
            Whether or not the Pytorch backend should use a sparse representation of the embedding weight.

    Returns:
        An Embedding module.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 padding_idx: int = None,  # Compat with nn.Embedding
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index if padding_index is not None else padding_idx
        self.padding_idx = self.padding_index  # Compat with nn.Embedding
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.output_dim = embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise Exception("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

class AdaptiveEmbedding(nn.Module):
    """
    https://github.com/taufique74/AdaptiveIO/blob/0fd05695fd599b2706934959b76bb5ccb55521f3/model.py#L209
    https://zhuanlan.zhihu.com/p/98080699
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 cutoffs: Optional[List[int]] = None,
                 div_value: float = 2.0,
                 head_bias: bool = False,
                 tail_drop: float = 0.5):
        super().__init__()
        if not cutoffs:
            cutoffs = [5000, 10000]
        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) >= (num_embeddings - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and num_embeddings-1")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cutoffs = cutoffs + [num_embeddings]
        self.div_value = div_value
        self.head_bias = head_bias
        self.tail_drop = tail_drop

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0]

        self.head = nn.Embedding(self.head_size, self.embedding_dim, padding_idx=padding_idx)

        self.tail = nn.ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.embedding_dim // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = nn.Sequential(
                nn.Embedding(osz, hsz, padding_idx=padding_idx),
                nn.Linear(hsz, self.embedding_dim, bias=False),
                nn.Dropout(self.tail_drop)
            )

            self.tail.append(projection)

    def forward(self, input):
        used_rows = 0
        input_size = list(input.size())

        output = input.new_zeros([input.size(0) * input.size(1)] + [self.embedding_dim]).to(dtype=self.head.weight.dtype)
        input = input.view(-1)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            out = self.head(input[input_mask] - low_idx) if i == 0 else self.tail[i - 1](input[input_mask] - low_idx)
            output.index_copy_(0, row_indices, out)
            used_rows += row_indices.numel()

        # if used_rows != input_size[0] * input_size[1]:
        #     raise RuntimeError("Target values should be in [0, {}], "
        #                        "but values in range [{}, {}] "
        #                        "were found. ".format(self.num_embeddings - 1,
        #                                              input.min().item(),
        #                                              input.max().item()))
        return output.view(input_size[0], input_size[1], -1)

class LMPredictionHead(torch.nn.Module):
    def __init__(self, option):
        """
        Args:
            option:
                dim:
                embedding_dim:
                layer_norm_eps:
                vocab_size:
                layernorm_type:
                act:
        """
        super().__init__()
        embedding_dim = option.get('embedding_dim') or option.get('dim')
        self.embedding_dim = embedding_dim
        self.dense = (torch.nn.Linear)(option.get('dim'), embedding_dim)
        self.activation = Activation(option.get('act', 'gelu_new'))
        assert LayerNormTypes.get(option.get('layernorm_type', 'v0'), None) is not None, f"layernorm_type: {option.get('layernorm_type', 'v0')} Type returned None"
        if option.get('use_ft_layernorm_amap'):
            assert FTLayerNorm is not None, "FTLayerNorm is not supported while use_ft_layernorm_amap=True"
        self.layer_norm = (LayerNormTypes[option.get('layernorm_type', 'v0')] if not option.get('use_ft_layernorm_amap') else FTLayerNorm)(embedding_dim)

        self.decoder = torch.nn.Linear(embedding_dim, option.get('vocab_size'), bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(option.get('vocab_size')))

    def forward(self, hidden_states):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(torch.nn.Module):

    lm_prediction_head = LMPredictionHead

    def __init__(self, option):
        """
        Args:
            option:
                dim:
                embedding_dim:
                layer_norm_eps:
                vocab_size:
                layernorm_type:
        """
        super().__init__()
        self.predictions = self.lm_prediction_head(option)
        if option.get('use_ft_linear_amap'):
            assert FTLinear is not None, "FTLinear is not supported while use_ft_linear_amap=True"
        self.seq_relationship = (torch.nn.Linear if not option.get('use_ft_linear_amap') else FTLinear)(option.get('dim'), 2)

    def forward(self, sequence_output, pooled_output=None):
        prediction_scores = self.predictions(sequence_output)  # MLM
        seq_relationship_score = None
        if pooled_output is not None:
            seq_relationship_score = self.seq_relationship(pooled_output)  # NSP
        return prediction_scores, seq_relationship_score


class BertPooler(torch.nn.Module):
    def __init__(self, option):
        """
        Args:
            option:
                dim:
        """
        super().__init__()
        if option.get('use_ft_linear_amap'):
            assert FTLinear is not None, "FTLinear is not supported while use_ft_linear_amap=True"
        self.dense = (torch.nn.Linear if not option.get('use_ft_linear_amap') else FTLinear)(option.get('dim'), option.get('dim'))
        self.activation = torch.nn.Tanh()
        self.use_ft_linear = bool(option.get('use_ft_linear_amap'))

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if self.use_ft_linear:
            first_token_tensor = first_token_tensor.contiguous()
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def make_bert_inputs_from_tokens(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make inputs of `BertEmbedding`.

    Args:
        tokens (torch.Tensor): token ids, in shape of (batch_size, seq_len)
    """
    input_shape = tokens.size()
    device = tokens.device
    segments = torch.zeros(input_shape, dtype=torch.long, device=device)
    position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
    positions = position_ids.unsqueeze(0).expand(input_shape)
    return (tokens, segments, positions)


@torch.jit.script
def slice_pos_ids(position_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return position_ids[:, :input_ids.size(1)]


class BertEmbedding(torch.nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_segments: int,
        max_len: int,
        p_drop_hidden: float = 0.1,
        padding_index: int = None,
        layer_norm_eps: float = 1e-12,
        token_embedding_dim: int = None,
        output_dim: int = None,
        adaptive_option: Optional[dict] = None,
        layernorm_type: str = 'v0',
        position_offset: int = 0,
        use_dist_token_update: bool = False,
        max_seq_len: int = 512,
        dist_token_update_option: Optional[Dict] = None,
    ):
        super().__init__()

        token_embedding_dim = token_embedding_dim or dim

        dist_token_update_option = dist_token_update_option or {}
        dist_token_method = dist_token_update_option.get('method', 'v0')
        use_sparse_embedding = dist_token_update_option.get('use_sparse_embedding')
        if use_sparse_embedding is None:
            if not use_dist_token_update:
                use_sparse_embedding = False
            else:
                use_sparse_embedding = dist_token_method != 'v0'

        if token_embedding_dim != dim:
            assert not adaptive_option, 'Cannot use `adaptive_option` when `token_embedding_dim` is set.'
            self.token_embedder_tokens = Embedding(vocab_size, token_embedding_dim, padding_index=padding_index)
            self.token_embedding_proj = nn.Linear(token_embedding_dim, dim)
        else:
            self.token_embedding_proj = None
            if not adaptive_option:
                self.token_embedder_tokens = Embedding(vocab_size, dim, padding_index=padding_index, sparse=use_sparse_embedding)
            else:
                self.token_embedder_tokens = AdaptiveEmbedding(vocab_size, dim, padding_idx=padding_index, **adaptive_option)

        self.position_offset = position_offset or 0
        self.token_embedder_positions = Embedding(max_len + self.position_offset, dim) if max_len else None
        self.token_embedder_segments = Embedding(n_segments, dim) if n_segments else None

        self.dim = dim
        self.output_dim = output_dim or dim
        if self.output_dim != self.dim:
            self.out_proj = nn.Linear(self.dim, self.output_dim, bias=False)
        else:
            self.out_proj = None

        assert LayerNormTypes.get(layernorm_type, None) is not None, f"layernorm_type: {layernorm_type} Type returned None"
        self.norm = (LayerNormTypes[layernorm_type])(self.output_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(p_drop_hidden)

        if max_len:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer('position_ids', torch.arange(max_len + self.position_offset).expand((1, -1)))

        self.add_pos_embedding = True
        self.add_seg_embedding = True

        dist_token_absent_grad_zero = dist_token_update_option.get('set_absent_grad_zero', True)
        dist_token_clip_only = dist_token_update_option.get('use_clip_only', False)
        dist_token_grad_by_count = dist_token_update_option.get('avg_grad_by_count', True)
        dist_token_grad_fp32 = dist_token_update_option.get('use_grad_fp32', True)
        self._use_dist_token_update = use_dist_token_update
        self._max_seq_len = max_seq_len
        self._cached_input_ids = None
        self._use_cached_input_ids = self._use_dist_token_update and (dist_token_method == 'v0')
        if self._use_dist_token_update:
            from ptx.distributed import get_world_size, create_process_group, get_rank

            world_size = get_world_size()
            cur_rank = get_rank()
            pg = None
            if dist_token_method == 'v1':
                pg = create_process_group(backend='gloo')

            def all_gather_embedding_grad_v0(grad):
                if getattr(self, '_cached_input_ids', None) is None:
                    return grad
                emp_grad = torch.zeros_like(grad)
                if dist_token_absent_grad_zero or dist_token_clip_only:
                    cur_ids = self._cached_input_ids.unique()
                    emp_grad[cur_ids] = grad[cur_ids]
                    grad = emp_grad
                    if dist_token_clip_only:
                        self._cached_input_ids = None
                        return grad
                cur_ids = F.pad(self._cached_input_ids, (0, self._max_seq_len - self._cached_input_ids.size(1), 0, 0), 'constant', -1)
                # print(f'local input_ids: {self._cached_input_ids.shape}')
                self._cached_input_ids = None
                all_ids = [torch.full_like(cur_ids, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_ids)  # use `all_gather_object` instead?
                avg_grad_by = world_size
                if not dist_token_grad_by_count:
                    uni_ids = torch.cat(all_ids).unique()  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                else:
                    uni_ids, uni_counts = torch.cat(all_ids).unique(return_counts=True)  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                        uni_counts = uni_counts[1:]
                    avg_grad_by = uni_counts.unsqueeze(1)
                cur_grad = grad[uni_ids]
                cur_grad[cur_grad != cur_grad] = 0
                # print(f'before allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                if dist_token_grad_fp32:
                    cur_grad = cur_grad.float()
                torch.distributed.all_reduce(cur_grad)
                # print(f'after allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                all_grad = cur_grad / avg_grad_by
                if dist_token_grad_fp32:
                    emp_grad[uni_ids] = all_grad.to(dtype=emp_grad.dtype)
                else:
                    emp_grad[uni_ids] = all_grad
                # TODO: check nan/inf ?
                # emp_grad[emp_grad != emp_grad] = 0
                return emp_grad

            def all_gather_embedding_grad_v1(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                # print(f'grad index: {grad._indices()}')
                # print(f'grad value: {grad._values()}')
                if dist_token_grad_fp32:
                    grad = grad.float()
                torch.distributed.all_reduce(grad, group=pg)
                grad = grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # grad = grad.to_dense()  # Cannot, because grad tensor type must not change
                # print(grad)
                return grad

            def all_gather_embedding_grad_v2(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                all_grads = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_grads, grad)
                a_grad = None
                for i_grad in all_grads:
                    if not dist_token_grad_fp32:
                        i_grad = i_grad.to(device=grad.device)
                    else:
                        i_grad = i_grad.to(device=grad.device, dtype=torch.float32)
                    # print(f'grad after, {i_grad.is_coalesced()}')
                    # i_grad = i_grad.coalesce()
                    # i_indices = i_grad.indices()
                    # i_values = i_grad.values()
                    if a_grad is None:
                        a_grad = i_grad
                    else:
                        a_grad.add_(i_grad)
                grad = a_grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # print(grad)
                return grad

            def all_gather_embedding_grad_v3(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                cur_size = grad.size()
                cur_indices = grad.indices()
                cur_indices = cur_indices.squeeze(0)
                cur_values = grad.values()

                cur_seqlen = cur_indices.size(0)
                all_seqlen = [torch.tensor([0], device=grad.device) for _ in range(world_size)]
                torch.distributed.all_gather(all_seqlen, torch.tensor([cur_seqlen], device=grad.device))

                max_seqlen = torch.cat(all_seqlen).max().item()
                if max_seqlen > cur_seqlen:
                    cur_idx = torch.cat((cur_indices, torch.full((max_seqlen - cur_seqlen,), -1, dtype=cur_indices.dtype, device=cur_indices.device)))
                    cur_val = torch.cat((cur_values, torch.zeros((max_seqlen - cur_seqlen, cur_values.size(1)), dtype=cur_values.dtype, device=cur_values.device)))
                else:
                    cur_idx = cur_indices
                    cur_val = cur_values

                all_ids = [torch.full_like(cur_idx, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_idx)

                all_grads = [torch.zeros_like(cur_val) for _ in range(world_size)]
                torch.distributed.all_gather(all_grads, cur_val)

                for i, (idx, val) in enumerate(zip(all_ids, all_grads)):
                    if i == cur_rank:
                        continue
                    this_seqlen = all_seqlen[i].item()
                    this_grad = torch.sparse_coo_tensor(idx[:this_seqlen].unsqueeze(0), val[:this_seqlen], size=cur_size, dtype=grad.dtype, device=grad.device)
                    grad.add_(this_grad)

                grad = grad / world_size
                grad = grad.coalesce()
                # print(grad)
                return grad

            dist_token_methods = {
                'v0': all_gather_embedding_grad_v0,
                'v1': all_gather_embedding_grad_v1,
                'v2': all_gather_embedding_grad_v2,
                'v3': all_gather_embedding_grad_v3,
            }

            self.token_embedder_tokens.weight.register_hook(dist_token_methods[dist_token_method])

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, mask=None, embed_only: bool = False):
        if isinstance(input_ids, dict):
            token_type_ids = input_ids.get('segments')
            position_ids = input_ids.get('positions')
            inputs_embeds = input_ids.get('token_embeddings')
            input_ids = input_ids.get('tokens')

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if self.training and self._use_cached_input_ids:
            self._cached_input_ids = input_ids

        if inputs_embeds is None:
            embeddings = self.token_embedder_tokens(input_ids)
            if self.token_embedding_proj is not None:
                embeddings = self.token_embedding_proj(embeddings)
        else:
            embeddings = inputs_embeds

        if (self.token_embedder_positions is not None) and self.add_pos_embedding:
            if position_ids is None:
                # position_ids = self.position_ids[:, :input_ids.size(1)]
                position_ids = slice_pos_ids(self.position_ids, input_ids)
            if self.position_offset:
                position_ids = position_ids + self.position_offset
            position_embeddings = self.token_embedder_positions(position_ids)
            embeddings += position_embeddings

        if (self.token_embedder_segments is not None) and self.add_seg_embedding:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=embeddings.device)
            token_type_embeddings = self.token_embedder_segments(token_type_ids)
            embeddings += token_type_embeddings

        if embed_only:
            return embeddings

        if self.out_proj is not None:
            embeddings = self.out_proj(embeddings)

        embeddings = self.norm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings *= mask.to(embeddings.dtype)

        embeddings = self.dropout(embeddings)
        return embeddings


class SentenceClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


def init_weights(module, initializer_range=0.02):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (torch.nn.Embedding, Embedding)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if getattr(module, 'padding_idx', None) is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, LayerNorm):
        # ptx.ops.layernorm init itself
        pass
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class BertModel(torch.nn.Module):

    def __init__(self,
                 vocab_size: int,
                 dim: int = 768,
                 dim_ff: int = 3072,
                 n_segments: int = 2,
                 max_len: int = 512,
                 p_drop_hidden: float = 0.1,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 layernorm_type: str = 'v0',
                 layernorm_fp16: bool = False,
                 head_layernorm_type: str = 'v0',
                 use_realformer: bool = False,
                 attention_clamp_inf: bool = False,
                 p_drop_attn: float = 0.1,
                 embedding_dim: int = None,
                 token_embedding_dim: int = None,
                 embedding_dropout: float = None,
                 act: str = 'gelu_new',
                 padding_index: int = 0,  # pad token id
                 ignore_index: int = -1,
                 calc_mlm_accuracy: bool = True,
                 use_deepspeed_transformer: bool = False,
                 use_ls_transformer: bool = False,
                 extra_transformer_config: Optional[Dict] = None,
                 use_ft_linear_amap: bool = False,
                 use_ft_layernorm_amap: bool = False,
                 use_ft_softmax_amap: bool = False,
                 use_fused_xeloss: bool = False,
                 omit_other_output: bool = False,
                 ddp_params_and_buffers_to_ignore: Optional[List[str]] = None,
                 use_dist_token_update: bool = False,
                 max_seq_len: int = 512,
                 dist_token_update_option: Optional[Dict] = None,
                 tie_emb: bool = True,
                 ):
        super().__init__()
        if use_ft_linear_amap:
            assert FTLinear is not None
        if use_ft_layernorm_amap:
            assert FTLayerNorm is not None
        if use_ft_softmax_amap:
            assert FTSoftmax is not None

        self.bare_mode = False
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

        embedding_dim = embedding_dim or dim

        self.embedding = BertEmbedding(
            dim=embedding_dim, vocab_size=self.vocab_size,
            n_segments=n_segments, max_len=max_len,
            p_drop_hidden=p_drop_hidden if embedding_dropout is None else embedding_dropout,
            padding_index=padding_index,
            layer_norm_eps=layer_norm_eps,
            token_embedding_dim=token_embedding_dim,
            layernorm_type=layernorm_type,
            use_dist_token_update=use_dist_token_update,
            max_seq_len=max_seq_len,
            dist_token_update_option=dist_token_update_option,
        )
        if layernorm_fp16:
            self.embedding.norm._simply_cast = True

        self.proj_embedding_hidden = None
        if embedding_dim != dim:
            self.proj_embedding_hidden = (torch.nn.Linear if not use_ft_linear_amap else FTLinear)(embedding_dim, dim)

        self.extra_transformer_config = extra_transformer_config or {}
        self._use_moe = self.extra_transformer_config.get('use_moe', False)
        self._use_moe_attn = self.extra_transformer_config.get('use_moe_attn', False)
        tsfm_cls = TransformerEncoder

        self._omit_other_output = omit_other_output
        self._calc_mlm_accuracy = calc_mlm_accuracy and not self._omit_other_output
        self.mlm_accuracy = self._calc_mlm_accuracy and CategoricalAccuracy()

        self.encoder = tsfm_cls(TransformerConfig(
            n_layers, dim, n_heads, dim_ff, act=act,
            layernorm_type=layernorm_type,
            use_realformer=use_realformer,
            clamp_inf_nan=attention_clamp_inf,
            p_drop_hidden=p_drop_hidden, p_drop_attn=p_drop_attn,
            return_layers=list(range(n_layers + 1)) if not self._omit_other_output else [],
            layer_norm_eps=layer_norm_eps,
            **self.extra_transformer_config,
        ))

        self.pooler = BertPooler(dict(
            dim=dim,
            use_ft_linear_amap=use_ft_linear_amap,
        ))

        self.cls = BertPreTrainingHeads(dict(
            dim=dim,
            embedding_dim=embedding_dim,
            layer_norm_eps=layer_norm_eps,
            vocab_size=self.vocab_size,
            act=act,
            layernorm_type=head_layernorm_type,
            use_ft_linear_amap=use_ft_linear_amap,
            use_ft_layernorm_amap=use_ft_layernorm_amap,
        ))
        if layernorm_fp16:
            self.cls.predictions.layer_norm._simply_cast = True

        self.apply(partial(init_weights, initializer_range=self.initializer_range))
        self._tie_emb = tie_emb
        if self._tie_emb:
            self._tie_weights()
        else:

            # def output_embedding_grad_hook(grad):
            #     return grad

            # self.cls.predictions.decoder.weight.register_hook(output_embedding_grad_hook)
            pass

        self.padding_index = padding_index

        self.ignore_index = ignore_index
        xeloss_cls = torch.nn.CrossEntropyLoss

        self.loss_function = xeloss_cls(ignore_index=self.ignore_index)
        self.nsp_loss_function = xeloss_cls()
        self.ppl_xe = torch.nn.CrossEntropyLoss(ignore_index=self.padding_index)

        self.local_metrics = {}

        if use_dist_token_update:
            if not ddp_params_and_buffers_to_ignore:
                ddp_params_and_buffers_to_ignore = ["embedding.token_embedder_tokens.weight"]
        if ddp_params_and_buffers_to_ignore:
            if not use_dist_token_update:
                assert "embedding.token_embedder_tokens.weight" not in ddp_params_and_buffers_to_ignore
            self._ddp_params_and_buffers_to_ignore = ddp_params_and_buffers_to_ignore

    def _tie_weights(self):
        self.cls.predictions.decoder.weight = self.embedding.token_embedder_tokens.weight

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        global_metrics = {}
        if self._calc_mlm_accuracy:
            global_metrics['mlm_accuracy'] = self.mlm_accuracy.get_metric(reset)
        global_metrics.update(self.local_metrics)
        return global_metrics

    def _update_local_metrics(self, mlm_logits, mlm_labels):
        if self._calc_mlm_accuracy:
            total_count, correct_count = float(self.mlm_accuracy.total_count), float(self.mlm_accuracy.correct_count)
            mlm_positions = torch.nonzero(mlm_labels != self.ignore_index, as_tuple=False).view(-1)
            self.mlm_accuracy(mlm_logits[mlm_positions], mlm_labels[mlm_positions])
            local_total_count = float(self.mlm_accuracy.total_count) - total_count
            local_correct_count = float(self.mlm_accuracy.correct_count) - correct_count
            local_accuracy = 0.0 if local_total_count == 0 else (float(local_correct_count) / float(local_total_count))
            self.local_metrics.update({
                'local_mlm_total_count': local_total_count,
                'local_mlm_correct_count': local_correct_count,
                'local_mlm_accuracy': local_accuracy,
            })

    def forward(
        self,
        input_ids,
        position_ids=None,
        segment_ids=None,
        attention_mask=None,
        masked_tokens=None,  # (batch_size, seq_len)
        sentence_label=None,  # (batch_size, )
        masked_lm_positions=None,  # (batch_size, max_pred_len)
        masked_lm_ids=None,  # (batch_size, max_pred_len)
        masked_lm_weights=None,  # (batch_size, max_pred_len)
        calc_perplexity=False,
    ):
        embeddings = self.embedding(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            position_ids=position_ids,
        )
        if self.proj_embedding_hidden is not None:
            embeddings = self.proj_embedding_hidden(embeddings)

        if attention_mask is not None:
            mask = attention_mask
        else:
            mask = input_ids != self.padding_index
        # mask = mask.to(dtype=embeddings.dtype)

        (encoder_out_last_hidden_state,  # Tensor, shape (bsz, seq_len, dim); hidden_states of the last layer.
         encoder_out_all_hidden_states,  # list of Tensors, each shape (bsz, seq_len, dim); hidden_states of selected layers defined by config.return_layers.
         encoder_out_all_attn_probs,  # list of Tensors, each shape (bsz, n_heads, seq_len, seq_len); attn_probs of selected layers
         encoder_out_all_attn_weights,  # list of Tensors, each shape (bsz, n_heads, seq_len, seq_len); attn_weights of selected layers
         encoder_out_all_q_states,  # list of Tensors, each shape (bsz, n_heads, seq_len, head_dim); q_states of selected layers
         encoder_out_all_k_states,  # list of Tensors, each shape (bsz, n_heads, seq_len, head_dim); k_states of selected layers
         encoder_out_all_v_states  # list of Tensors, each shape (bsz, n_heads, seq_len, head_dim); v_states of selected layers
         ) = self.encoder(embeddings, mask)
        sequence_output = encoder_out_last_hidden_state  # (batch_size, seq_len, hidden_size)

        pooled_output = self.pooler(sequence_output)

        if self.bare_mode:
            output = {
                'sequence_output': sequence_output,
                'pooled_output': pooled_output,
            }

            output['all_q_states'] = encoder_out_all_q_states
            output['all_k_states'] = encoder_out_all_k_states
            output['all_v_states'] = encoder_out_all_v_states
            if self._use_moe and self.training:
                loss = 0
                for i in range(len(self.encoder.l_aux)):
                    loss += self.encoder.l_aux[i] * self.extra_transformer_config.get('moe_l_aux_factor', 0.01)
                output['loss'] = loss
            if self._omit_other_output:
                return output
            output.update({
                'mask': mask,
                'embedding': embeddings,
                'hidden': encoder_out_all_hidden_states,
                'attention': encoder_out_all_attn_weights,
            })
            return output

        # Shrink `sequence_output` and `masked_tokens` according to `masked_lm_positions` and `masked_lm_ids`
        positioned = masked_lm_positions is not None
        if (masked_tokens is not None) and positioned:
            masked_lm_positions_dim = masked_lm_positions.dim()
            if masked_lm_positions_dim == 2:
                sequence_output = gather_positions(sequence_output, masked_lm_positions)
                # Well, `ignore_index` may vary with this case
                masked_tokens = masked_lm_ids
                # masked_tokens = masked_lm_weights * masked_tokens
                # masked_tokens[masked_tokens == 0] = self.ignore_index
            elif masked_lm_positions_dim == 1:
                sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(-1))[masked_lm_positions]
                masked_tokens = masked_lm_ids
            else:
                raise Exception('Invalid dim of masked_lm_positions and masked_lm_ids')

        pred_score, seq_score = self.cls(sequence_output, pooled_output)

        mask_lm_loss, next_sentence_loss = 0.0, 0.0
        if masked_tokens is not None:
            mlm_logits = pred_score.view(-1, self.vocab_size)
            mlm_labels = masked_tokens.view(-1)
            mask_lm_loss = self.loss_function(mlm_logits, mlm_labels)
            if not self._omit_other_output:
                self._update_local_metrics(mlm_logits, mlm_labels)
                self.local_metrics['local_mlm_loss'] = mask_lm_loss.item()
        if sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(seq_score.view(-1, 2), sentence_label.view(-1))
            if not self._omit_other_output:
                self.local_metrics['local_nsp_loss'] = next_sentence_loss.item()

        total_loss = mask_lm_loss + next_sentence_loss
        if self._use_moe:
            for i in range(len(self.encoder.l_aux)):
                total_loss += self.extra_transformer_config.get('moe_l_aux_factor', 0.01) * self.encoder.l_aux[i]
                total_loss += self.extra_transformer_config.get('moe_z_loss_factor', 0.0) * self.encoder.z_loss[i]
                self.local_metrics['l_aux' + str(i)] = self.encoder.l_aux[i].item()
                self.local_metrics['z_loss' + str(i)] = self.encoder.z_loss[i].item()
        if self._use_moe_attn:
            for i in range(len(self.encoder.l_aux_attn)):
                total_loss += self.extra_transformer_config.get('moe_l_aux_factor_attn', 0.01) * self.encoder.l_aux_attn[i]
                self.local_metrics['l_aux_attn' + str(i)] = self.encoder.l_aux_attn[i].item()
        if self._omit_other_output:
            output = {'loss': total_loss}
        else:
            output = {
                'loss': total_loss,
                'mlm': pred_score,
                'pooled_output': pooled_output,
                'hidden': encoder_out_all_hidden_states,
                'attention': encoder_out_all_attn_weights,
            }
            if sentence_label is not None:
                output['nsp'] = seq_score

        if calc_perplexity and not self.training:
            ppl_loss = self.ppl_xe(pred_score.view(-1, self.vocab_size), input_ids.view(-1))
            perplexity = torch.exp(ppl_loss)
            output['perplexity'] = perplexity

        return output
