"""Import logics for xperf training libs. Ref: https://code.byted.org/data/xperf_training/blob/master/examples/faster_transformer.py"""
import os
import torch
from torch import nn, autograd
import logging

try:
    # import xperf_training
    # fast_transformer_dir = xperf_training.__path__[0]
    # fast_transformer_lib = os.path.join(fast_transformer_dir, 'libxperf_training_torch_ops_dyn.so')
    # torch.ops.load_library(fast_transformer_lib)
    import lego_ops
    lego_ops.load_ft_torch()

    class LayerNormFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, gamma, beta, residual=None, eps=1e-6):
            output, mean_var_rsqrt = torch.ops.FasterTransformer.LayerNorm_forward(
                input_tensor, gamma, beta, residual, eps)
            ctx.save_for_backward(input_tensor, gamma, mean_var_rsqrt, residual)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            grad_in, grad_gamma, grad_beta, grad_residual = torch.ops.FasterTransformer.LayerNorm_backward(
                grad_out, *ctx.saved_tensors)
            return grad_in, grad_gamma, grad_beta, grad_residual, None

    class FasterLayerNorm(nn.Module):
        def __init__(self, hidden_dim, eps=1e-6):
            super(FasterLayerNorm, self).__init__()

            self.hidden_dim = hidden_dim
            self.weight = nn.Parameter(torch.Tensor(hidden_dim))
            self.bias = nn.Parameter(torch.Tensor(hidden_dim))
            self.eps = eps
            nn.init.constant_(self.weight, 1.0)
            nn.init.constant_(self.bias, 0.0)

        def forward(self, input_tensor, residual=None):
            if self.training:
                return LayerNormFunction.apply(input_tensor, self.weight, self.bias, residual, self.eps)
            else:
                output, mean_var_rsqrt = torch.ops.FasterTransformer.LayerNorm_forward(
                    input_tensor, self.weight, self.bias, residual, self.eps)
                return output

        def extra_repr(self):
            return 'hidden_dim={}'.format(self.hidden_dim)

    FTLayerNorm = FasterLayerNorm

    class LinearFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, residual, weight, bias, act_gelu = False, dropout_rate = 0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            if act_gelu == True or dropout_rate > 0.0:
                output, bias_out, dropout_mask = torch.ops.FasterTransformer.Linear_forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
            else:
                output = torch.ops.FasterTransformer.Linear_forward(input_tensor, weight, bias, residual)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.act_gelu = act_gelu
            ctx.dropout_rate = dropout_rate
            ctx.has_residual = residual is not None
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            if ctx.act_gelu == True or ctx.dropout_rate > 0.0:
                grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.Linear_backward_gelu_dropout(
                    grad_out, input_tensor, weight, ctx.act_gelu, ctx.dropout_rate, bias_out, dropout_mask)
            else:
                grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.Linear_backward(
                    grad_out, input_tensor, weight)
            return grad_in, grad_out.detach().clone() if ctx.has_residual else None, grad_weight, grad_bias, None, None

    class FasterLinear(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super(FasterLinear, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias.data.zero_()
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

        def forward(self, input_tensor, residual=None):
            if self.training:
                return LinearFunction.apply(input_tensor, residual, self.weight, self.bias, self.act_gelu, self.dropout_rate)
            else:
                if self.act_gelu:
                    output, bias_out, dropout_mask = torch.ops.FasterTransformer.Linear_forward_gelu_dropout(input_tensor, self.weight, self.bias, self.act_gelu, 0.0)
                else:
                    output = torch.ops.FasterTransformer.Linear_forward(input_tensor, self.weight, self.bias, residual)
                return output

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    FTLinear = FasterLinear

    class FasterLinearWeightTransposed(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super(FasterLinearWeightTransposed, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))  # the weight is transposed
            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias.data.zero_()
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

        def forward(self, input_tensor, residual=None):
            weight_normal = self.weight.transpose(1, 0).contiguous()
            if self.training:
                return LinearFunction.apply(input_tensor, residual, weight_normal, self.bias, self.act_gelu, self.dropout_rate)
            else:
                if self.act_gelu:
                    output, bias_out, dropout_mask = torch.ops.FasterTransformer.Linear_forward_gelu_dropout(input_tensor, weight_normal, self.bias, self.act_gelu, 0.0)
                else:
                    output = torch.ops.FasterTransformer.Linear_forward(input_tensor, weight_normal, self.bias, residual)
                return output

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    FTLinearWeightTransposed = FasterLinearWeightTransposed


    class LinearTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            output = torch.ops.FasterTransformer.LinearTranspose_forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.LinearTranspose_backward(grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FasterLinearTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02):
            super(FasterLinearTranspose, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

    FTLinearTranspose = FasterLinearTranspose

    class LinearSplitTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            q_output, k_output, v_output = torch.ops.FasterTransformer.LinearSplitTranspose_forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return q_output, k_output, v_output

        @staticmethod
        def backward(ctx, q_grad_out, k_grad_out, v_grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.LinearSplitTranspose_backward(q_grad_out, k_grad_out, v_grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FasterLinearSplitTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type = "0213"):
            super(FasterLinearSplitTranspose, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias   = nn.Parameter(torch.Tensor(out_features))
            torch.nn.init.normal_(self.weight, mean=0, std=1)
            torch.nn.init.normal_(self.bias,   mean=0, std=1)

        def forward(self, input_tensor):
            return LinearSplitTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

    class TorchGatherFunction(autograd.Function):
        @staticmethod
        def forward(ctx, c2p_tensor, p2c_tensor, score_tensor, score_scaler):
            output = torch.ops.FasterTransformer.TorchGather_forward(c2p_tensor, p2c_tensor, score_tensor, score_scaler)
            ctx.score_scaler = score_scaler
            return output[0]

        @staticmethod
        def backward(ctx, grad_out):
            c2p_tensor_grad, p2c_tensor_grad, score_tensor_grad = torch.ops.FasterTransformer.TorchGather_backward(grad_out, ctx.score_scaler)
            return c2p_tensor_grad, p2c_tensor_grad, score_tensor_grad, None

    class FasterTorchGather(nn.Module):
        def __init__(self, score_scaler):
            super(FasterTorchGather, self).__init__()
            self.score_scaler = score_scaler

        def forward(self, c2p_tensor, p2c_tensor, score_tensor):
            return TorchGatherFunction.apply(c2p_tensor, p2c_tensor, score_tensor, self.score_scaler)

        def extra_repr(self):
            return 'score_scaler={}'.format(self.score_scaler)

    class FTDAGather(nn.Module):
        def __init__(self, score_scaler):
            super().__init__()

            self.score_scaler = score_scaler

        def forward(self, c2p_tensor, p2c_tensor, score_tensor):
            return TorchGatherFunction.apply(c2p_tensor, p2c_tensor, score_tensor, self.score_scaler)

        def extra_repr(self):
            return 'score_scaler={}'.format(self.score_scaler)

    class SoftmaxFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, mask_tensor, head_num = 1, dropout_rate = 0.0, batch_first = True):
            mask_tensor = mask_tensor.to(input_tensor.dtype)
            softmax_out, softmax_dropout_out, dropout_mask = torch.ops.FasterTransformer.Softmax_forward(
                input_tensor, mask_tensor, head_num, dropout_rate, batch_first)
            ctx.save_for_backward(softmax_out, dropout_mask)
            ctx.dropout_rate = dropout_rate
            return softmax_dropout_out if dropout_rate != 0.0 else softmax_out

        @staticmethod
        def backward(ctx, grad_out):
            softmax_out, dropout_mask = ctx.saved_tensors
            grad_in = torch.ops.FasterTransformer.Softmax_backward(
                grad_out, softmax_out, dropout_mask, ctx.dropout_rate)
            return grad_in, None, None, None, None

    def faster_softmax(input_tensor, mask_tensor = None, head_num = 1, dropout_rate = 0.0, batch_first = True):
        if torch.jit.is_tracing():
            return torch.ops.FasterTransformer.Softmax_infer(input_tensor, mask_tensor, head_num, dropout_rate, batch_first)
        else:
            return SoftmaxFunction.apply(input_tensor, mask_tensor, head_num, dropout_rate, batch_first)

    class FTSoftmax(nn.Module):
        def forward(self, input_tensor, mask_tensor, head_num, dropout_rate, batch_first):
            return SoftmaxFunction.apply(input_tensor, mask_tensor, head_num, dropout_rate if self.training else 0, batch_first)

    class TransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, transpose_type):
            transpose_out = torch.ops.FasterTransformer.Transpose4d_forward(input_tensor, transpose_type)
            ctx.transpose_type = transpose_type
            return transpose_out

        @staticmethod
        def backward(ctx, grad_out):
            grad_in = torch.ops.FasterTransformer.Transpose4d_backward(grad_out, ctx.transpose_type)
            return grad_in, None


    def faster_transpose(input_tensor, transpose_type = "0213"):
        return TransposeFunction.apply(input_tensor, transpose_type)

    def FTTransposeV1(transpose_type="0213"):
        default_transpose_type = transpose_type

        def faster_transpose(input_tensor, transpose_type = default_transpose_type):
            return TransposeFunction.apply(input_tensor, transpose_type)

        return faster_transpose


    class MatMulFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_A, input_B, transpose_a = False, transpose_b = False, scale = 1.0):
            matmul_out = torch.ops.FasterTransformer.MatMul_forward(
                input_A, input_B, transpose_a, transpose_b, scale)
            ctx.transpose_a = transpose_a
            ctx.transpose_b = transpose_b
            ctx.scale = scale
            ctx.save_for_backward(input_A, input_B)
            return matmul_out

        @staticmethod
        def backward(ctx, grad_out):
            input_A, input_B = ctx.saved_tensors
            grad_A, grad_B = torch.ops.FasterTransformer.MatMul_backward(
                grad_out, input_A, input_B, ctx.transpose_a, ctx.transpose_b, ctx.scale)
            return grad_A, grad_B, None, None, None

    def faster_matmul(input_A, input_B, transpose_a = False, transpose_b = False, scale = 1.0):
        input_B = input_B.to(input_A.dtype)
        if torch.jit.is_tracing():
            return torch.ops.FasterTransformer.MatMul_forward(input_A, input_B, transpose_a, transpose_b, scale)
        else:
            return MatMulFunction.apply(input_A, input_B, transpose_a, transpose_b, scale)

    FTMatMul = lambda: faster_matmul


    class RotaryEmbeddingFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_Q, input_K):
            output_Q, output_K = RotaryEmbedding.forward(input_Q, input_K)
            return output_Q, output_K

        @staticmethod
        def backward(ctx, grad_out_Q, grad_out_K):
            grad_Q, grad_K = RotaryEmbedding.backward(grad_out_Q, grad_out_K)
            return grad_Q, grad_K

    def faster_rotary_embedding(input_Q, input_K):
        return RotaryEmbeddingFunction.apply(input_Q, input_K)


    def faster_attention_infer(input_list, mask=None, head_num=1):
        input_len = len(input_list)
        if input_len == 3:  #input_q, input_k, input_v
            return torch.ops.FasterTransformer.FuseAttention_infer(*input_list, mask, head_num)
        elif input_len == 2:#input_q, input_kv
            return torch.ops.FasterTransformer.FuseAttention_infer_q_kv(*input_list, mask, head_num)
        elif input_len == 1:#input_qkv
            return torch.ops.FasterTransformer.FuseAttention_infer_qkv(*input_list, mask, head_num)
        else:
            print("Wrong input list")


    class GatedLinearUnitFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, dropout_rate = 0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            output, bias_out, dropout_mask = torch.ops.FasterTransformer.GatedLinearUnit_forward(input_tensor, weight, bias, dropout_rate)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.dropout_rate = dropout_rate
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.GatedLinearUnit_backward(
                grad_out, input_tensor, weight, ctx.act_gelu, ctx.dropout_rate, bias_out, dropout_mask)
            return grad_in, grad_weight, grad_bias, None


    class FasterGatedLinearUnit(nn.Module):
        def __init__(self, in_features, out_features, dropout_rate = 0.0):
            super(FasterGatedLinearUnit, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias   = nn.Parameter(torch.Tensor(out_features))
            torch.nn.init.normal_(self.weight, mean=0, std=1)
            torch.nn.init.normal_(self.bias,   mean=0, std=1)
            self.dropout_rate = dropout_rate

        def forward(self, input_tensor):
            if self.training:
                return GatedLinearUnitFunction.apply(input_tensor, self.weight, self.bias, self.dropout_rate)
            else:
                output, bias_out, dropout_mask = torch.ops.FasterTransformer.GatedLinearUnit_forward(input_tensor, self.weight, self.bias, self.dropout_rate)
                return output

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    class FuseAttentionFunction(autograd.Function):
        @staticmethod
        def forward(
            ctx, input_Q, input_K, input_V, softmax_mask, head_num, dropout_rate = 0.0):

            attention_out, softmax_out, dropout_mask, softmax_dropout_out = torch.ops.FasterTransformer.FuseAttention_forward(
                input_Q, input_K, input_V, softmax_mask, head_num, dropout_rate)

            ctx.save_for_backward(
                input_Q, input_K, input_V, softmax_out, dropout_mask, softmax_dropout_out)
            ctx.head_num = head_num
            ctx.dropout_rate = dropout_rate
            return attention_out

        @staticmethod
        def backward(ctx, grad_out):
            input_Q, input_K, input_V, softmax_out, dropout_mask, softmax_dropout_out = ctx.saved_tensors
            grad_Q, grad_K, grad_V = torch.ops.FasterTransformer.FuseAttention_backward(
                grad_out, softmax_out, input_Q, input_K, input_V, ctx.head_num, ctx.dropout_rate, dropout_mask, softmax_dropout_out)

            return grad_Q, grad_K, grad_V, None, None, None

    class FasterFuseAttention(nn.Module):
        def __init__(self, head_num, dropout_rate = 0.0):
            super(FasterFuseAttention, self).__init__()
            self.head_num = head_num
            self.dropout_rate = dropout_rate

        def forward(self, input_Q, input_K, input_V, softmax_mask):
            return FuseAttentionFunction.apply(
                input_Q, input_K, input_V, softmax_mask, self.head_num, self.dropout_rate)

    FTFusedAttention = FasterFuseAttention


    def _flash_attn_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                            dropout_p, attn_mask, attn_bias, softmax_scale, causal, return_softmax, num_splits=0,
                            generator=None):
        """
        num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
        it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
        Don't change it unless you know what you're doing.
        """
        softmax_lse, *rest = torch.ops.Extern.FlashAttn_forward(
            q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            softmax_scale, False, causal, return_softmax, num_splits, generator, attn_mask, attn_bias
        )
        # if out.isnan().any() or softmax_lse.isnan().any():
        #     breakpoint()
        S_dmask = rest[0] if return_softmax else None
        return out, softmax_lse, S_dmask

    def _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k, dropout_p, attn_mask, attn_bias, softmax_scale, causal, num_splits=0,
                             generator=None):
        """
        num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
        it will be set by an internal heuristic. Setting this too large (e.g. > 10) could make
        numerical error of dK and dV larger (scaling as sqrt(num_splits)).
        This hyperparameter can be tuned for performance, but default value (heuristic) should work fine.
        """
        softmax_d, *rest = torch.ops.Extern.FlashAttn_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal, num_splits, generator, attn_mask, attn_bias)
        # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
        #     breakpoint()
        dbias = None if attn_bias is None else rest[0]
        return dq, dk, dv, softmax_d, dbias

    class FlashAttnFunc(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                    attn_mask, attn_bias, softmax_scale, causal, return_softmax):
            # Save rng_state because the backward pass will regenerate the dropout mask
            rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            out, softmax_lse, S_dmask = _flash_attn_forward(
                q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p, attn_mask, attn_bias, softmax_scale, causal=causal, return_softmax=return_softmax
            )
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, attn_mask, attn_bias)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            return out if not return_softmax else (out, softmax_lse, S_dmask)

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, attn_mask, attn_bias= ctx.saved_tensors
            if rng_state is not None:
                cur_rng_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng_state)
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _, _, _, softmax_d, dbias = _flash_attn_backward(
                dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, attn_mask, attn_bias, ctx.softmax_scale, ctx.causal
            )
            if rng_state is not None:
                torch.cuda.set_rng_state(cur_rng_state)
            return dq, dk, dv, None, None, None, None, None, None, dbias, None, None, None

    def flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                 dropout_p, attn_mask = None, attn_bias = None, softmax_scale=None, causal=False, return_attn_probs=False):
        if torch.jit.is_tracing():
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            out, softmax_lse, S_dmask = _flash_attn_forward(
                q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p, attn_mask, attn_bias, softmax_scale, causal=causal, return_softmax=return_attn_probs)
            return out
        else:
            return FlashAttnFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                    dropout_p, attn_mask, attn_bias, softmax_scale, causal, return_attn_probs)

    def get_seq_len(attention_mask):
        return torch.ops.FasterTransformer.RemovePadding_get_seq_len(attention_mask)

    def faster_flash_attention(input_list, head_num, attn_mask = None, attn_bias = None, causal = False,
        cu_seqlens_q = None, cu_seqlens_k = None, max_seqlens_q = None, max_seqlens_k = None,
        softmax_scale = None, attention_dropout = 0.0, word_idx=None, use_rmpad_attn=False):
        input_count = len(input_list)
        assert input_count == 3
        if input_count == 3:
            q, k, v = input_list
            
            batch, seqlen, _ = q.shape

            if causal == False:
                if cu_seqlens_q is None:
                    max_seqlens_q = q.shape[1]
                    max_seqlens_k = k.shape[1]
            else:
                # assert attn_mask.shape[1] == 1
                if use_rmpad_attn:
                    max_seqlens_q = torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1])
                    max_seqlens_k = torch.max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])
                else:
                    max_seqlens_q = q.shape[1]
                    max_seqlens_k = k.shape[1]
                # attn_mask = attn_mask.type(q.dtype)
            if use_rmpad_attn:
                q = Compress_input(q.contiguous(), word_idx)
                k = Compress_input(k.contiguous(), word_idx)
                v = Compress_input(v.contiguous(), word_idx)
            q = q.view(-1, head_num, q.shape[-1] // head_num)
            k = k.view(-1, head_num, k.shape[-1] // head_num)
            v = v.view(-1, head_num, v.shape[-1] // head_num)

            if causal == False:
                output = flash_attn_unpadded_func(q, k, v,
                            cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k,
                            attention_dropout, attn_mask, attn_bias, softmax_scale, causal, False)
            else:
                if use_rmpad_attn:
                    output = flash_attn_unpadded_func(q, k, v,
                    cu_seqlens_q, cu_seqlens_k , max_seqlens_q, max_seqlens_k,
                    attention_dropout, None, None, None, causal, False)
                else:
                    output = flash_attn_unpadded_func(q, k, v,
                            None, None, max_seqlens_q, max_seqlens_k,
                            attention_dropout, None, None, None, causal, False)
        if use_rmpad_attn:
            output = output.view(output.shape[0], output.shape[1]*output.shape[2])
            output = Restore_output(output, word_idx, batch, seqlen)
        else:
            output = output.view(batch, seqlen, -1)
        return output

    FTFlashAttention = lambda: faster_flash_attention

    class CompressFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, word_idx):
            compress_input = torch.ops.FasterTransformer.RemovePadding_compress(input_tensor, word_idx)
            ctx.save_for_backward(word_idx)
            ctx.batch_size = input_tensor.shape[0]
            ctx.seq_len = input_tensor.shape[1]
            return compress_input

        @staticmethod
        def backward(ctx, grad_out):
            restore_grad_out = torch.ops.FasterTransformer.RemovePadding_restore(grad_out, ctx.saved_tensors[0], ctx.batch_size, ctx.seq_len)
            return restore_grad_out, None


    class RestoreFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, word_idx, batch_size, seq_len):
            restore_output = torch.ops.FasterTransformer.RemovePadding_restore(input_tensor, word_idx, batch_size, seq_len)
            ctx.save_for_backward(word_idx)
            return restore_output

        @staticmethod
        def backward(ctx, grad_out):
            compress_grad_out = torch.ops.FasterTransformer.RemovePadding_compress(grad_out, ctx.saved_tensors[0])
            return compress_grad_out, None, None, None

    Compress_input = CompressFunction.apply
    Restore_output = RestoreFunction.apply

    def Get_valid_word_index(attention_mask):
        return torch.ops.FasterTransformer.RemovePadding_get_valid_word_index(attention_mask)

    def Get_seq_len(attention_mask):
        return torch.ops.FasterTransformer.RemovePadding_get_seq_len(attention_mask)


except ImportError:
    logging.warning("Unable to import xperf_training, skip FT ops...")
    FTLinear = None
    FTLinearWeightTransposed = None
    FTTranspose = None
    FTTransposeV1 = None
    FTMatMul = None
    FTLinearTranspose = None
    FTDAGather = None
    FTSoftmax = None
    FTLayerNorm = None
    FTFusedAttention = None
    FTFlashAttention = None
