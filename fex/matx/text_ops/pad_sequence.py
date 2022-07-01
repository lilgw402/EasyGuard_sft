import torch
from typing import List


def pad_sequence(sequences: List[torch.Tensor],
                 padding_value: float,
                 max_len: int = 4):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``B x T x *``
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    size = [s.size(0) for s in sequences]
    max_len = max(max_len, max(size))
    min_len = min(size)
    # 尽量在eternal上使用固定的padding size，这样速度更快 @yejiandong
    if max_len == min_len:
        # return torch.stack(sequences)
        return torch.cat(sequences).view([len(sequences), ] + list(max_size))

    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor
