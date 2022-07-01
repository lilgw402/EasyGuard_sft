
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if len(tensor.shape) == 3:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return tensor
        elif len(tensor.shape) == 4:
            for b in tensor:
                for t, m, s in zip(b, self.mean, self.std):
                    t.mul_(s).add_(m)
                    # The normalize code -> t.sub_(m).div_(s)
            return tensor
