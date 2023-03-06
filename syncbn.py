import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        mean = input.mean(1)
        mean_squares = torch.square(input).mean(1)
        communication_tensor = torch.cat([mean, mean_squares])
        dist.all_reduce(communication_tensor)
        communication_tensor /= dist.get_world_size()
        cur_mean = communication_tensor[:mean.size(0)]
        cur_var = communication_tensor[mean.size(0):] - cur_mean ** 2
        norm_input = (input - cur_mean) / torch.sqrt(cur_var + eps)
        ctx.save_for_backward(cur_mean, cur_var, norm_input, running_std)

        return norm_input * running_std + running_mean

    @staticmethod
    def backward(ctx, grad_output):
        cur_mean, cur_var, norm_input, running_std = ctx.saved_tensors
        grad_running_std = norm_input * grad_output
        grad_running_mean = norm_input
        grad_output = grad_output * running_std
        # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/_functions.py
        # https://kevinzakka.github.io/2016/09/14/batch_normalization/
        # https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
        


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        pass
