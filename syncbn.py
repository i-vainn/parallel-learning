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
        
        sum = input.sum(0)
        sum_squares = torch.square(input).sum(0)
        batch_msg = torch.tensor([input.shape[0]], device=input.device)

        msg_tensor = torch.cat([
            sum, sum_squares, batch_msg
        ])
        dist.all_reduce(msg_tensor)
        global_sum, global_sum_squares, global_batch_size = torch.split(msg_tensor, input.shape[1])

        global_mean = global_sum / global_batch_size.item()
        global_var = (global_sum_squares - (global_sum ** 2) / global_batch_size) / global_batch_size
        global_std = torch.sqrt(global_var + eps)

        running_mean = running_mean * (1 - momentum) + global_mean * momentum
        running_std = running_std * (1 - momentum) + global_std * momentum
        
        norm_input = (input - running_mean) / running_std
        ctx.save_for_backward(norm_input, running_std, global_batch_size)

        return norm_input

    @staticmethod
    def backward(ctx, grad_output):
        norm_input, global_std, global_batch_size = ctx.saved_tensors

        sum_gout = grad_output.sum(0)
        sum_prod_gout = (grad_output * norm_input).sum(0)

        msg_tensor = torch.cat([
            sum_gout, sum_prod_gout
        ])
        dist.all_reduce(msg_tensor)
        sum_gout, sum_prod_gout = torch.split(msg_tensor, norm_input.size(1))

        grad_output = (
            global_batch_size * grad_output - 
            sum_gout - norm_input * sum_prod_gout
        ) / global_batch_size / global_std

        return grad_output, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 1.):
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
        self.bn_func = sync_batch_norm.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training and self.track_running_stats:
            std = torch.sqrt(self.running_var + self.eps)
            return (input - self.running_mean) / std
        else:
            return self.bn_func(input, self.running_mean, self.running_var, self.eps, self.momentum)
