import torch
from typing import Callable
from spikingjelly.activation_based import neuron, surrogate

class IQIFNode(neuron.BaseNode):

    def __init__(self, a: float = 1., b: float = 1., v_threshold: float = 1.,
                 v_reset: float = 0., v_unstable: float = 0.5,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode = 's',
                 backend = 'torch', store_v_seq: bool = False):

        assert isinstance(a, float) and a > 0.
        assert isinstance(b, float) and b > 0.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.a = a
        self.b = b
        self.v_unstable = v_unstable
        self.f_min = (a * v_reset + b * v_unstable) / (a + b)
        self.ss = 0.125

    def extra_repr(self):
        return super().extra_repr() + f', a={self.a}, b={self.b}, v_unstable={self.v_unstable}, f_min={self.f_min}, ss={self.ss}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v += torch.where(torch.tensor(self.v < self.f_min, device=x.device),
                             (self.a * (self.v_reset - self.v)) * self.ss + x,
                             (self.b * (self.v - self.v_unstable)) * self.ss + x)
        self.v *= self.v >= 0.

