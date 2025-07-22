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
        #self.v += torch.where(torch.tensor(self.v < self.f_min, device=x.device),
        self.v += torch.where(self.v < self.f_min,
                             (self.a * (self.v_reset - self.v)) * self.ss + x,
                             (self.b * (self.v - self.v_unstable)) * self.ss + x)
        self.v *= self.v >= 0.

class AdaptiveLIFNode(neuron.LIFNode):
    def __init__(self, n_exc=400, tau=100.0, **kwargs):
        super().__init__(tau=tau, **kwargs)

        # Adaptive threshold parameters
        self.theta_plus = 0.0038
        self.tau_theta = 1e7

        # Adaptive Threshold Buffer
        initial_theta = torch.full((1, n_exc), 1.0)
        self.register_buffer('theta', initial_theta)

    def neuronal_fire(self):
        # Fire by comparing membrane potential `v` to the adaptive threshold `theta`
        return self.surrogate_function(self.v - self.theta)

    def neuronal_reset(self, spike):
        # First, perform the standard LIF reset
        super().neuronal_reset(spike)
        
        # Then, apply the homeostatic updates to theta
        self.theta -= self.theta / self.tau_theta
        self.theta += self.theta_plus * spike

    def reset(self):
        # Also reset theta when the neuron is reset
        super().reset()
        self.theta.fill_(1.0)

