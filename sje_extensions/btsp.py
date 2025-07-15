import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import base, neuron, monitor

class BTSPLearner(base.MemoryModule):
    def __init__(self, synapse: nn.Linear, sn: neuron.BaseNode,
                 tau_et: float, tau_post_burst: float, tau_is: float,
                 burst_threshold: float,
                 lr_pot: float, lr_dep: float,
                 alpha_pot: float, beta_pot: float,
                 alpha_dep: float, beta_dep: float,
                 step_mode: str = 's'):
        super().__init__()
        self.step_mode = step_mode
        self.synapse = synapse
        
        # Store all BTSP-specific hyperparameters
        self.tau_et = tau_et
        self.tau_post_burst = tau_post_burst # Time constant for burst detection trace
        self.tau_is = tau_is                # Time constant for the IS trace (i_s)
        self.burst_threshold = burst_threshold
        self.lr_pot = lr_pot
        self.lr_dep = lr_dep
        self.alpha_pot = alpha_pot
        self.beta_pot = beta_pot
        self.alpha_dep = alpha_dep
        self.beta_dep = beta_dep

        # Pre-calculate normalization constants for the sigmoids
        high_pot = 1.0 / (1.0 + math.exp(-beta_pot * (1 - alpha_pot)))
        low_pot = 1.0 / (1.0 + math.exp(-beta_pot * (0 - alpha_pot)))
        self.norm_pot = high_pot - low_pot
        self.low_pot = low_pot
        # ... same for depression terms (high_dep, low_dep, norm_dep)

        # Set up monitors
        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

        # Register stateful traces
        self.register_memory('et', None)
        self.register_memory('post_trace', None) # For burst detection
        self.register_memory('i_s', None)        # The capped Instructive Signal

    def step(self, on_grad: bool = True, scale: float = 1.0):
        if self.step_mode != 's':
            raise NotImplementedError("Only single-step mode ('s') is shown here.")

        num_steps = len(self.in_spike_monitor.records)
        total_delta_w = 0.0

        for _ in range(num_steps):
            in_spike = self.in_spike_monitor.records.pop(0)
            out_spike = self.out_spike_monitor.records.pop(0)

            # Call our new single-step function
            delta_w_step = self.btsp_linear_single_step(in_spike, out_spike)
            total_delta_w += delta_w_step

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -total_delta_w * scale
            else:
                self.synapse.weight.grad -= total_delta_w * scale
        else:
            return total_delta_w * scale

    def btsp_linear_single_step(self, in_spike, out_spike):
        # Initialize traces on the first run
        if self.et is None:
            self.et = torch.zeros_like(in_spike)
        if self.post_trace is None:
            self.post_trace = torch.zeros_like(out_spike)
        if self.i_s is None:
            self.i_s = torch.zeros_like(out_spike)

        # 1. Update pre-synaptic eligibility trace 'et'
        self.et = self.et * math.exp(-1.0 / self.tau_et) + in_spike

        # 2. Update burst detection trace and the capped Instructive Signal 'i_s'
        # First, decay both post-synaptic traces
        self.post_trace = self.post_trace * math.exp(-1.0 / self.tau_post_burst)
        self.i_s = self.i_s * math.exp(-1.0 / self.tau_is)
        
        # Add incoming spikes to the burst detection trace
        self.post_trace += out_spike

        # Find which neurons have burst (exceeded the threshold)
        burst_detected_mask = (self.post_trace >= self.burst_threshold)
        
        # Reset i_s to 1 for neurons that have burst, capping its value
        if burst_detected_mask.any():
            self.i_s[burst_detected_mask] = 1.0
        
        # 3. Calculate the weight update based on the interaction of et and the capped i_s
        w = self.synapse.weight.data
        interaction = self.et.unsqueeze(1) * self.i_s.unsqueeze(2)

        # Potentiation calculation
        s_pot = torch.sigmoid(self.beta_pot * (interaction - self.alpha_pot))
        dw_pot = self.lr_pot * (1. - w) * ((s_pot - self.low_pot) / self.norm_pot)

        # Depression calculation
        s_dep = torch.sigmoid(self.beta_dep * (interaction - self.alpha_dep))
        dw_dep = self.lr_dep * w * ((s_dep - self.low_dep) / self.norm_dep)

        # The final delta_w is the sum of updates over the batch dimension
        delta_w = (dw_pot - dw_dep).sum(dim=0)
        
        return delta_w

