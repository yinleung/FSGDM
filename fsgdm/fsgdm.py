from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

# Parts of this code are modifications of Pytorch's SGD optimizer

class FSGDM(Optimizer):
    r"""
    Frequency SGD with Momentum
    
    This is the official implementation of the FSGDM optimizer (version 1.0) in PyTorch.
    
    Paper: "On the Performance Analysis of Momentum Method: A Frequency Domain Perspective" 
    (https://arxiv.org/abs/2411.19671).
    
    Github repo: https://github.com/yinleung/FSGDM.
    
    This optimizer requires that users define n_stages (the number of stages) and sigma 
    (the number of gradient update steps) MANUALLY. The default settings of these two 
    arguments in this code are for CIFAR-100 experiments.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional):
            Learning rate (default: 1e-1)
        weight_decay (float, optional): 
            Weight decay (L2 penalty) (default: 5e-4)
        c_scaling (float):
            Scaling factor (default: 33e-3)
        v_coefficient (float):
            Momentum coefficient for FSGDM (default: 1.0)
        n_stages (int):
            The number of stages for FSGDM (default: 300)
        sigma (int):
            The number of gradient update steps (default: 117000)
    """

    def __init__(
            self, 
            params, 
            lr: Union[float, Tensor] = 1e-1, 
            weight_decay: float = 5e-4,
            c_scaling: float = 33e-3,
            v_coefficient: float = 1.0,
            n_stages: int = 300,
            sigma: int = 117000,
            **kwargs
        ):
            if lr < 0.0:
                raise ValueError(f"Invalid learning rate: {lr}")
            if weight_decay < 0.0:
                raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            if c_scaling < 0.0:
                raise ValueError(f"Invalid scaling factor value: {c_scaling}")
            if v_coefficient < 0.0:
                raise ValueError(f"Invalid momentum coefficient value: {v_coefficient}")
            if n_stages <= 0.0:
                raise ValueError(f"Invalid number of stages value: {n_stages}")
            if sigma <= 0.0:
                raise ValueError(f"Invalid number of gradient update steps value: {sigma}")
            defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                c_scaling=c_scaling,
                v_coefficient=v_coefficient,
                n_stages=n_stages,
                sigma=sigma,
                step=1.0,
            )
            super().__init__(params, defaults) 
  
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """   
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            c_scaling = group['c_scaling']
            v_coefficient = group['v_coefficient']
            n_stages = group['n_stages']
            sigma = group['sigma']
            step = group['step']
            
            step_per_epoch = sigma // n_stages # steps per epoch
            
            stage = (step // step_per_epoch) * step_per_epoch # stage number
            
            mu = c_scaling * sigma # mu coefficient
            
            # update momentum coefficients
            u = stage / (stage + mu)
            v = v_coefficient

            step = step + 1
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])
                    

            fsgdm_step(params_with_grad,
                   d_p_list,
                   momentum_buffer_list,
                   weight_decay=weight_decay,
                   u=u,
                   v=v,
                   lr=group['lr'],)
            
            group['step'] = step
            
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        
        return loss

def fsgdm_step(params: List[Tensor],
         d_p_list: List[Tensor],
         momentum_buffer_list: List[Optional[Tensor]],
         *,
         weight_decay: float,
         u: float,
         v: float,
         lr: float):

    for i, param in enumerate(params):
        
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        
        buf = momentum_buffer_list[i]

        if buf is None:
            buf = torch.clone(d_p).detach()
            momentum_buffer_list[i] = buf
        else:
            buf.mul_(u).add_(d_p, alpha=v)

        d_p = buf
        
        param.add_(d_p, alpha=-lr)