"""
Scheduler

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


import math
import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            milestones=[rate * total_steps for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
        verbose=False,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps, power=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps, gamma=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, total_steps, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class TwoOneCycleLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=(0.3, 0.3),
        div_factor=(25.0, 25.0),
        anneal_strategy=("cos", "cos"),
        final_div_factor=(1e4, 1e4),
        last_epoch=-1,
        verbose=False,
    ):

        super().__init__(
            optimizer=optimizer,
            lr_lambda=[
                lambda step: self.one_cycle_lambda(
                    step,
                    total_steps,
                    max_lr[0],
                    div_factor[0],
                    pct_start[0],
                    anneal_strategy=anneal_strategy[0],
                    final_div_factor=final_div_factor[0]
                ),
                lambda step: self.one_cycle_lambda(
                    step,
                    total_steps,
                    max_lr[1],
                    div_factor[0],
                    pct_start[1],
                    anneal_strategy=anneal_strategy[1],
                    final_div_factor=final_div_factor[1]
                )
            ],
            last_epoch=last_epoch,
            verbose=verbose,
        )

    # Define the OneCycleLR lambda logic
    @staticmethod
    def one_cycle_lambda(step, total_steps, max_lr, div_factor, pct_start, anneal_strategy, final_div_factor):
        """
        Lambda function for OneCycleLR.
        """
        min_lr = max_lr / div_factor
        final_lr = max_lr / final_div_factor
        pct = step / total_steps

        if pct < pct_start:  # Warm-up phase
            scale = pct / pct_start
            lr = min_lr + (max_lr - min_lr) * scale
        else:  # Annealing phase
            scale = (pct - pct_start) / (1 - pct_start)
            if anneal_strategy == "cos":
                # Cosine annealing: smooth decay from max_lr to final_lr
                lr = final_lr + 0.5 * (max_lr - final_lr) * (1 + math.cos(scale * math.pi))
            elif anneal_strategy == "linear":
                lr = max_lr - (max_lr - final_lr) * scale
            else:
                raise ValueError(f"Invalid anneal_strategy: {anneal_strategy}")

        return lr / max_lr  # Normalize to use as a scaling factor


def build_scheduler(cfg, optimizer):
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
