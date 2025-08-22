import math

class WarmupCosine:
    def __init__(
        self, 
        base_lr: float,
        warmup_steps: int,
        max_steps: int
    ):   
        self.base_lr = base_lr
        self.warmup = max(1, warmup_steps)
        self.max_steps = max_steps

    def lr_at(
        self, 
        step: int
    ) -> float:

        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup

        progress = (step - self.warmup) / max(1, (self.max_steps - self.warmup))
        
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))
