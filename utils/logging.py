def log(
    step: int,
    **metrics: dict
):
    parts = [f"step={step}"] + [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    print(" | ".join(parts))
