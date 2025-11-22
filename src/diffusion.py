import numpy as np
import torch

def get_linear_alphas_cumprod(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Calculate alphas_cumprod (based on a linear schedule)
    """
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=np.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod

def ddim_deterministic(
    x_start,
    model,
    alphas_cumprod,
    ddim_timesteps,
    device,
    logger=None,
    requires_grad=False,
):
    """
    DDIM deterministic diffusion (forward or reverse).

    Args:
        x_start: starting tensor (x0 for forward, x_t for reverse)
        model: UNet diffusion model
        alphas_cumprod: precomputed alpha cumprod tensor
        ddim_timesteps: list of timesteps
        device: torch device
        logger: optional logger
        requires_grad: if False, disables gradient tracking (default)
    Returns:
        x_out: resulting tensor (xt for forward, x0 for reverse)
    """
    
    grad_context = torch.enable_grad if requires_grad else torch.no_grad
    with grad_context():
        if logger is not None:
            logger.info(f"Starting DDIM diffusion with {len(ddim_timesteps)} steps.")

        x = x_start.clone()
        B = x.shape[0]
        alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)

        for i in range(len(ddim_timesteps) - 1):
            t = int(ddim_timesteps[i])
            t_next = int(ddim_timesteps[i + 1])

            if logger is not None:
                logger.info(f"DDIM step {i+1}/{len(ddim_timesteps)-1}: {t} -> {t_next}")

            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

            # Predict noise
            out = model(x, t_tensor)
            # If model predicts mean and variance, only take the mean
            eps = out[:, :3] if out.shape[1] == 6 else out

            # Fetch alpha bars
            alpha_bar_t = alphas_cumprod[t_tensor].view(B, 1, 1, 1)
            alpha_bar_next = alphas_cumprod[t_next].view(B, 1, 1, 1)

            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_ab_next = torch.sqrt(alpha_bar_next)
            sqrt_one_minus_ab_next = torch.sqrt(1.0 - alpha_bar_next)

            # Compute predicted x0
            x0_pred = (x - sqrt_one_minus_ab_t * eps) / sqrt_ab_t

            # Deterministic DDIM update
            x = sqrt_ab_next * x0_pred + sqrt_one_minus_ab_next * eps

        if logger is not None:
            logger.info(f"DDIM diffusion completed.")
    return x