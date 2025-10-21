import numpy as np
import torch
import torchvision.transforms as T

def make_ddim_timesteps(num_timesteps_total, T_remov):
    """
    Make evenly spaced descending list of timesteps for DDIM reverse process.
    """
    assert T_remov>=1 and T_remov<=num_timesteps_total
    timesteps = np.linspace(0, num_timesteps_total-1, T_remov, dtype=int)
    timesteps = timesteps[::-1] #reverse for descending order
    return timesteps

@torch.no_grad()
def ddim_reverse_deterministic(x_t, model, diffusion, ddim_timesteps, device):
    """
    Perform DDIM deterministic reverse diffusion from x_t to estimate x0.
    Args:
        x_t: noised input tensor at timestep t with shape [B, C, H, W]
        model: UNet diffusion model that predicts noise for input (x, t)
        diffusion: diffusion process object
        ddim_timesteps: list/array of timesteps for DDIM reverse process
        device: torch device
    
    Returns:
        x0_est: estimated clean image tensor with shape [B, C, H, W]
    """
    x = x_t
    B = x.shape[0]

    #retrieve alphas from diffusion process
    alphas_cumprod = diffusion.alphas_cumprod

    for i in range(len(ddim_timesteps)-1):
        t = int(ddim_timesteps[i])
        t_prev = int(ddim_timesteps[i+1])
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        t_prev_tensor = torch.full((B,), t_prev, dtype=torch.long, device=device)

        #predict noise eps with pre-trained model
        out = model(x, t_tensor)
        # If model predicts both mean and variance
        if out.shape[1] == 6:
            eps, logvar = torch.split(out, 3, dim=1)
        else:
            eps = out
            
        #fetch alpha bars
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_prev = alphas_cumprod[t_prev]

        #compute x0_est
        x0_pred = (x - np.sqrt(1.0 - alpha_bar_t) * eps) / np.sqrt(alpha_bar_t)

        #deterministic DDIM update (eta = 0)
        x = np.sqrt(alpha_bar_prev) * x0_pred + np.sqrt(1.0 - alpha_bar_prev) * eps

    # Final step: move all the way to t=0 estimation if last timestep isn't 0
    last_t = int(ddim_timesteps[-1])
    if last_t != 0:
        t_tensor = torch.full((B,), last_t, dtype=torch.long, device=device)
        out = model(x, t_tensor)
        # If model predicts both mean and variance
        if out.shape[1] == 6:
            eps, logvar = torch.split(out, 3, dim=1)
        else:
            eps = out

        alpha_bar_t = alphas_cumprod[last_t]
        x0_pred = (x - np.sqrt(1.0 - alpha_bar_t) * eps) / np.sqrt(alpha_bar_t)
        x = x0_pred  # move to estimated x0

    return x