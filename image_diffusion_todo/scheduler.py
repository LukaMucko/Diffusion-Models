from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, model, num_train_timesteps: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02, mode="linear",
        device="cuda"
    ):
        super().__init__()
        self.model = model
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.device = device
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        ).to(device)

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        # TODO: Compute alphas and alphas_cumprod
        # alphas and alphas_cumprod correspond to $\alpha$ and $\bar{\alpha}$ in the DDPM paper (https://arxiv.org/abs/2006.11239).
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) 

        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alphas", alphas.to(device))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(device))

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        model=None,
        num_train_timesteps: int = 1000,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode="linear",
        sigma_type="small",
        device="cuda"
    ):
        super().__init__(model=None, num_train_timesteps=num_train_timesteps, beta_1=beta_1, beta_T=beta_T, mode=mode, device=device)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        variances = torch.zeros(num_train_timesteps, device=self.device)
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                    [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]
                )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            )
            variances = torch.clamp(sigmas, min=1e-20)
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("variances", variances.to(self.device))
        self.register_buffer("sigmas", sigmas.to(self.device))

    def step(self, sample: torch.Tensor, t: int, noise_pred: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int`): current timestep in a reverse process.
            noise_pred (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t-1] if (t-1) >= 0 else torch.tensor(1.0, device=self.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        pred_original_sample = pred_original_sample.clamp(-1, 1)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        variance = 0
        if t>0:
            variance = self.variances[t]**0.5 * torch.randn_like(sample)
        return pred_prev_sample + variance

    def add_noise(
        self,
        sample: torch.Tensor,
        timesteps: torch.IntTensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            timesteps: (`torch.IntTensor [B]`)
            noise: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_noisy: (`torch.Tensor [B,C,H,W]`): noisy samples
            noise: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        # TODO: Implement the function that samples $\mathbf{x}_t$ from $\mathbf{x}_0$.
        # Refer to Equation 4 in the DDPM paper (https://arxiv.org/abs/2006.11239).
        t = timesteps
        if noise is None:
            noise = torch.randn_like(sample, device=sample.device)
            noisy_sample = torch.sqrt(self.alphas_cumprod[t]) * sample + torch.sqrt(1-self.alphas_cumprod[t]) * noise
            return noisy_sample, noise
    
        noisy_sample = torch.sqrt(self.alphas_cumprod[t])[:,None,None,None] * sample + torch.sqrt(1-self.alphas_cumprod[t])[:,None,None,None] * noise
        return noisy_sample



class DDIMScheduler(DDPMScheduler):
    def __init__(
        self,
        model=None,
        num_train_timesteps: int = 1000,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode="linear",
        sigma_type="small",
        device="cuda"
    ):        
        super().__init__(None, num_train_timesteps, beta_1, beta_T, mode, device=device)

        self.sigma_type = sigma_type
        variances = torch.zeros(num_train_timesteps, device=self.device)

        if sigma_type == "small":
            alphas_cumprod_t_prev = torch.cat(
                    [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]
                )
            beta_prod_t = 1 - self.alphas_cumprod
            beta_prod_t_prev = 1 - alphas_cumprod_t_prev
            sigmas = (beta_prod_t_prev / beta_prod_t) * (1 - self.alphas_cumprod / alphas_cumprod_t_prev)
            #variances = torch.clamp(sigmas, min=1e-20)
        
        # Register buffer for variances
        self.register_buffer("variances", variances.to(self.device))
    
    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        """
        Set timesteps for inference and move them to the correct device.
        """
        self.num_inference_timesteps = num_inference_timesteps

        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps).to(self.device)

    def step(
        self,
        sample: torch.Tensor,
        t: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
    ):
        """
        Perform one reverse DDIM step: x_t -> x_{t-1}.
        """
        timestep = t
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_timesteps
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device)

        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        pred_epsilon = noise_pred

        pred_original_sample = pred_original_sample.clamp(-1, 1)
        
        std_dev_t = eta * self.variances[timestep] ** 0.5
        
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            variance = std_dev_t * torch.randn_like(sample)
            return prev_sample + variance
        else:
            return prev_sample