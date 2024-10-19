import torch
import torch.nn as nn
import torch.nn.functional as F
from scheduler import DDPMScheduler, DDIMScheduler
import os

class DiffusionPipeline(nn.Module):
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        
    def get_loss(self, x0, class_label=None):
        B = x0.shape[0]
        timesteps = self.scheduler.uniform_sample_t(B, x0.device)
        noise = torch.randn_like(x0, device=x0.device)
        x_noisy = self.scheduler.add_noise(x0, timesteps, noise)
        
        noise_pred = self.model(x_noisy, timesteps, class_label)
        
        loss = F.mse_loss(noise, noise_pred)
        
        return loss
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @torch.no_grad()
    def sample(self, batch_size=4, return_traj=False, class_labels=None):
        x_T = torch.randn([batch_size, 3, 32, 32], device=self.device)
        
        traj = [x_T]
        
        for t in self.scheduler.timesteps:
            x_t = traj[-1]
            timesteps = torch.full((batch_size,), t).to(self.device)
            noise_pred = self.model(x_t, timesteps, class_labels)
            
            x_t_prev = self.scheduler.step(sample=x_t, t=t, noise_pred=noise_pred)
            
            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())
        
        if return_traj:
            return traj
        else:
            return traj[-1]
        
    def save(self, file_path):
        hparams = {
            "model": self.model,
            "scheduler": self.scheduler
        }
        state_dict = self.state_dict()
        
        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path+"model.pt")
        
    def load(self, file_path):
        dic = torch.load(file_path, map_location="cuda")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]()

        self.model = hparams["model"]
        self.scheduler = hparams["scheduler"]

        self.load_state_dict(state_dict)
        
    @torch.no_grad()
    def inpaint(self, known, mask, num_inpaints=10):
        
        x_t = torch.randn_like(known, device=self.device)
        
        for t in self.scheduler.timesteps:
            timestep = torch.tensor([t]).to(self.device)
            eps = torch.randn_like(x_t, device=self.device) if t>0 else 0
            x_t_1_known = self.scheduler.add_noise(known, timestep, eps)
            
            noise_pred = self.model(x_t, timestep)
            x_t_1_unknown = self.scheduler.step(x_t, t, noise_pred) #Unknown
            
            x_t = mask * x_t_1_known + (1-mask) * x_t_1_unknown
        
        return x_t
                
        