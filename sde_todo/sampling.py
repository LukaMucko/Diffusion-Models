import abc
import torch
from tqdm.notebook import tqdm
from jaxtyping import Float, Int

class Sampler():

    def __init__(self, eps: Float, device="cuda"):
        self.eps = eps
        self.device = "cuda"

    def get_sampling_fn(self, sde, dataset):

        def sampling_fn(N_samples: Int):
            """
            return the final denoised sample, number of step,
                   timesteps, and trajectory.

            Args:
                N_samples: number of samples

            Returns:
                out: the final denoised samples (out == x_traj[-1])
                ntot (int): the total number of timesteps
                timesteps Int[Array]: the array of timesteps used
                x_traj: the entire diffusion trajectory
            """
            x = dataset[range(N_samples)].to(self.device) # initial sample
            timesteps = torch.linspace(sde.T, self.eps, sde.N)
            x_traj = torch.zeros((sde.N, *x.shape))
            with torch.no_grad():
                for i, t in enumerate(tqdm(timesteps, desc='sampling', leave=True)):
                    t = torch.full((x.size(0),), t, device=self.device)
                    x_t = sde.predict_fn(t, x)
                    x_traj[i, :] = x_t
                    x = x_t  

            out = x_traj[-1]
            ntot = sde.N
            return out, ntot, timesteps, x_traj

        return sampling_fn
