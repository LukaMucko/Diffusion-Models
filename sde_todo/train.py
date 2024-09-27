import torch
from tqdm.notebook import tqdm
from itertools import repeat
import matplotlib.pyplot as plt
from loss import ISMLoss, DSMLoss

def freeze(model):
    """
    (Optional) This is for Alternating Schrodinger Bridge.
    """
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def unfreeze(model):
    """
    (Optional) This is for Alternating Schrodinger Bridge.
    """
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    return model


def get_sde_step_fn(model, ema, opt, loss_fn, sde):
    def step_fn(batch):
        # uniformly sample time step
        t = sde.T*torch.rand(batch.shape[0]).to(batch.device)

        # TODO forward diffusion
        mean, std = sde.marginal_prob(t, batch)
        xt = mean + std * torch.randn_like(batch, device=batch.device)
        
        # get loss
        if type(loss_fn) == DSMLoss:
            logp_grad = -(xt - mean) / (std**2 + 1e-15)
            _, g = sde.sde_coeff(t, batch)
            diff_sq = g*g
            loss = loss_fn(t, xt.float(), model, logp_grad.to(batch.device), diff_sq.to(batch.device))
        elif isinstance(loss_fn, ISMLoss):
            loss = loss_fn(t, xt.float(), model)
        else:
            print(loss_fn)
            raise Exception("undefined loss")

        # optimize model
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()

        if ema is not None:
            ema.update()

        return loss.item()

    return step_fn


def get_sb_step_fn(model_f, model_b, ema_f, ema_b,
                   opt_f, opt_b, loss_fn, sb, joint=True):
    def step_fn_alter(batch, forward):
        """
        (Optional) Implement the optimization step for alternating
        likelihood training of Schrodinger Bridge
        """
        pass

    def step_fn_joint(batch):
        """
        (Optional) Implement the optimization step for joint likelihood
        training of Schrodinger Bridge
        """
        pass

    if joint:
        return step_fn_joint
    else:
        return step_fn_alter


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def train_diffusion(dataloader, step_fn, N_steps, plot=False, device="cuda"):
    loader = iter(repeater(dataloader))

    log_freq = 200
    loss_history = torch.zeros(N_steps//log_freq)
    with tqdm(range(N_steps), bar_format="{desc}{bar}{r_bar}", mininterval=1) as pbar:
        for i, step in enumerate(pbar):
            batch = next(loader).to(device)
            loss = step_fn(batch)

            if step % log_freq == 0:
                loss_history[i // log_freq] = loss
                pbar.set_description("Loss: {:.3f}".format(loss))

    if plot:
        plt.plot(range(len(loss_history)), loss_history)
        plt.yscale('log')
        plt.show()
