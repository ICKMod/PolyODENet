import os
import torch
import torch.distributed as dist
import numpy as np
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdiffeq import odeint_adjoint as odeint

from polyode import PolynomialODE


def dist_setup(rank, world_size):
    #  initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def dist_cleanup():
    dist.destroy_process_group()


class KnownTrainer:

    def __init__(self, indices, scaler, guess,
                 n_max_reaction_order=2, n_species=3, include_zeroth_order=False,
                 include_self_reaction=False, err_thresh=1.0E-4, max_iter=1000,
                 lr=1.0E-2, weight_decay=1.0E-2, verbose=True, igpu=-1,
                 sparsity_weight=0.0):
        super(KnownTrainer, self).__init__()
        self.ode = PolynomialODE(indices, scaler, guess,
                                 n_max_reaction_order, n_species,
                                 include_zeroth_order, include_self_reaction)
        if igpu >= 0:
            self.device = torch.device(f'cuda:{igpu}')
        else:
            self.device = torch.device('cpu')
        self.ode.to(self.device)
        self.err_thresh = err_thresh
        self.max_iter = max_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.sparsity_weight = sparsity_weight

    def train(self, concentrations, timestamps, sum_c):
        self.par_train(-1, -1, concentrations, timestamps, sum_c)

    def par_train(self, rank, world_size, concentrations, timestamps, sum_c):
        if world_size > 1:
            print(f"Enter training on rank {rank}")
            dist_setup(rank, world_size)
            worker_ode = DDP(self.ode)
        else:
            worker_ode = self.ode

        tensor_concentrations = torch.tensor(np.array(concentrations), dtype=torch.float64, requires_grad=False)
        sum_c = torch.tensor(sum_c, dtype=torch.float64, requires_grad=False)
        tensor_timestamps = torch.tensor(np.array(timestamps), dtype=torch.float64, requires_grad=False)
        rate_coeff_params = list(worker_ode.parameters())[0]

        optimizer = optim.AdamW(worker_ode.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50)

        tensor_concentrations = tensor_concentrations.to(self.device)
        sum_c = sum_c.to(self.device)
        tensor_timestamps = tensor_timestamps.to(self.device)

        def sparse_loss():
            bw_abs = rate_coeff_params.abs()
            # return torch.mean(torch.log(bw_abs / bw_abs.sum(dim=0) + 1.0).sum(dim=0))
            return bw_abs.sum()

        def conserve_loss(t, predicted_c):
            if len(sum_c) == 0:
                c_loss = 0
            else:
                sum_cr = torch.reshape(sum_c[t], (sum_c[t].size()[0], 1))
                sum_c_ls = torch.cat((predicted_c[:, -2:], -sum_cr), 1)
                c_loss = torch.mean(torch.square(sum_c_ls.sum()))
            return c_loss

        def loss_func():
            # # The loss function: 1) only compare data with known values, i.e. not -1;
            # #                    2) concentrations cannot be negative;
            #                      3) conservation or summation constraints.
            ls = 0.0

            for t in range(len(timestamps)):
                if world_size > 1 and t % world_size != rank:
                    continue
                try:
                    tensor_c0 = tensor_concentrations[t][0, :]
                    target = tensor_concentrations[t]
                    predicted_c = odeint(worker_ode, tensor_c0, tensor_timestamps[t])
                    ls = ls + torch.mean(torch.square((predicted_c - target)[target > -0.1]))
                    ls = ls + predicted_c[predicted_c < 0.0].abs().sum()
                    #  ls = ls + conserve_loss(t, predicted_c)
                except AssertionError as ex:
                    if "underflow" in ex.args[0]:
                        print("loss underflow, use sparse loss ")
                        ls = ls + sparse_loss()
            return ls

        if world_size > 1:
            dist.barrier()

        for itr in range(1, self.max_iter + 1):
            if itr == 10000:
                print(rate_coeff_params.data.numpy())

            scheduler_step = True
            optimizer.zero_grad()
            loss = loss_func()
            if loss == sparse_loss():
                scheduler_step = False
            if world_size > 1:
                dist.all_reduce(loss)
                dist.barrier()
            if self.verbose or itr % 10 == 0:
                if rank == 0 or rank == -1:
                    print('{}Iter {:04d} | Total Loss {:.6f} | Sparse Loss {:.6f}'
                          .format("" if world_size < 1 else f"\nRank {rank}, ",
                                  itr, loss.item(), sparse_loss()))
            if loss.item() < self.err_thresh:
                print("Good Enough. Stop")
                break

            loss = loss + sparse_loss() * self.sparsity_weight
            if world_size > 1:
                dist.barrier()
            loss.backward()
            optimizer.step()
            if world_size > 1:
                dist.barrier()

            if scheduler_step:
                scheduler.step(loss)
            if optimizer.param_groups[0]['lr'] < 1.0E-8:
                print("LR too small. Stop")
                break

            #  Make sure there is no negative cross effect.
            clamp_idx = (self.ode.exponent_indices == 0) & (rate_coeff_params.detach().numpy() < 0)
            clamp_idx = torch.tensor(clamp_idx)
            rate_coeff_params.data[clamp_idx] = 0

            #  Assume no auto-catalytic reactions
            clamp_idx = (self.ode.exponent_indices != 0) & (rate_coeff_params.detach().numpy() > 0)
            clamp_idx = torch.tensor(clamp_idx)
            rate_coeff_params.data[clamp_idx] = 0

        # print(rate_coeff_params.grad.numpy())
        if world_size > 1:
            dist_cleanup()
