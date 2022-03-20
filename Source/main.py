import argparse
from trainer import KnownTrainer
from visualize import plot_write_ode
import numpy as np
import os
import torch
import datetime
import torch.multiprocessing as mp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--conc", type=str, required=True,
                        help="File name for concentration profile to fit")
    parser.add_argument('-d', "--weight_decay", type=float, default=1.0E-2,
                        help="Weight decay for ODE training")
    parser.add_argument('-e', "--error_thresh", type=float, default=1.0E-4,
                        help="Error threshold for loss function")
    parser.add_argument('-g', "--guess", type=str, required=False,
                        help="File name for the initial guess")
    parser.add_argument('-i', "--indices", type=str, required=False,
                        help="Filename for rate coefficient indices")
    parser.add_argument('-j', "--jobs", type=int, default=-1,
                        help="Number of parallel jobs")
    parser.add_argument('-l', "--lr", type=float, default=1.0E-2,
                        help="Learning rate for ODE training")
    parser.add_argument('-m', "--max_iter", type=int, default=100,
                        help="Maximum number of iterations")
    parser.add_argument("--name", type=str, default='test',
                        help="The name of this training")
    parser.add_argument('-n', "--nsets", type=int, default=1,
                        help="Number of datasets (equal sized)")
    parser.add_argument('-o', "--output", type=str, default='net.pt',
                        help="File name to write the trained network")
    parser.add_argument('-p', "--sparsity_weight", type=float, default=1.0E-3,
                        help="Weight for sparsity loss")
    parser.add_argument('-q', "--gpu_i", type=int, default=-1,
                        help="Index for GPU to use")
    parser.add_argument('-r', "--order", type=int, default=1,
                        help="Reaction order")
    parser.add_argument('-s', "--scaler", type=str, required=False,
                        help="File name for scaler to be applied for rate coefficients")
    parser.add_argument("--self", action='store_true',
                        help="Enable self reaction in 2nd order")
    parser.add_argument('-sm', "--sum_c", type=str, required=False,
                        help="File name for the combined concentrations")
    parser.add_argument('-v', "--verbose", action='store_true',
                        help="Print messages in training")
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory")
    parser.add_argument('-z', "--zeroth", action='store_true',
                        help="Enable 0th order term")
    args = parser.parse_args()

    world_size = args.jobs
    os.chdir(os.path.expandvars(os.path.expanduser(args.work_dir)))
    raw_data = np.genfromtxt(args.conc)
    print('Raw data from ', args.conc)

    nsets = args.nsets
    nspecies = raw_data.shape[1] - 1
    timestamps = np.split(raw_data[:, 0], nsets)
    concentrations = np.vsplit(raw_data[:, 1:], nsets)

    if args.guess is not None:
        guess = np.genfromtxt(args.guess)
        print('Initial guess from ', args.guess)
    else:
        guess = np.array([])
    if args.indices is not None:
        indices = np.genfromtxt(args.indices)
        print('Use coefficient indices from ', args.indices)
    else:
        indices = np.array([])
    if args.scaler is not None:
        scaler = np.genfromtxt(args.scaler)
        print('Use scaler from ', args.scaler)
    else:
        scaler = np.array([])

    if args.sum_c is not None:
        com_conc = np.genfromtxt(args.sum_c)
        sum_c = np.split(com_conc, nsets)
        print('combined concentrations from ', args.sum_c)
    else:
        sum_c = np.array([])

    trainer = KnownTrainer(indices=indices,
                           scaler=scaler,
                           guess=guess,
                           n_max_reaction_order=args.order,
                           n_species=nspecies,
                           include_zeroth_order=args.zeroth,
                           include_self_reaction=args.self,
                           max_iter=args.max_iter,
                           lr=args.lr,
                           weight_decay=args.weight_decay,
                           err_thresh=args.error_thresh,
                           igpu=args.gpu_i,
                           sparsity_weight=args.sparsity_weight,
                           verbose=args.verbose)

    t1 = datetime.datetime.now()
    if world_size <= 0:
        print("Run sequentially")
        trainer.train(concentrations, timestamps, sum_c)
    else:
        print(f"Run training with {world_size} parallel jobs")
        mp.spawn(trainer.par_train,
                 args=(world_size, concentrations, timestamps, sum_c),
                 nprocs=world_size,
                 join=True)
    t2 = datetime.datetime.now()

    np.set_printoptions(precision=4)
    print(trainer.ode.basis_weights.data.numpy())
    print(f"Total {(t2 - t1).seconds + (t2 - t1).microseconds * 1.0E-6 :.2f}s used in training")

    torch.save(trainer.ode, args.output)
    plot_write_ode(trainer.ode, concentrations, timestamps, args.name, trainer.device)


if __name__ == '__main__':
    main()
