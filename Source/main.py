import argparse
from Source.trainer import KnownTrainer
from Source.visualize import plot_write_ode
import numpy as np
import os
import sys
import torch
import datetime
import torch.multiprocessing as mp


def main():
    '''
    Main and Do_it construct.

    Introducing this two level approach allows the code to be run in two
    different ways:
    - From the command line when the main function will pass the system
      command line arguments.
    - From inside another Python script when the do_it function can have an
      artificial set of command line arguments passed to it.
    The latter approach is a way to run test cases with, e.g. pytest.
    '''
    do_it(sys.argv[1:])

def do_it(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--conserve", type=int, default=0,
                        help="Number of conservation constraints")
    parser.add_argument('-d', "--weight_decay", type=float, default=1.0E-2,
                        help="Weight decay for ODE training")
    parser.add_argument('-e', "--error_thresh", type=float, default=5.0E-4,
                        help="Error threshold for loss function")
    parser.add_argument('-f', "--file", type=str, required=True,
                        help="File name for input concentration profiles")
    parser.add_argument('-g', "--guess", action='store_true',
                        help="Use initial guesses from .guess file")
    parser.add_argument('-i', "--indices", action='store_true',
                        help="Use rate coefficient indices from .ind file")
    parser.add_argument('-j', "--jobs", type=int, default=-1,
                        help="Number of parallel jobs")
    parser.add_argument('-l', "--lr", type=float, default=1.0E-2,
                        help="Learning rate for ODE training")
    parser.add_argument('-m', "--max_iter", type=int, default=100,
                        help="Maximum number of iterations")
    parser.add_argument('-N', "--name", type=str, required=True,
                        help="The name of this training")
    parser.add_argument('-n', "--n_sets", type=int, default=1,
                        help="Number of datasets (equal sized)")
    parser.add_argument('-p', "--sparsity_weight", type=float, default=0.0,
                        help="Weight for sparsity loss")
    parser.add_argument('-q', "--gpu_i", type=int, default=-1,
                        help="Index for GPU to use")
    parser.add_argument('-r', "--order", type=int, default=1,
                        help="Reaction order")
    parser.add_argument('-s', "--scaler", action='store_true',
                        help="Use rate coefficients scalers from .scale file")
    parser.add_argument("--self", action='store_true',
                        help="Add 2nd order self reaction terms")
    parser.add_argument('-v', "--verbose", action='store_true',
                        help="Print messages in training")
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory")
    parser.add_argument("--zeroth", action='store_true',
                        help="Add 0th order terms")
    args = parser.parse_args(arguments)

    world_size = args.jobs
    os.chdir(os.path.expandvars(os.path.expanduser(args.work_dir)))
    raw_data = np.genfromtxt(args.file)
    print('Target data from', args.file)

    n_sets = args.n_sets
    n_data = raw_data.shape[0]
    if n_data % n_sets != 0:
        print("Number of data points not divisible by n_sets.")
        return
    else:
        print(n_sets, 'data sets, each of', int(n_data/n_sets))
    n_species = raw_data.shape[1] - 1
    timestamps = np.split(raw_data[:, 0], n_sets)
    concentrations = np.vsplit(raw_data[:, 1:], n_sets)

    if args.guess:
        guess = np.genfromtxt(args.name+'.guess')
        print('Use initial guesses from ', args.name+'.guess')
    else:
        guess = np.array([])
    if args.indices:
        indices = np.genfromtxt(args.name+'.ind')
        print('Use coefficient indices from ', args.name+'.ind')
    else:
        indices = np.array([])
    if args.scaler:
        scaler = np.genfromtxt(args.name+'.scale')
        print('Use scaler from ', args.name+'.scale')
    else:
        scaler = np.array([])

    trainer = KnownTrainer(indices=indices,
                           scaler=scaler,
                           guess=guess,
                           conserve=args.conserve,
                           n_max_reaction_order=args.order,
                           n_species=n_species,
                           n_data=n_data,
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
        trainer.train(concentrations, timestamps)
    else:
        print(f"Run training with {world_size} parallel jobs")
        mp.spawn(trainer.par_train,
                 args=(world_size, concentrations, timestamps),
                 nprocs=world_size,
                 join=True)
    t2 = datetime.datetime.now()
    print(f"Total {(t2 - t1).seconds + (t2 - t1).microseconds * 1.0E-6 :.2f}s used in training")

    np.set_printoptions(precision=4)
    print(trainer.ode.basis_weights.data.numpy())
    torch.save(trainer.ode, args.name+'.pt')
    plot_write_ode(trainer.ode, concentrations, timestamps, args.name, trainer.device)
    return trainer.ode.basis_weights.data.numpy()


if __name__ == '__main__':
    main()
