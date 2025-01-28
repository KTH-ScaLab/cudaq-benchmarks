#!/usr/bin/env python3
import time
import argparse
import sys

import cudaq
import numpy as np

import nvtx

# import the circuits from circuits.py
from circuits import QAOA, QuantumVolume, QFT, CounterfeitCoin, GHZ

CIRCUIT_CLASSES = [GHZ, QAOA, QuantumVolume, QFT, CounterfeitCoin]

def eprint(*args, **kwargs):
    """
    Print to stderr.
    """
    print(*args, **kwargs, file=sys.stderr)


def print_result(result, num_qubits):
    """
    Print a histogram of the result
    """
    if num_qubits > 10:
        eprint('WARNING: cannot print results, num_qubits > 10')
        return

    d = {i: 0 for i in range(2**num_qubits)}
    d.update({int(k, 2): v for k, v in result.items()})

    print('\n'.join(f'{k} {v}' for k, v in d.items()))

def run_one_experiment(num_qubits, num_shots, circuit_class, print_histo=False):
  
    circuit = circuit_class(num_qubits)
    # --- instantiate circuits ---
    # circuit = QuantumVolume(num_qubits)
    # circuit = QAOA(num_qubits, np.pi/3, np.pi/6)
    # circuit = CounterfeitCoin(num_qubits)
    # circuit = QFT(num_qubits)
    #circuit = GHZ(num_qubits)

    # --- get kernel ---
    k, params = circuit.kernel, circuit.kernel_params

    ts = []
    # --- main experiment loop ---
    for i in range(args.warmup + args.iter):
        r = nvtx.start_range(f'it{i}')
        t0 = time.time()
        result = cudaq.sample(k, *params, 
                              shots_count=num_shots)

        if print_histo:
            print(f'Most likely outcome: {result.most_probable()}')
            print_result(result, num_qubits)

        t1 = time.time()
        nvtx.end_range(r)
        
        t = t1 - t0
        if i >= args.warmup: ts.append(t)
        eprint(t) # for debug

    avg = np.mean(ts)
    std = np.std(ts)
    print(f'{num_qubits} {avg} {std}')

if __name__ == '__main__':
    # we need to set it manually, since CUDA-Q disables it by default
    sys.tracebacklimit = 1000

    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-qubits',   
                   type=int, required=True)
    p.add_argument('-N', '--num-qubits-max',
                   type=int, required=False,
                   help='if set, launches several repeat with `num_qubits` from `num_qubits` to `num_qubits_max`')
    p.add_argument('-s', '--num-shots',   
                   type=int, default=1024)
    p.add_argument('-w', '--warmup',
                   type=int, default=1,
                   help='numer of warmup iterations (default: 1)')
    p.add_argument('-i', '--iter',
                   type=int, default=10,
                   help='number of iterations (default: 10)')
    # NOTE: `target` is parsed automatically by CUDA-Q, we do not need to handle it.
    p.add_argument('--target',
                   type=str, required=False,
                   help='target for CUDA-Q execution')

    circuit_names = [str(c.__name__) for c in CIRCUIT_CLASSES]
    p.add_argument('--circuit',
                   type=str, default=circuit_names[0], choices=circuit_names,
                   help=f'Circuit name to benchmark ({circuit_names}), default: {circuit_names[0]}.')
    p.add_argument('--seed',
                   type=int, required=False,
                   help='seed for random number generation (both NumPy and CUDA-Q)')
    p.add_argument('--histo',
                   action='store_true',
                   help='outputs a histogram of measurements for all quantum states')

    args = p.parse_args()
    num_qubits_min = args.num_qubits
    num_qubits_max = (args.num_qubits_max or num_qubits_min) + 1
    num_shots  = args.num_shots
    circuit_class = CIRCUIT_CLASSES[circuit_names.index(args.circuit)]

    if args.seed is not None:
        np.random.seed(args.seed)
        cudaq.set_random_seed(args.seed)

    eprint(args)

    for num_qubits in range(num_qubits_min, num_qubits_max):
        run_one_experiment(num_qubits, num_shots, circuit_class, args.histo)
    
