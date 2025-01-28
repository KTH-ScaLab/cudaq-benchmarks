# Benchmark suite for MPS/Tensor Network simulation
This repository proposes a set of five configurable quantum circuits for evaluation
of Nvidia's CUDA-Q simulator backends, targeting specifically plain Tensor Network
and Matrix Product State (MPS) simulators.

## Usage
To obtain help on the utilization of the benchmark script, use:
```
python benchmark.py --help
```

Depending on the architecture of the executing system - and notably on Arm platform -,
it might be simpler to execute CUDA-Q within a Docker container. Nvidia provides a
[Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum)
which streamlines this process. We propose an experimental script `scripts/run.sh`, which
automatizes creating and running the container, used as a wrapper:
```
scripts/./run.sh python3 benchmark.py --help
```

## Citation
G. Schieffer, S. Markidis, and I. Peng. (2025). Harnessing CUDA-Q's MPS for Tensor Network Simulations of Large-Scale Quantum Circuits. *2025 33rd Euromicro International Conference on Parallel, Distributed and Network-Based Processing*.     

Arxiv pre-print: https://doi.org/10.48550/arXiv.2501.15939
