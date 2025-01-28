#!/bin/sh
# execute a script inside the CUDA-Q container

# include Nsight Systems, as found in the host environment
IMAGE=nvcr.io/nvidia/quantum/cuda-quantum:0.8.0
docker run \
    --rm --gpus all \
    -v .:/scripts \
    -v /opt/nvidia/nsight-systems-cli:/opt/nvidia/nsight-systems-cli/:ro \
    --workdir /scripts --entrypoint '' \
    -it $IMAGE \
    /bin/sh -c "exec env MPLCONFIGDIR=/tmp/.mplconfig/ PATH=\$PATH:/opt/nvidia/nsight-systems-cli/2024.2.1/target-linux-sbsa-armv8/ $*"
