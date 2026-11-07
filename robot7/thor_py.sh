#!/bin/bash
# Python launcher for the Jetson Thor (NVIDIA's ARM torch needs the pip-installed cuDNN/cuDSS/NVPL libraries on
# the loader path; same as ~/so101_vla/run_pi05.sh but without changing the working directory, so scripts can be
# run from anywhere). Usage: SIMLAB=~/simlab robot7/thor_py.sh so101flow.py check --live
SO101_VLA=${SO101_VLA:-$HOME/so101_vla}
SP=$SO101_VLA/venv_pi05/lib/python3.12/site-packages
LIBS=$(find "$SP/nvidia" "$SP/nvpl" -name "*.so*" 2>/dev/null | xargs -n1 dirname | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1          # the Jetson torch has no Triton
export SIMLAB=${SIMLAB:-$HOME/simlab}
exec "$SO101_VLA/venv_pi05/bin/python" "$@"
