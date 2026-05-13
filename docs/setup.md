# Setup guide

## Environment
- WSL2 Ubuntu 24.04 LTS
- NVIDIA Quadro RTX 5000 16GB, driver 591.86
- CUDA 12.6 toolkit
- GCC 13.3, CMake 3.28, Python 3.12

## Phase 1 — CUDA toolkit
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Phase 2 — llama.cpp
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --parallel $(nproc)
```

## Phase 3 — Model
```bash
mkdir -p ~/models
pip3 install huggingface-hub --break-system-packages
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id='bartowski/Phi-3-mini-4k-instruct-GGUF',
    filename='Phi-3-mini-4k-instruct-Q4_K_M.gguf',
    cache_dir='/tmp/hf_cache'
)
shutil.copy(path, os.path.expanduser('~/models/phi3-mini-q4.gguf'))
"
```
Verify: `./build/bin/llama-cli -m ~/models/phi3-mini-q4.gguf -ngl 99 -n 50 -p "test"`

## Phase 4 — Python venv
```bash
cd ~/socratic-engine
bash scripts/bootstrap.sh
source python/venv/bin/activate
```

## Phase 5 — Docker
Enable WSL2 integration in Docker Desktop settings.

## Verified working
- faiss 1.13.2
- huggingface-hub 1.14.0  
- networkx 3.6.1
- Python venv isolated at python/venv/ (not system Python)
