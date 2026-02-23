# SPS Installation Steps

```bash
# Create conda environment
conda create -n vseek-vllm-1 python=3.11
conda activate vseek-vllm-1

# Install main dependencies
pip install -r requirements_vllm.txt

# Install verl
cd vendor/verl
pip install -e .
cd ../..

# CUDA toolkit
conda install -c conda-forge cudatoolkit-dev -y

# Flash attention
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

# Install vseek
pip install -e .

# setuptools 82+ removed pkg_resources; downgrade required for viclip_text.py
pip install "setuptools<82"
```
