```
sudo apt install openexr
pip install -e .
pip install OpenEXR==1.3.2
pip install tensorflow_graphics
git clone https://github.com/kvablack/dlimp.git
cd dlimp
vim ./setup.py +%s/tensorflow==2.15.0/tensorflow\>=2.15.0/ +wq!
pip install .
cd
pip install -r requirements.txt
# pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
wget https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.20+cuda12.cudnn89-cp310-cp310-manylinux2014_aarch64.whl
pip install --upgrade ./jaxlib-0.4.20+cuda12.cudnn89-cp310-cp310-manylinux2014_aarch64.whl --force-reinstall --no-deps
```
```
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export XLA_PYTHON_CLIENT_ALLOCATOR=default
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export JAX_TRACEBACK_FILTERING=off
export JAX_PLATFORMS=cpu
export JAX_DEBUG_NANS=True
export JAX_TRACEBACK_FILTERING=off

```
```
ls /usr/include/cudnn*
ls /usr/lib/aarch64-linux-gnu/libcudnn*
sudo cp /usr/include/cudnn*.h /usr/local/cuda/include
sudo cp -P /usr/lib/aarch64-linux-gnu/libcudnn* /usr/local/cuda/lib64  
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

