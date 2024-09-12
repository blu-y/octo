### Jetson Orin Nano에서 TF 2.16 빌드

#### 0. Env
Jetson Orin Nano  
Jetpack 6.0  
Ubuntu 22.04  
python3.10  

#### 1. llvm-17 설치 (clang-17)
apt로 설치가 안되므로 제공하는 sh 파일 사용  
제공된 CURRENT_LLVM_STABLE로 자동 설치되는 것을 확인  
17을 CURRENT_LLVM_STABLE로 지정  
```
wget https://apt.llvm.org/llvm.sh
vim ~/llvm.sh +%s/CURRENT_LLVM_STABLE=18/CURRENT_LLVM_STABLE=17 +wq
sudo bash ./llvm.sh
```

#### 2. PATH, LD_LIBRARY_PATH 지정
~/.bashrc에 추가 후 source
```
echo "export PATH=/usr/local/cuda-12.2/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PATH=/home/jet/.local/bin:$PATH" >> ~/.bashrc
echo "export PATH=/usr/local/bin:$PATH" >> ~/.bashrc
echo "export PATH=/usr/lib/llvm-17/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64" >> ~/.bashrc
echo "export TF_PYTHON_VERSION=3.10" >> ~/.bashrc
source ~/.bashrc
```
clang 확인
```
clang --version
```
```
Ubuntu clang version 17.0.6 (++20231209124227+6009708b4367-1~exp1~20231209124336.77)
Target: aarch64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/lib/llvm-17/bin
```

#### 3. TF는 r2.16 branch 사용
```
# git clone https://github.com/tensorflow/tensorflow.git
git stash
git checkout r2.16
```

#### 4. configure 파일
```
./configure
```
configure 내용 (빈칸은 default)

a. Please specify the location of python. [Default is /usr/bin/python3]:  
b. Please input the desired Python library path to use.  Default is [/home/jet/dmap_ws/build/dmap]: **/usr/lib/python3.10/dist-packages**  
c. Do you wish to build TensorFlow with ROCm support? [y/N]:  
d. Do you wish to build TensorFlow with CUDA support? [y/N]: **y**    
e. Do you wish to build TensorFlow with TensorRT support? [y/N]: **y**  
        - Found CUDA 12.2, cuDNN 8, TensorRT 8.6.2 확인  
f. CUDA compute capabilities [Default is: 3.5,7.0]: **compute_87**  
g. Do you want to use clang as CUDA compiler? [Y/n]:  
h. Please specify clang path that to be used as host compiler. [Default is /usr/lib/llvm-17/bin/clang]:  
        - Default가 /usr/lib/llvm-17/bin/clang인지 확인  
i. Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]:  
j. Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:  


<details>
<summary>상세 내용</summary>

```
jet@ubuntu:~/tensorflow$ ./configure 
You have bazel 6.5.0 installed.
Please specify the location of python. [Default is /usr/bin/python3]: 


Found possible Python library paths:
  /home/jet/dmap_ws/build/dmap
  /home/jet/dmap_ws/install/dmap/lib/python3.10/site-packages
  /home/jet/dmap_ws/install/dmap_msgs/local/lib/python3.10/dist-packages
  /home/jet/moveit_pg/install/moveit_task_constructor_core/local/lib/python3.10/dist-packages
  /home/jet/moveit_pg/install/moveit_task_constructor_msgs/local/lib/python3.10/dist-packages
  /home/jet/ws_moveit2/install/launch_param_builder/lib/python3.10/site-packages
  /home/jet/ws_moveit2/install/moveit_configs_utils/lib/python3.10/site-packages
  /home/jet/ws_moveit2/install/moveit_task_constructor_core/local/lib/python3.10/dist-packages
  /home/jet/ws_moveit2/install/moveit_task_constructor_msgs/local/lib/python3.10/dist-packages
  /home/jet/ws_moveit2/install/srdfdom/local/lib/python3.10/dist-packages
  /opt/ros/humble/lib/python3.10/site-packages
  /opt/ros/humble/local/lib/python3.10/dist-packages
  /usr/lib/python3.10/dist-packages
  /usr/lib/python3/dist-packages
  /usr/local/lib/python3.10/dist-packages
Please input the desired Python library path to use.  Default is [/home/jet/dmap_ws/build/dmap]
/usr/lib/python3.10/dist-packages
Do you wish to build TensorFlow with ROCm support? [y/N]: 
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: y
TensorRT support will be enabled for TensorFlow.

Found CUDA 12.2 in:
    /usr/local/cuda-12.2/targets/aarch64-linux/lib
    /usr/local/cuda-12.2/targets/aarch64-linux/include
Found cuDNN 8 in:
    /usr/lib/aarch64-linux-gnu
    /usr/include
Found TensorRT 8.6.2 in:
    /usr/lib/aarch64-linux-gnu
    /usr/include/aarch64-linux-gnu


Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: compute_87


Do you want to use clang as CUDA compiler? [Y/n]: 
Clang will be used as CUDA compiler.

Please specify clang path that to be used as host compiler. [Default is /usr/lib/llvm-17/bin/clang]: 


You have Clang 17.0.6 installed.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=mkl_aarch64    # Build with oneDNN and Compute Library for the Arm Architecture (ACL).
        --config=monolithic     # Config for mostly static monolithic build.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
        --config=v1             # Build with TensorFlow 1 API instead of TF 2 API.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=nogcp          # Disable GCP support.
        --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```
</details>

#### 5. 빌드(14시간 소요)
```
bazel build //tensorflow/tools/pip_package:build_pip_package --repo_env=WHEEL_NAME=tensorflow --config=cuda --verbose_failures --copt=-Wno-unused-command-line-argument
```
