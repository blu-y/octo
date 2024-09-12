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

<details>
<summary>상세 내용</summary>

```
WARNING: The following configs were expanded more than once: [tensorrt, cuda_clang, cuda]. For repeatable flags, repeats are counted twice and may lead to unexpected behavior.
INFO: Reading 'startup' options from /home/jet/tensorflow/.bazelrc: --windows_enable_symlinks
INFO: Options provided by the client:
  Inherited 'common' options: --isatty=1 --terminal_columns=278
INFO: Reading rc options for 'build' from /home/jet/tensorflow/.bazelrc:
  Inherited 'common' options: --experimental_repo_remote_exec
INFO: Reading rc options for 'build' from /home/jet/tensorflow/.bazelrc:
  'build' options: --define framework_shared_object=true --define tsl_protobuf_header_only=true --define=use_fast_cpp_protos=true --define=allow_oversize_protos=true --spawn_strategy=standalone -c opt --announce_rc --define=grpc_no_ares=true --noincompatible_remove_legacy_whole_archive --features=-force_no_whole_archive --enable_platform_specific_config --define=with_xla_support=true --config=short_logs --config=v2 --define=no_aws_support=true --define=no_hdfs_support=true --experimental_cc_shared_library --experimental_link_static_libraries_once=false --incompatible_enforce_config_setting_visibility
INFO: Reading rc options for 'build' from /home/jet/tensorflow/.tf_configure.bazelrc:
  'build' options: --action_env PYTHON_BIN_PATH=/usr/bin/python3 --action_env PYTHON_LIB_PATH=/usr/lib/python3.10/dist-packages --python_path=/usr/bin/python3 --config=tensorrt --action_env CUDA_TOOLKIT_PATH=/usr/local/cuda-12.2 --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_87 --action_env LD_LIBRARY_PATH=/usr/lib/llvm-17/lib:/usr/local/cuda-12.2/lib64:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/extras/CUPTI/lib64 --config=cuda_clang --action_env CLANG_CUDA_COMPILER_PATH=/usr/lib/llvm-17/bin/clang --copt=-Wno-gnu-offsetof-extensions --config=cuda_clang
INFO: Found applicable config definition build:short_logs in file /home/jet/tensorflow/.bazelrc: --output_filter=DONT_MATCH_ANYTHING
INFO: Found applicable config definition build:v2 in file /home/jet/tensorflow/.bazelrc: --define=tf_api_version=2 --action_env=TF2_BEHAVIOR=1
INFO: Found applicable config definition build:tensorrt in file /home/jet/tensorflow/.bazelrc: --repo_env TF_NEED_TENSORRT=1
INFO: Found applicable config definition build:cuda_clang in file /home/jet/tensorflow/.bazelrc: --config=cuda --config=tensorrt --action_env=TF_CUDA_CLANG=1 --@local_config_cuda//:cuda_compiler=clang --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=sm_50,sm_60,sm_70,sm_80,compute_90
INFO: Found applicable config definition build:cuda in file /home/jet/tensorflow/.bazelrc: --repo_env TF_NEED_CUDA=1 --crosstool_top=@local_config_cuda//crosstool:toolchain --@local_config_cuda//:enable_cuda
INFO: Found applicable config definition build:tensorrt in file /home/jet/tensorflow/.bazelrc: --repo_env TF_NEED_TENSORRT=1
INFO: Found applicable config definition build:cuda_clang in file /home/jet/tensorflow/.bazelrc: --config=cuda --config=tensorrt --action_env=TF_CUDA_CLANG=1 --@local_config_cuda//:cuda_compiler=clang --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=sm_50,sm_60,sm_70,sm_80,compute_90
INFO: Found applicable config definition build:cuda in file /home/jet/tensorflow/.bazelrc: --repo_env TF_NEED_CUDA=1 --crosstool_top=@local_config_cuda//crosstool:toolchain --@local_config_cuda//:enable_cuda
INFO: Found applicable config definition build:tensorrt in file /home/jet/tensorflow/.bazelrc: --repo_env TF_NEED_TENSORRT=1
INFO: Found applicable config definition build:cuda in file /home/jet/tensorflow/.bazelrc: --repo_env TF_NEED_CUDA=1 --crosstool_top=@local_config_cuda//crosstool:toolchain --@local_config_cuda//:enable_cuda
INFO: Found applicable config definition build:linux in file /home/jet/tensorflow/.bazelrc: --host_copt=-w --copt=-Wno-all --copt=-Wno-extra --copt=-Wno-deprecated --copt=-Wno-deprecated-declarations --copt=-Wno-ignored-attributes --copt=-Wno-array-bounds --copt=-Wunused-result --copt=-Werror=unused-result --copt=-Wswitch --copt=-Werror=switch --copt=-Wno-error=unused-but-set-variable --define=PREFIX=/usr --define=LIBDIR=$(PREFIX)/lib --define=INCLUDEDIR=$(PREFIX)/include --define=PROTOBUF_INCLUDE_PATH=$(PREFIX)/include --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --config=dynamic_kernels --experimental_guard_against_concurrent_changes
INFO: Found applicable config definition build:dynamic_kernels in file /home/jet/tensorflow/.bazelrc: --define=dynamic_loaded_kernels=true --copt=-DAUTOLOAD_DYNAMIC_KERNELS
WARNING: The following configs were expanded more than once: [tensorrt, cuda_clang, cuda]. For repeatable flags, repeats are counted twice and may lead to unexpected behavior.
INFO: Build options --@local_config_cuda//:cuda_compiler, --action_env, and --copt have changed, discarding analysis cache.
INFO: Analyzed target //tensorflow/tools/pip_package:build_pip_package (706 packages loaded, 50676 targets configured).
INFO: Found 1 target...
Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 49591.181s, Critical Path: 2448.98s
INFO: 25422 processes: 443 internal, 24979 local.
INFO: Build completed successfully, 25422 total actions
```
</details>

#### 6. wheel 생성
```
# sudo apt-get install patchelf # patchelf: command not found 오류 발생 시
bazel-bin/tensorflow/tools/pip_package/build_pip_package .
```
위 명령실행 시 현재 디렉토리에 tensorflow-2.16.2-cp310-cp310-linux_aarch64.whl 같은 whl 파일이 생성됨

#### 7. 설치
```
pip install tensorflow-2.16.2-cp310-cp310-linux_aarch64.whl
```
설치 확인 
```
cd # source 파일 내부 실행 시 오류 방지
python -c "import tensorflow as tf; print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"
```
실행 결과
```
2024-09-12 13:29:51.470081: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-09-12 13:29:51.500908: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-09-12 13:29:51.515039: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-09-12 13:29:55.592109: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:984] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-09-12 13:29:55.639399: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:984] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-09-12 13:29:55.639685: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:984] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
Num GPUs Available:  1
```
몇가지 경고가 뜨지만 일단 GPU 인식 확인
