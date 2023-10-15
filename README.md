# How to build
## Prepare

### clone repository
```
git clone --recursive https://github.com/xiaoleitongxue/mountain_cpp.git
```
### Download libtorch and unzip
```
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
## build
### 1. build 3rd party

###### 1.1 build darknet

```shell
cd 3rdparty/darknet
mkdir build_release
cd build_release
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --target install --parallel 8
```

###### 1.2. build json

```shell
cd 3rdparty/json
mkdir build
cd build
cmake ..
cmake --build .
```

###### 1.3. install libtorch

```shell
cd 3rdparty
mkdir libtorch
unzip torch.whl
```

### 2. build mountain_cpp

```shell
cd mountain_cpp
mkdir build
cd build
cmake ..
cmake --build . --target install --parallel 8
```

### 3. Before Run

please modified launch/launch.json for ip_address and port, and other configures.

### 4. How to Run

###### 4.1. launch worker

launch a new terminal

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nx/src/mountain_cpp/3rdparty/libtorch/torch/lib:/home/nx/src/mountain_cpp/3rdparty/darknet

cd launch/bin
# ./main [launch_config_file] [worker type] [worker id]
./main ../launch-yolov2-tiny.json worker 0
```

###### 4.2. launch master

launch a new terminal

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nx/src/mountain_cpp/3rdparty/libtorch/torch/lib:/home/nx/src/mountain_cpp/3rdparty/darknet

cd launch/bin
# ./main [launch_config_file] [worker type]
./main ../launch-yolov2-tiny.json master
```



