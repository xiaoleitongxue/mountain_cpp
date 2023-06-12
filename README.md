# mountain_cpp

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
./main ../launch.json worker 0
```

###### 4.2. launch master

launch a new terminal

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nx/src/mountain_cpp/3rdparty/libtorch/torch/lib:/home/nx/src/mountain_cpp/3rdparty/darknet

cd launch/bin
# ./main [launch_config_file] [worker type]
./main ../launch.json master
```



