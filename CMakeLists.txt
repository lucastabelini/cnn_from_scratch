cmake_minimum_required(VERSION 3.5)
project(neural_net_in_cpp)

set(CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_FLAGS "-pg") -- for profiling

add_executable(neural_net_in_cpp src/main.cpp src/NetworkModel.cpp src/MNISTDataLoader.cpp src/Tensor.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/SoftmaxClassifier.cpp src/Dropout.cpp src/ReLU.cpp src/Conv2d.cpp src/MaxPool.cpp src/LRScheduler.cpp src/LinearLRScheduler.cpp)
