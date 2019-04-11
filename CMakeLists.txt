cmake_minimum_required(VERSION 3.10)
project(neural_net_in_cpp)

set(CMAKE_CXX_STANDARD 11)

add_executable(neural_net_in_cpp src/main.cpp src/NetworkModel.cpp src/MNISTDataLoader.cpp src/Tensor2d.cpp src/Tensor1d.cpp src/FullyConnected.cpp src/Sigmoid.cpp src/SoftmaxClassifier.cpp src/Dropout.cpp src/Dropout.h)