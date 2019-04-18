//
// Created by lucas on 12/04/19.
//

#ifndef NEURAL_NET_IN_CPP_TENSOR_H
#define NEURAL_NET_IN_CPP_TENSOR_H

#include <vector>
#include <random>
#include <assert.h>
#include <memory>
#include <iostream>

/*
 * Tensor class - Supports from 1 to 4 dimensions
 */
template<typename T>
class Tensor {
private:
    T *data_; // TODO: create an storage class to share data between tensors with different views?
    int size_ = -1; // -1 means the size is undefined
public:
    int num_dims = 0;
    int dims[4]{}; // Max tensor dimensions is 4 (could be unlimited, but this makes the implementation simpler)
    Tensor() = default;

    Tensor(int num_dims, int const *dims);

    void view(int new_num_dims, int *new_dims);

    void zero();

    T get(int i); // 1d tensor
    T get(int i, int j); // 2d tensor
    T get(int i, int j, int k); // 3d tensor
    T get(int i, int j, int k, int l); // 4d tensor

    void set(int i, T value);

    void set(int i, int j, T value);

    void set(int i, int j, int k, T value);

    void set(int i, int j, int k, int l, T value);

    void add(int i, T value);

    void add(int i, int j, int k, int l, T value);

    /*
     * Matrix multiplication
     */
    Tensor<T> matmul(Tensor<T> other);

    /*
     * 2D Convolution
     */
    Tensor<T> convolve2d(Tensor<T> kernels, int stride, int padding, Tensor<T> bias);

    /*
     * Returns the transposal
     */
    Tensor<T> matrixTranspose();

    Tensor<T> relu();

    Tensor<T> sigmoid();

    void dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p);

    /*
     * Returns the derivative of the sigmoid function
     */
    Tensor<T> sigmoidPrime();

    Tensor<T> softmax();

    /*
     * Sum every element
     */
    T sum();

    Tensor<T> reluPrime();
//
//    Tensor<T> crossEntropyPrime(Tensor<T> &output, std::vector<int> const &y);
//
//    std::vector<T> sumColumns();

    /*
     * Sum of two 2d tensors
     */
    Tensor<T> operator+(Tensor<T> &other);

    /*
     * Element-wise multiplication of two 2d tensors
     */
    Tensor<T> operator*(Tensor<T> other);

    /*
     * Multiplies every element of the tensor by a value
     */
    Tensor<T> operator*(T multiplier);

    /*
     * Divides every element of the tensor by a value
     */
    Tensor<T> operator/(T divisor);

    /*
     * Subtracts two 2d tensors
     */
    Tensor<T> operator-=(Tensor<T> difference);

    /*
     * Calculates the mean across each row
     */
    Tensor<T> columnWiseSum();

    Tensor<T> channelWiseSum();

    /*
     * Initializes a tensor's values from a distribution
     */
    void randn(std::default_random_engine generator, std::normal_distribution<T> distribution, double multiplier);

    /*
     * Prints the tensor's data
     */
    void print();

    Tensor<T> &operator=(const Tensor<T> &other);

    Tensor(const Tensor<T> &other);

    virtual ~Tensor();

//    Tensor<T>(const Tensor<T> &other);

//    ~Tensor();

};


#endif //NEURAL_NET_IN_CPP_TENSOR_H
