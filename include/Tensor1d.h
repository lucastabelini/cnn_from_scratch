//
// Created by lucas on 07/04/19.
//

#ifndef NEURAL_NET_IN_CPP_TENSOR1D_H
#define NEURAL_NET_IN_CPP_TENSOR1D_H

#include <random>
#include <assert.h>

/*
 * Single dimension tensor class
 */

template<typename T>
class Tensor1d {
private:
    T *data_;
public:
    int length = 0;

    Tensor1d() = default;

    Tensor1d(int length);

    /*
     * Initialize the tensor's values with the given distribution.
     */
    void randn(std::default_random_engine generator, std::normal_distribution<T> distribution, double multiplier);

    /*
     * Access the ith item of the tensor.
     */
    T operator[](int i);

    /*
     * Set the ith item of the tensor.
     */
    void set(int i, T value);

    /*
     * Multiplies tensor by scalar
     */
    Tensor1d<T> operator*(T multiplier);

    /*
     * Subtracts a tensor from another
     */
    Tensor1d<T> operator-=(Tensor1d<T> &difference);

    Tensor1d<T>& operator=(const Tensor1d<T> &other);

    ~Tensor1d();
};

#endif //NEURAL_NET_IN_CPP_TENSOR1D_H
