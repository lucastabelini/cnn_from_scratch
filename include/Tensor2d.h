//
// Created by lucas on 04/04/19.
//

#ifndef NEURAL_NET_IN_CPP_TENSOR2D_H
#define NEURAL_NET_IN_CPP_TENSOR2D_H

#include <stdlib.h>
#include <ostream>
#include <random>
#include <assert.h>
#include <vector>
#include <iostream>
#include "Tensor1d.h"

/*
 * Two dimensional tensor class
 */

template<typename T>
class Tensor2d {
private:
    T *data_;
public:
    int rows = 0, cols = 0;

    Tensor2d();

    Tensor2d(int rows, int cols);

//    void full(T value);

    /*
     * Sets the value in the ith row of the jth column
     */
    T set(int i, int j, T value);

    /*
     * Gets the value in the ith row of the jth column
     */
    T get(int i, int j);

    /*
     * Matrix multiplication
     */
    Tensor2d<T> matmul(Tensor2d<T> other);

    /*
     * Returns the transposal
     */
    Tensor2d<T> transpose();

    /*
     * Sums a 1-dimensional tensor to a 2-dimensional tensor.
     */
    Tensor2d<T> operator+(Tensor1d<T> &bias);

//    Tensor2d<T> relu();

    Tensor2d<T> sigmoid();

    /*
     * Returns the derivative of the sigmoid function
     */
    Tensor2d<T> sigmoidPrime();

    Tensor2d<T> softmax();

    /*
     * Sum every element
     */
    double sum();

//    Tensor2d<T> reluPrime(Tensor2d<T> &x);
//
//    Tensor2d<T> crossEntropyPrime(Tensor2d<T> &output, std::vector<int> const &y);
//
//    std::vector<T> sumColumns();

    /*
     * Sum of two 2d tensors
     */
    Tensor2d<T> operator+(Tensor2d<T> other);

    /*
     * Element-wise multiplication of two 2d tensors
     */
    Tensor2d<T> operator*(Tensor2d<T> multiplier);

    /*
     * Multiplies every element of the tensor by a value
     */
    Tensor2d<T> operator*(T multiplier);

    /*
     * Divides every element of the tensor by a value
     */
    Tensor2d<T> operator/(T divisor);

    /*
     * Subtracts two 2d tensors
     */
    Tensor2d<T> &operator-=(Tensor2d<T> difference);

    /*
     * Calculates the mean across each row
     */
    Tensor1d<T> rowWiseSum();

    /*
     * Initializes a tensor's values from a distribution
     */
    void randn(std::default_random_engine generator, std::normal_distribution<T> distribution, double multiplier);

    /*
     * Prints the tensor's data
     */
    void print();

    Tensor2d<T> &operator=(const Tensor2d<T> &other);

    Tensor2d<T>(const Tensor2d<T> &other);

    ~Tensor2d();

};


#endif //NEURAL_NET_IN_CPP_TENSOR2D_H
