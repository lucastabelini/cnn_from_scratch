//
// Created by lucas on 04/04/19.
//

#ifndef NEURAL_NET_IN_CPP_NETWORK_H
#define NEURAL_NET_IN_CPP_NETWORK_H


#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include "Tensor1d.h"
#include "Tensor2d.h"
#include <random>

/*
 * Neural Network class - Train and test
 */

class Network {
    std::vector<Tensor2d<double> *> weights;
    std::vector<Tensor1d<double> *> biases;
    std::vector<Tensor2d<double> > weight_gradients_;
    std::vector<Tensor1d<double> > bias_gradients_;
    std::vector<Tensor2d<double> > products_;
    std::vector<Tensor2d<double> > activations_;
    unsigned int num_layers_;
    double learning_rate_ = 0.75;

    Tensor2d<double> forward(Tensor2d<double> &x);

    double computeGradients(Tensor2d<double> &x, std::vector<int> const &y);

    /*
     * Updates the weights with the given gradients.
     */
    void updateWeights();

    /*
     * Calculates cross-entropy (or log loss). y_hat (predictions) should have shape (output size, batch_size_),
     * and y (batch_size_) with each y_i having the label's id as value (0-indexed).
     */
    double crossEntropy(Tensor2d<double> &y_hat, std::vector<int> const &y);

    /*
     * Calculates the derivative of a classifier with softmax + cross entropy.
     */
    Tensor2d<double> crossEntropyPrime(Tensor2d<double> &output, std::vector<int> const &y);

public:
    /*
     * layers should be a vector with the number of neurons in each layer. The first value should be the input size,
     * while the last should be the output size. The seed will be used to initialize the network's weights.
     */
    explicit Network(std::vector<unsigned int> &layers, int seed, double learning_rate);

    ~Network();

    /*
     * Save the network's weights to a .txt file.
     */
    void save();

    /*
     * Loads weights from a previous training.
     */
    void load(std::string path);

    /*
     * Predicts output on a data set
     * Output: Vector with the same size as the batch, with values corresponding to the class number.
     */
    std::vector<int> predict(Tensor2d<double> &x);

    /*
     * Trains the network on a batch. x should have shape (input size, batch size) and y (batch size)
     */
    double trainStep(Tensor2d<double> &x, std::vector<int> const &y);

    double
    gradientCheck(Tensor2d<double> &x, std::vector<int> const &y, double h = 1e-7, int max_gradients_check = 800);
};


#endif //NEURAL_NET_IN_CPP_NETWORK_H
