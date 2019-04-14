//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_SIGMOID_H
#define NEURAL_NET_IN_CPP_SIGMOID_H

#include "Module.h"
#include "Tensor.h"

/*
 * Sigmoid activation layer
 * Output: 1.0 / (1.0 + exp(-x))
 */
class Sigmoid : public Module {
private:
    Tensor<double> input_;
    Tensor<double> product_;
public:
    Sigmoid();

    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
    // ~Sigmoid();
};

#endif //NEURAL_NET_IN_CPP_SIGMOID_H
