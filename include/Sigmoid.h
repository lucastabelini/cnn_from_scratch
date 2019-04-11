//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_SIGMOID_H
#define NEURAL_NET_IN_CPP_SIGMOID_H

#include "Module.h"
#include "Tensor2d.h"

/*
 * Sigmoid activation layer
 * Output: 1.0 / (1.0 + exp(-x))
 */
class Sigmoid : public Module {
private:
    Tensor2d<double> input_;
    Tensor2d<double> product_;
public:
    Sigmoid();

    Tensor2d<double> &forward(Tensor2d<double> &input) override;

    Tensor2d<double> backprop(Tensor2d<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
    // ~Sigmoid();
};

#endif //NEURAL_NET_IN_CPP_SIGMOID_H
