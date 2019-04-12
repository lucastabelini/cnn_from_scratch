//
// Created by lucas on 11/04/19.
//

#include "../include/ReLU.h"

ReLU::ReLU() = default;

Tensor2d<double> &ReLU::forward(Tensor2d<double> &input) {
    input_ = input;
    product_ = input.relu();

    return product_;
}

Tensor2d<double> ReLU::backprop(Tensor2d<double> chainGradient, double learning_rate) {
    return chainGradient * input_.reluPrime();
}

void ReLU::load(FILE *file_model) {

}

void ReLU::save(FILE *file_model) {

}
