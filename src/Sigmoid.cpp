//
// Created by lucas on 10/04/19.
//

#include "../include/Sigmoid.h"

Sigmoid::Sigmoid() = default;


Tensor2d<double> &Sigmoid::forward(Tensor2d<double> &input) {
    input_ = input;
    product_ = input.sigmoid();

    return product_;
}

Tensor2d<double> Sigmoid::backprop(Tensor2d<double> chainGradient, double learning_rate) {
    return chainGradient * input_.sigmoidPrime();
}

void Sigmoid::load(FILE *file_model) {

}

void Sigmoid::save(FILE *file_model) {

}

// Sigmoid::~Sigmoid() {
// }