//
// Created by lucas on 11/04/19.
//

#include "../include/Dropout.h"
#include "../include/Tensor.h"

Dropout::Dropout(double p) {
    p_ = p;
}

Tensor<double> &Dropout::forward(Tensor<double> &input, int seed) {
    dropout_ = Tensor<double>(input.num_dims, input.dims);
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<> distribution(0., 1.);

    dropout_.dropout(generator, distribution, p_);
    product_ = input * dropout_;
    return product_;
}

Tensor<double> Dropout::backprop(Tensor<double> chainGradient, double learning_rate) {
    return chainGradient * dropout_;
}

void Dropout::load(FILE *file_model) {

}

void Dropout::save(FILE *file_model) {

}
