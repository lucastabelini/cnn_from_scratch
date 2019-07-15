//
// Created by lucas on 11/04/19.
//

#include "../include/Dropout.h"
#include "../include/Tensor.h"

Dropout::Dropout(double p, int seed) {
    p_ = p;
    seed_ = seed;
}

Tensor<double> &Dropout::forward(Tensor<double> &input) {
//    if (isEval) {
//        return input;
//    }

    dropout_ = Tensor<double>(input.num_dims, input.dims);
    std::default_random_engine generator(seed_);
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
