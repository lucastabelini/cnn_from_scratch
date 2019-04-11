//
// Created by lucas on 11/04/19.
//

#include "../include/Dropout.h"

Dropout::Dropout(double p) {
    p_ = p;
}

Tensor2d<double> &Dropout::forward(Tensor2d<double> &input) {
    dropout_ = Tensor2d<double>(input.rows, input.cols);
    std::default_random_engine generator(0);
    std::uniform_real_distribution<> distribution(0., 1.);

    for (int i = 0; i < dropout_.rows; ++i) {
        for (int j = 0; j < dropout_.cols; ++j) {
            dropout_.set(i, j, (distribution(generator) < p_) / p_);
        }
    }
    product_ = input * dropout_;
    return product_;
}

Tensor2d<double> Dropout::backprop(Tensor2d<double> chainGradient, double learning_rate) {
    return chainGradient * dropout_;
}

void Dropout::load(FILE *file_model) {

}

void Dropout::save(FILE *file_model) {

}
