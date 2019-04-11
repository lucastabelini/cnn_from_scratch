//
// Created by lucas on 10/04/19.
//

#include <random>
#include "../include/FullyConnected.h"

FullyConnected::FullyConnected(int input_size, int output_size) {
    std::default_random_engine generator(0);
    std::normal_distribution<double> distribution(0.0, 1.0);

    weights = Tensor2d<double>(output_size, input_size);
    weights.randn(generator, distribution, sqrt(2.0 / input_size));

    bias = Tensor1d<double>(output_size);
    bias.randn(generator, distribution, 0);
}


Tensor2d<double> &FullyConnected::forward(Tensor2d<double> &input) {
    input_ = input;
    product_ = (weights.matmul(input)) + bias;

    return product_;
}

Tensor2d<double> FullyConnected::backprop(Tensor2d<double> chainGradient, double learning_rate) {
    Tensor2d<double> weightGradient = chainGradient.matmul(input_.transpose());
    Tensor1d<double> biasGradient = chainGradient.rowWiseSum() * learning_rate;
    chainGradient = weights.transpose().matmul(chainGradient);
    weights -= weightGradient * learning_rate;
    bias -= biasGradient;
    return chainGradient;
}

void FullyConnected::load(FILE *file_model) {
    double value;
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
            if (read != 1) throw "Invalid model file";
            weights.set(i, j, value);
        }
    }

    for (int i = 0; i < bias.length; ++i) {
        int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
        if (read != 1) throw "Invalid model file";
        bias.set(i, value);
    }
}

void FullyConnected::save(FILE *file_model) {
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            fprintf(file_model, "%lf", weights.get(i, j));
        }
    }

    for (int i = 0; i < bias.length; ++i) {
        fprintf(file_model, "%lf", bias[i]);
    }
}

// FullyConnected::~FullyConnected() {
// }