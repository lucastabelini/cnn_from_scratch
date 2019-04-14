//
// Created by lucas on 10/04/19.
//

#include <random>
#include "../include/FullyConnected.h"
#include "../include/Tensor.h"

FullyConnected::FullyConnected(int input_size, int output_size, int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    int weights_dims[] = {input_size, output_size};
    weights = Tensor<double>(2, weights_dims);
    weights.randn(generator, distribution, sqrt(2.0 / input_size));
    int bias_dims[] = {output_size};
    bias = Tensor<double>(1, bias_dims);
    bias.randn(generator, distribution, 0);
}


Tensor<double> &FullyConnected::forward(Tensor<double> &input) {
    if (input.num_dims != 2) {
        // flatten tensor
        int flatten_size = 1;
        for (int i = 1; i < input.num_dims; ++i) {
            flatten_size *= input.dims[i];
        }
        int dims[] = {input.dims[0], flatten_size};
        input.view(2, dims);
    }
    input_ = input;
    product_ = input.matmul(weights) + bias;

    return product_;
}

Tensor<double> FullyConnected::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> weightGradient = input_.matrixTranspose().matmul(chainGradient);
    Tensor<double> biasGradient = chainGradient.columnWiseSum();
    chainGradient = chainGradient.matmul(weights.matrixTranspose());
    weights -= weightGradient * learning_rate;
    bias -= biasGradient * learning_rate;
    return chainGradient;
}

void FullyConnected::load(FILE *file_model) {
    double value;
    for (int i = 0; i < weights.dims[0]; ++i) {
        for (int j = 0; j < weights.dims[1]; ++j) {
            int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
            if (read != 1) throw std::runtime_error("Invalid model file");
            weights.set(i, j, value);
        }
    }

    for (int i = 0; i < bias.dims[0]; ++i) {
        int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
        if (read != 1) throw std::runtime_error("Invalid model file");
        bias.set(i, value);
    }
}

void FullyConnected::save(FILE *file_model) {
    for (int i = 0; i < weights.dims[0]; ++i) {
        for (int j = 0; j < weights.dims[1]; ++j) {
            fprintf(file_model, "%.18lf ", weights.get(i, j));
        }
    }

    for (int i = 0; i < bias.dims[0]; ++i) {
        fprintf(file_model, "%.18lf ", bias.get(i));
    }
}

// FullyConnected::~FullyConnected() {
// }