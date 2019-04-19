//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_FULLYCONNECTED_H
#define NEURAL_NET_IN_CPP_FULLYCONNECTED_H

#include "Module.h"
#include "Tensor.h"

/*
 * Fully Connected layer
 * Output: Mx + b
 */
class FullyConnected : public Module {
private:
    Tensor<double> weights;
    Tensor<double> bias;
    Tensor<double> input_;
    Tensor<double> product_;
    int input_dims[4];
    int input_num_dims;
public:
    FullyConnected(int input_size, int output_size, int seed = 0);

    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_FULLYCONNECTED_H
