//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_FULLYCONNECTED_H
#define NEURAL_NET_IN_CPP_FULLYCONNECTED_H

#include "Module.h"
#include "Tensor1d.h"
#include "Tensor2d.h"

/*
 * Fully Connected layer
 * Output: Mx + b
 */
class FullyConnected : public Module {
private:
    Tensor2d<double> weights;
    Tensor1d<double> bias;
    Tensor2d<double> input_;
    Tensor2d<double> product_;
public:
    FullyConnected(int input_size, int output_size);

    Tensor2d<double> &forward(Tensor2d<double> &input) override;

    Tensor2d<double> backprop(Tensor2d<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    virtual void save(FILE *file_model);
};


#endif //NEURAL_NET_IN_CPP_FULLYCONNECTED_H
