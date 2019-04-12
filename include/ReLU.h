//
// Created by lucas on 11/04/19.
//

#ifndef NEURAL_NET_IN_CPP_RELU_H
#define NEURAL_NET_IN_CPP_RELU_H


#include "Tensor2d.h"
#include "Module.h"

class ReLU : public Module{
private:
    Tensor2d<double> input_;
    Tensor2d<double> product_;
public:
    ReLU();

    Tensor2d<double> &forward(Tensor2d<double> &input) override;

    Tensor2d<double> backprop(Tensor2d<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_RELU_H
