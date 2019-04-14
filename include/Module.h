//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_MODULE_H
#define NEURAL_NET_IN_CPP_MODULE_H

#include "Tensor.h"

/*
 * Interface to be used as a building block for models
 */
class Module {
public:
    virtual Tensor<double> &forward(Tensor<double> &input) = 0;

    virtual Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) = 0;

    virtual void load(FILE *file_model) = 0;

    virtual void save(FILE *file_model) = 0;

    virtual ~Module() = default;
};


#endif //NEURAL_NET_IN_CPP_MODULE_H
