//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_MODULE_H
#define NEURAL_NET_IN_CPP_MODULE_H

#include "Tensor2d.h"

/*
 * Interface to be used as a building block for models
 */
class Module {
public:
    virtual Tensor2d<double> &forward(Tensor2d<double> &input) = 0;

    virtual Tensor2d<double> backprop(Tensor2d<double> chainGradient, double learning_rate) = 0;

    virtual void load(FILE *file_model) = 0;

    virtual void save(FILE *file_model) = 0;

    virtual ~Module() {};
};


#endif //NEURAL_NET_IN_CPP_MODULE_H
