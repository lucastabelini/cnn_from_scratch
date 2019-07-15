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
protected:
    bool isEval = false;
public:
    virtual Tensor<double> &forward(Tensor<double> &input) = 0;

    virtual Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) = 0;

    virtual void load(FILE *file_model) = 0;

    virtual void save(FILE *file_model) = 0;

    void train();

    void eval();

    virtual ~Module() = default;
};

inline void Module::eval() {
    this->isEval = true;
}

inline void Module::train() {
    this->isEval = false;
}


#endif //NEURAL_NET_IN_CPP_MODULE_H
