//
// Created by lucas on 11/04/19.
//

#ifndef NEURAL_NET_IN_CPP_DROPOUT_H
#define NEURAL_NET_IN_CPP_DROPOUT_H


#include "Module.h"

class Dropout : public Module {
private:
    double p_;
    Tensor2d<double> product_;
    Tensor2d<double> dropout_;
public:
    explicit Dropout(double p = 0.5);

    Tensor2d<double> &forward(Tensor2d<double> &input) override;

    Tensor2d<double> backprop(Tensor2d<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
    // ~Sigmoid();
};


#endif //NEURAL_NET_IN_CPP_DROPOUT_H
