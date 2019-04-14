//
// Created by lucas on 11/04/19.
//

#ifndef NEURAL_NET_IN_CPP_DROPOUT_H
#define NEURAL_NET_IN_CPP_DROPOUT_H


#include "Module.h"

class Dropout : public Module {
private:
    double p_;
    Tensor<double> product_;
    Tensor<double> dropout_;
public:
    explicit Dropout(double p = 0.5);

    Tensor<double> &forward(Tensor<double> &input, int seed) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_DROPOUT_H
