//
// Created by tabelini on 18/04/19.
//

#ifndef NEURAL_NET_IN_CPP_MAXPOOL_H
#define NEURAL_NET_IN_CPP_MAXPOOL_H


#include "Module.h"

class MaxPool : public Module {
private:
    Tensor<double> output_;
    Tensor<double> input_;
    Tensor<int> indexes;
    int stride_, size_;
public:
    explicit MaxPool(int size, int stride);

    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_MAXPOOL_H
