//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_OUTPUTLAYER_H
#define NEURAL_NET_IN_CPP_OUTPUTLAYER_H

#include "Tensor.h"

/*
 * Interface specific for model outputs
 */
class OutputLayer {
public:
    virtual Tensor<double> predict(Tensor<double> input) = 0;

    virtual std::pair<double, Tensor<double>> backprop(std::vector<int> ground_truth) = 0;

    virtual ~OutputLayer() = default;;
};


#endif //NEURAL_NET_IN_CPP_OUTPUTLAYER_H
