//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H
#define NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H


#include "OutputLayer.h"

/*
 * Applies softmax and uses cross entropy as loss function
 */
class SoftmaxClassifier : public OutputLayer {
private:
    Tensor<double> output_;
public:
    Tensor<double> predict(Tensor<double> input) override;

    std::pair<double, Tensor<double>> backprop(std::vector<int> ground_truth) override;

    Tensor<double> crossEntropyPrime(Tensor<double> &output, std::vector<int> &y);

    double crossEntropy(Tensor<double> &y_hat, std::vector<int> &y);
};


#endif //NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H
