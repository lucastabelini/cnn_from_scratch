//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H
#define NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H


#include "OutputLayer.h"
#include "Tensor1d.h"

/*
 * Applies softmax and uses cross entropy as loss function
 */
class SoftmaxClassifier : public OutputLayer {
private:
    Tensor2d<double> output_;
public:
    Tensor2d<double> predict(Tensor2d<double> input) override;

    std::pair<double, Tensor2d<double>> backprop(std::vector<int> ground_truth) override;

    Tensor2d<double> crossEntropyPrime(Tensor2d<double> &output, std::vector<int> &y);

    double crossEntropy(Tensor2d<double> &y_hat, std::vector<int> &y);
};


#endif //NEURAL_NET_IN_CPP_SOFTMAXCLASSIFIER_H
