//
// Created by lucas on 10/04/19.
//

#include "../include/SoftmaxClassifier.h"

Tensor2d<double> SoftmaxClassifier::predict(Tensor2d<double> input) {
    output_ = input.softmax();
    return output_;
}

std::pair<double, Tensor2d<double>> SoftmaxClassifier::backprop(std::vector<int> ground_truth) {
    double loss = crossEntropy(output_, ground_truth);
    Tensor2d<double> gradient = crossEntropyPrime(output_, ground_truth);

    return std::make_pair(loss, gradient);
}


Tensor2d<double> SoftmaxClassifier::crossEntropyPrime(Tensor2d<double> &output, std::vector<int> &y) {
    Tensor2d<double> prime = output;
    for (int i = 0; i < y.size(); ++i) {
        prime.set(y[i], i, prime.get(y[i], i) - 1);
    }

    return prime / output.cols;
}


double SoftmaxClassifier::crossEntropy(Tensor2d<double> &y_hat, std::vector<int> &y) {
    double total = 0;
    for (int i = 0; i < y_hat.cols; ++i) {
        double x = y_hat.get(y[i], i);
        // Sets a minimum value to prevent division by zero (log(0))
        total += -log(x < 0.0000000001 ? 0.0000000001 : x);
    }

    return total / y.size(); // batch-wise mean
}