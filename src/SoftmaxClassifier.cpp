//
// Created by lucas on 10/04/19.
//

#include "../include/SoftmaxClassifier.h"
#include "../include/Tensor.h"

Tensor<double> SoftmaxClassifier::predict(Tensor<double> input) {
    output_ = input.softmax();
    return output_;
}

std::pair<double, Tensor<double>> SoftmaxClassifier::backprop(std::vector<int> ground_truth) {
    double loss = crossEntropy(output_, ground_truth);
    Tensor<double> gradient = crossEntropyPrime(output_, ground_truth);

    return std::make_pair(loss, gradient);
}


Tensor<double> SoftmaxClassifier::crossEntropyPrime(Tensor<double> &output, std::vector<int> &y) {
    Tensor<double> prime = output;
    for (int i = 0; i < y.size(); ++i) {
        prime.set(i, y[i], prime.get(i, y[i]) - 1);
    }

    return prime / output.dims[0];
}


double SoftmaxClassifier::crossEntropy(Tensor<double> &y_hat, std::vector<int> &y) {
    double total = 0;
    for (int i = 0; i < y.size(); ++i) {
        double x = y_hat.get(i, y[i]);
        // Sets a minimum value to prevent division by zero (log(0))
        total += -log(x < 0.0000000001 ? 0.0000000001 : x);
    }

    return total / y.size(); // batch-wise mean
}