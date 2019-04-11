//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_NETWORKMODEL_H
#define NEURAL_NET_IN_CPP_NETWORKMODEL_H

#include <vector>
#include "Module.h"
#include "OutputLayer.h"

/*
 * Train and test a neural network defined by Modules
 */
class NetworkModel {
private:
    std::vector<Module *> modules_;
    OutputLayer *output_layer_;
    double learning_rate_;
public:
    NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, double learning_rate);

    double trainStep(Tensor2d<double> &x, std::vector<int> y);

    Tensor2d<double> forward(Tensor2d<double> &x);

    std::vector<int> predict(Tensor2d<double> &x);

    void load(std::string path);

    void save(std::string path);

    virtual ~NetworkModel();
};


#endif //NEURAL_NET_IN_CPP_NETWORKMODEL_H
