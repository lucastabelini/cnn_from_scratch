//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_NETWORKMODEL_H
#define NEURAL_NET_IN_CPP_NETWORKMODEL_H

#include <vector>
#include "Tensor.h"
#include "Module.h"
#include "OutputLayer.h"
#include "../include/LRScheduler.h"

/*
 * Train and test a neural network defined by Modules
 */
class NetworkModel {
private:
    std::vector<Module *> modules_;
    OutputLayer *output_layer_;
    LRScheduler* lr_scheduler_;
    int iteration = 0;
public:
    NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, LRScheduler* lr_scheduler);

    double trainStep(Tensor<double> &x, std::vector<int> &y);

    Tensor<double> forward(Tensor<double> &x);

    std::vector<int> predict(Tensor<double> &x);

    void load(std::string path);

    void save(std::string path);

    virtual ~NetworkModel();
};


#endif //NEURAL_NET_IN_CPP_NETWORKMODEL_H
