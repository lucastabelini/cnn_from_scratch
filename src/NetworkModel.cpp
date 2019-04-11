//
// Created by lucas on 10/04/19.
//

#include "../include/NetworkModel.h"

using namespace std;

NetworkModel::NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, double learning_rate) {
    modules_ = modules;
    learning_rate_ = learning_rate;
    output_layer_ = output_layer;
}

double NetworkModel::trainStep(Tensor2d<double> &x, vector<int> y) {
    // Forward
    Tensor2d<double> output = forward(x);

    //Backprop
    pair<double, Tensor2d<double>> loss_and_cost_gradient = output_layer_->backprop(y);
    Tensor2d<double> chain_gradient = loss_and_cost_gradient.second;
    for (int i = (int) modules_.size() - 1; i >= 0; --i) {
        chain_gradient = modules_[i]->backprop(chain_gradient, learning_rate_);
    }

    // Return loss
    return loss_and_cost_gradient.first;
}

Tensor2d<double> NetworkModel::forward(Tensor2d<double> &x) {
    for (auto &module : modules_) {
        x = module->forward(x);
    }
    return output_layer_->predict(x);
}

std::vector<int> NetworkModel::predict(Tensor2d<double> &x) {
    Tensor2d<double> output = forward(x);

    std::vector<int> predictions;
    for (int j = 0; j < output.cols; ++j) {
        int argmax = -1;
        double max = -1;
        for (int i = 0; i < output.rows; ++i) {
            if (output.get(i, j) > max) {
                max = output.get(i, j);
                argmax = i;
            }
        }
        predictions.push_back(argmax);
    }

    return predictions;
}

void NetworkModel::load(std::string path) {
    FILE *model_file = fopen(path.c_str(), "r");
    for (auto &module : modules_) {
        module->load(model_file);
    }
}

void NetworkModel::save(std::string path) {
    FILE *model_file = fopen(path.c_str(), "w");
    for (int i = 0; i < modules_.size(); ++i) {
        modules_[i]->save(model_file);
    }
}

NetworkModel::~NetworkModel() {
    for (int i = 0; i < modules_.size(); ++i) {
        delete modules_[i];
    }
    delete output_layer_;
}
