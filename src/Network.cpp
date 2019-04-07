//
// Created by lucas on 04/04/19.
//

#include "../include/Network.h"
#include "../include/Tensor2d.h"


Network::Network(std::vector<unsigned int> &layers, int seed = 0, double learning_rate = 0.75) {
    this->learning_rate_ = learning_rate;
    this->num_layers_ = (unsigned int) layers.size();
    std::default_random_engine generator(seed);
    weights.resize(num_layers_ - 1);
    biases.resize(num_layers_ - 1);

    // Initialize weights
    for (int i = 0; i < this->num_layers_ - 1; ++i) {
        std::normal_distribution<double> distribution(0.0, sqrt(2.0 / layers[i + 1]));
        weights[i] = new Tensor2d<double>(layers[i + 1], layers[i]);
        weights[i]->randn(generator, distribution);
        biases[i] = new Tensor1d<double>(layers[i + 1]);
        biases[i]->randn(generator, distribution);
    }
}

Network::~Network() {
    for (int i = 0; i < weights.size(); ++i) {
        delete weights[i];
        delete biases[i];
    }
}

void Network::save() {
    FILE *file = fopen("network.txt", "w");
    for (int i = 0; i < num_layers_ - 1; ++i) {
        for (int j = 0; j < weights[i]->rows; ++j) {
            for (int k = 0; k < weights[i]->cols; ++k) {
                fprintf(file, "%.18lf ", weights[i]->get(j, k));
            }
        }
        for (int l = 0; l < biases[i]->length; ++l) {
            fprintf(file, "%.18lf ", (*biases[i])[l]);
        }
    }
    fprintf(file, "\n");
    fclose(file);
}

std::vector<int> Network::predict(Tensor2d<double> &x) {
    // Forward flow of the network
    Tensor2d<double> activation = x;
    for (int i = 0; i < (num_layers_ - 1); ++i) {
        Tensor2d<double> product = (weights[i]->matmul(activation)) + *biases[i];
        activation = product.sigmoid();
    }
    Tensor2d<double> output = activation.softmax();

    // Get max probability for each input (argmax)
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

double Network::train_step(Tensor2d<double> &x, std::vector<int> const &y) {
    int batch_size = x.cols;
    // Forward flow of the network
    Tensor2d<double> activation = x;
    // Caching results for backprop
    std::vector<Tensor2d<double> > products;
    std::vector<Tensor2d<double> > activations;
    activations.push_back(activation);
    for (int i = 0; i < (num_layers_ - 1); ++i) {
        Tensor2d<double> product = (weights[i]->matmul(activation)) + *biases[i];
        products.push_back(product);
        activation = product.sigmoid();
        activations.push_back(activation);
    }
    Tensor2d<double> output = activation.softmax();
    double loss = crossEntropy(output, y);

    // backprop
    std::vector<Tensor2d<double> > weight_gradients(weights.size());
    std::vector<Tensor2d<double> > bias_gradients(biases.size());

    Tensor2d<double> cost_prime = softmaxPrime(output, y);

    Tensor2d<double> delta = cost_prime * products[products.size() - 1].sigmoidPrime();

    // Last layer's gradients is a special case
    weight_gradients[weight_gradients.size() - 1] = (delta.matmul(
            activations[activations.size() - 2].transpose())) / batch_size;
    bias_gradients[bias_gradients.size() - 1] = delta;

    //The rest follows a pattern
    for (int j = 2; j < num_layers_; ++j) {
        Tensor2d<double> rp = (products[products.size() - j]).sigmoidPrime();
        delta = ((weights[weights.size() - j + 1]->transpose()).matmul(delta)) * rp;
        bias_gradients[bias_gradients.size() - j] = delta;
        Tensor2d<double> preDiv = (delta.matmul(activations[activations.size() - j - 1].transpose()));
        weight_gradients[weight_gradients.size() - j] = preDiv / batch_size;
    }

    updateWeights(weight_gradients, bias_gradients);

    return loss;
}


void
Network::updateWeights(std::vector<Tensor2d<double> > &weight_gradients,
                       std::vector<Tensor2d<double> > &bias_gradients) {
    assert(weight_gradients.size() == weights.size());
    assert(bias_gradients.size() == biases.size());
    assert(bias_gradients.size() == weights.size());

    for (int i = 0; i < weight_gradients.size(); ++i) {
        *weights[i] -= weight_gradients[i] * learning_rate_;
        Tensor1d<double> biasStep = (bias_gradients[i] * learning_rate_).rowWiseMean();
        *biases[i] -= biasStep;
    }
}


double Network::crossEntropy(Tensor2d<double> &y_hat, std::vector<int> const &y) {
    double total = 0;
    for (int i = 0; i < y.size(); ++i) {
        double x = y_hat.get(y[i], i);
        // Sets a minimum value to prevent division by zero (log(0))
        total += -log(x < 0.0000000001 ? 0.0000000001 : x);
    }

    return total / y.size(); // batch-wise mean
}


Tensor2d<double> Network::softmaxPrime(Tensor2d<double> &output, std::vector<int> const &y) {
    Tensor2d<double> prime = output;
    for (int i = 0; i < y.size(); ++i) {
        prime.set(y[i], i, prime.get(y[i], i) - 1);
    }

    return prime;
}

void Network::load(std::string path) {
    double value;
    FILE *file = fopen(path.c_str(), "r");
    for (int i = 0; i < num_layers_ - 1; ++i) {
        for (int j = 0; j < weights[i]->rows; ++j) {
            for (int k = 0; k < weights[i]->cols; ++k) {
                int scanResult = fscanf(file, "%lf", &value); // NOLINT(cert-err34-c)
                if (scanResult != 1) {
                    printf("Invalid weights file. Exiting.\n");
                    exit(1);
                }
                weights[i]->set(j, k, value);
            }
        }
        for (int l = 0; l < biases[i]->length; ++l) {
            int scanResult = fscanf(file, "%lf", &value); // NOLINT(cert-err34-c)
            if (scanResult != 1) {
                printf("Invalid weights file. Exiting.\n");
                exit(1);
            }
            biases[i]->set(l, value);
        }
    }
    fclose(file);
}
