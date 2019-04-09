//
// Created by lucas on 04/04/19.
//

#include <math.h> // fmax
#include "../include/Network.h"
#include "../include/Tensor2d.h"


Network::Network(std::vector<unsigned int> &layers, int seed = 0, double learning_rate = 0.75) {
    this->learning_rate_ = learning_rate;
    this->num_layers_ = (unsigned int) layers.size();
    std::default_random_engine generator(seed);
    weights.resize(num_layers_ - 1);
    biases.resize(num_layers_ - 1);
    weight_gradients_.resize(num_layers_ - 1);
    bias_gradients_.resize(num_layers_ - 1);
    products_.resize(num_layers_ - 1);
    activations_.resize(num_layers_); // no -1 since the input will also be placed here to be used in backprop

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


Tensor2d<double> Network::forward(Tensor2d<double> &x) {
    // Forward flow of the network
    Tensor2d<double> activation = x;
    // Caching results for backprop
    activations_[0] = activation;
    for (int i = 0; i < (num_layers_ - 1); ++i) {
        Tensor2d<double> product = (weights[i]->matmul(activation)) + *biases[i];
        products_[i] = product;
        if (i == (num_layers_ - 2)) {
            activation = product;
        } else {
            activation = product.sigmoid();
        }

        activations_[i + 1] = activation; // +1 since the input is the first "activation"
    }

    return activation.softmax();
}

std::vector<int> Network::predict(Tensor2d<double> &x) {
    Tensor2d<double> output = forward(x);
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

double Network::computeGradients(Tensor2d<double> &x, std::vector<int> const &y) {
    int batch_size = x.cols;
    Tensor2d<double> output = forward(x);
    double loss = crossEntropy(output, y);

    // backprop
    Tensor2d<double> cost_prime = crossEntropyPrime(output, y) / batch_size;

    Tensor2d<double> delta = cost_prime; //* products_[products_.size() - 1].sigmoidPrime();

    // Last layer's gradients is a special case
    weight_gradients_[weight_gradients_.size() - 1] = (delta.matmul(
            activations_[activations_.size() - 2].transpose()));
    bias_gradients_[bias_gradients_.size() - 1] = delta.rowWiseSum();

    //The rest follows a pattern
    for (int j = 2; j < num_layers_; ++j) {
        Tensor2d<double> activationPrime = (products_[products_.size() - j]).sigmoidPrime();
        delta = ((weights[weights.size() - j + 1]->transpose()).matmul(delta)) * activationPrime;
        bias_gradients_[bias_gradients_.size() - j] = delta.rowWiseSum();

        weight_gradients_[weight_gradients_.size() - j] =
                (delta.matmul(activations_[activations_.size() - j - 1].transpose()));
    }

    return loss;
}

double Network::trainStep(Tensor2d<double> &x, std::vector<int> const &y) {
    double loss = computeGradients(x, y);
    updateWeights();

    return loss;
}


void
Network::updateWeights() {
    for (int i = 0; i < weight_gradients_.size(); ++i) {
        Tensor2d<double> weightStep = weight_gradients_[i] * learning_rate_;
        Tensor1d<double> biasStep = bias_gradients_[i] * learning_rate_;
        *weights[i] -= weightStep;
        *biases[i] -= biasStep;
    }
}


double Network::crossEntropy(Tensor2d<double> &y_hat, std::vector<int> const &y) {
    double total = 0;
    for (int i = 0; i < y_hat.cols; ++i) {
        double x = y_hat.get(y[i], i);
        // Sets a minimum value to prevent division by zero (log(0))
        total += -log(x < 0.0000000001 ? 0.0000000001 : x);
//        for (int j = 0; j < y_hat.rows; ++j) {
//            if (y[i] == j) {
//                total += -log(fmax(y_hat.get(j, i), 1e-8));
//            } else {
//                total += -log(fmax(1 - y_hat.get(j, i), 1e-8));
//            }
//        }
    }

    return total / y.size(); // batch-wise mean
}


Tensor2d<double> Network::crossEntropyPrime(Tensor2d<double> &output, std::vector<int> const &y) {
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

double Network::gradientCheck(Tensor2d<double> &x, std::vector<int> const &y, double h, int max_gradients_check) {
    int gradients_check;
    double weight_ijk, bias_i_j_before, loss_minus, loss_plus, numerical_dw, analytical_dw,
            numerical_db, analytical_db;
    Tensor2d<double> output;


    computeGradients(x, y);

    double max_relative_error = 0, relative_error;
    printf("Checking gradients...");
    for (int i = 0; i < num_layers_ - 1; ++i) {
        // Weights gradient check
        gradients_check = 0;
        for (int j = 0; j < weights[i]->rows; ++j) {
            for (int k = 0; k < weights[i]->cols; ++k) {
                if (gradients_check > max_gradients_check) break;
                gradients_check++;
                weight_ijk = weights[i]->get(j, k);

                weights[i]->set(j, k, weight_ijk - h);
                output = forward(x); // f(w - h)
                loss_minus = crossEntropy(output, y);

                weights[i]->set(j, k, weight_ijk + h);
                output = forward(x); // f(w + h)
                loss_plus = crossEntropy(output, y);

                numerical_dw = (loss_plus - loss_minus) / (2 * h);
                analytical_dw = weight_gradients_[i].get(j, k);
                relative_error =
                        fabs(analytical_dw - numerical_dw) / (fmax(1e-8, fabs(analytical_dw) + fabs(numerical_dw)));
                if (relative_error > max_relative_error) {
                    max_relative_error = relative_error;
                    printf("\rMaximum relative error: %e", max_relative_error);
                    fflush(stdout);
                }

                weights[i]->set(j, k, weight_ijk);
            }
        }
        // Bias gradient check
        gradients_check = 0;
        for (int j = 0; j < biases[i]->length; ++j) {
            if (gradients_check > max_gradients_check) break;
            gradients_check++;
            bias_i_j_before = (*biases[i])[j];

            biases[i]->set(j, bias_i_j_before - h);
            output = forward(x); // f(w - h)
            loss_minus = crossEntropy(output, y);

            biases[i]->set(j, bias_i_j_before + h);
            output = forward(x); // f(w + h)
            loss_plus = crossEntropy(output, y);

            numerical_db = (loss_plus - loss_minus) / (2 * h);
            analytical_db = bias_gradients_[i][j];
            relative_error =
                    fabs(analytical_db - numerical_db) / (fmax(1e-8, fabs(analytical_db) + fabs(numerical_db)));
            if (relative_error > max_relative_error) {
                printf("\rMaximum relative error: %e", max_relative_error);
                fflush(stdout);
            }

            biases[i]->set(j, bias_i_j_before);
        }
    }
    printf("\rMaximum relative error: %e\n", max_relative_error);

    return 0;
}
