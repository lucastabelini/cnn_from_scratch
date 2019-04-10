#include <iostream>
#include "../include/Tensor2d.h"
#include "../include/Network.h"
#include "../include/MNISTDataLoader.h"

using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Please provide the data directory path as an argument. Exiting.\n");
        exit(1);
    }
    printf("Data directory: %s\n", argv[1]);
    string data_path = argv[1];

    printf("Loading training set... ");
    fflush(stdout);
    MNISTDataLoader train_loader(data_path + "/train-images-idx3-ubyte", data_path + "/train-labels-idx1-ubyte", 32);
    printf("Loaded.\n");

    // Define network with 3 layers (1 input, 1 hidden and 1 output), input size = 784, hidden layer size = 30, output size = 10
    vector<unsigned int> layers = {784, 30, 10};
    Network net(layers, 0, 2.0, 1e-3);


    int epochs = 1;
    printf("Training for %d epoch(s).\n", epochs);
    // Train network
    int num_train_batches = train_loader.getNumBatches();
    for (int k = 0; k < epochs; ++k) {
        printf("Epoch %d\n", k + 1);
        for (int i = 0; i < num_train_batches; ++i) {
            pair<Tensor2d<double>, vector<int> > xy = train_loader.nextBatch();
            double loss = net.trainStep(xy.first, xy.second);
            if ((i + 1) % 10 == 0) {
                printf("\rIteration %d/%d - Batch Loss: %.4lf", i + 1, num_train_batches, loss);
                fflush(stdout);
            }
        }
        printf("\n");
    }

    // Save weights
    net.save();


    printf("Loading testing set... ");
    fflush(stdout);
    MNISTDataLoader test_loader(data_path + "/t10k-images-idx3-ubyte", data_path + "/t10k-labels-idx1-ubyte", 32);
    printf("Loaded.\n");

    // Test and measure accuracy
    int hits = 0;
    int total = 0;
    printf("Testing...\n");
    int num_test_batches = test_loader.getNumBatches();
    for (int i = 0; i < num_test_batches; ++i) {
        if ((i + 1) % 10 == 0 || i == (num_test_batches - 1)) {
            printf("\rIteration %d/%d", i + 1, num_test_batches);
            fflush(stdout);
        }
        pair<Tensor2d<double>, vector<int> > xy = test_loader.nextBatch();
        vector<int> predictions = net.predict(xy.first);
        for (int j = 0; j < predictions.size(); ++j) {
            if (predictions[j] == xy.second[j]) {
                hits++;
            }
        }
        total += xy.second.size();
    }
    printf("\n");

    printf("Accuracy: %.2f%% (%d/%d)\n", ((double) hits * 100) / total, hits, total);

    return 0;
}