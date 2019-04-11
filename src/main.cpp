#include <iostream>
#include "../include/Tensor2d.h"
#include "../include/NetworkModel.h"
#include "../include/Module.h"
#include "../include/FullyConnected.h"
#include "../include/Sigmoid.h"
#include "../include/Dropout.h"
#include "../include/SoftmaxClassifier.h"
#include "../include/MNISTDataLoader.h"

using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */

int main(int argc, char **argv) {
    if (argc < 2) {
        throw "Please provide the data directory path as an argument. Exiting.\n";
    }
    printf("Data directory: %s\n", argv[1]);
    string data_path = argv[1];

    printf("Loading training set... ");
    fflush(stdout);
    MNISTDataLoader train_loader(data_path + "/train-images-idx3-ubyte", data_path + "/train-labels-idx1-ubyte", 32);
    printf("Loaded.\n");

    vector<Module *> modules = {new FullyConnected(784, 30), new Sigmoid(), new FullyConnected(30, 10)};
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), 2.0);
//    model.load("network.txt");


    int epochs = 1;
    printf("Training for %d epoch(s).\n", epochs);
    // Train network
    int num_train_batches = train_loader.getNumBatches();
    for (int k = 0; k < epochs; ++k) {
        printf("Epoch %d\n", k + 1);
        for (int i = 0; i < num_train_batches; ++i) {
            pair<Tensor2d<double>, vector<int> > xy = train_loader.nextBatch();
            double loss = model.trainStep(xy.first, xy.second);
            if ((i + 1) % 10 == 0) {
                printf("\rIteration %d/%d - Batch Loss: %.4lf", i + 1, num_train_batches, loss);
                fflush(stdout);
            }
        }
        printf("\n");
    }

    // Save weights
    model.save("network.txt");

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
        vector<int> predictions = model.predict(xy.first);
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