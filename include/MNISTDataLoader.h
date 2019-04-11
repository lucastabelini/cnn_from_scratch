//
// Created by lucas on 05/04/19.
//

#ifndef NEURAL_NET_IN_CPP_MNISTDATALOADER_H
#define NEURAL_NET_IN_CPP_MNISTDATALOADER_H

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <errno.h>
#include <string.h>
#include "Tensor2d.h"

/*
 * Utility to read MNIST data.
 */

class MNISTDataLoader {
private:
    std::vector<std::vector<std::vector<double> > > images_;
    std::vector<int> labels_;

    unsigned int batch_idx_ = 0;
    unsigned int batch_size_;
    unsigned int rows_ = 28, cols_ = 28, num_images_ = 0;

    /*
     * Loads MNIST's labels
     */
    void loadLabels(std::string const &path);

    /*
     * Converts an array of 4 bytes to an unsigned int
     */
    unsigned int bytesToUInt(const char *bytes);

    /*
     * Loads MNIST's image set
     */
    void loadImages(std::string const &path);

public:
    MNISTDataLoader(std::string const &imagesPath, std::string const &labelsPath, unsigned int batch_size);

    /*
     * Get the number of batches in the data set.
     */
    int getNumBatches();

//    void printImage(int idx);

    /*
     * Get next batch. Last batch of the dataset may not have the same size of the others.
     * Is cyclical, so it can be used indefinitely.
     */
    std::pair<Tensor2d<double>, std::vector<int> > nextBatch();
};

#endif //NEURAL_NET_IN_CPP_MNISTDATALOADER_H
