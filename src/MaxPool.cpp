//
// Created by tabelini on 18/04/19.
//

#include "../include/MaxPool.h"

MaxPool::MaxPool(int size, int stride) {
    size_ = size;
    stride_ = stride;
}

Tensor<double> &MaxPool::forward(Tensor<double> &input) {
    int w = ((input.dims[3] - (size_ - 1) - 1) / stride_) + 1;
    int h = ((input.dims[2] - (size_ - 1) - 1) / stride_) + 1;
    int dims[] = {input.dims[0], input.dims[1], h, w};
    output_ = Tensor<double>(4, dims);
    indexes = Tensor<int>(4, dims);
    for (int i = 0; i < input.dims[0]; ++i) { // for each batch image
        for (int j = 0; j < input.dims[1]; ++j) { // for each image channel
            for (int k = 0; k < dims[2]; ++k) { // for each output y
                for (int l = 0; l < dims[3]; ++l) { // for each output x
                    double max = -999999999; // -infinity
                    int index = 0;
                    for (int m = 0; m < size_; ++m) {
                        for (int n = 0; n < size_; ++n) {
                            int input_y = k * stride_ + m;
                            int input_x = l * stride_ + n;
                            double value = input.get(i, j, input_y, input_x);
                            if (value > max) {
                                index = m * size_ + n;
                                max = value;
                            }
                        }
                    }
                    output_.set(i, j, k, l, max);
                    indexes.set(i, j, k, l, index);
                }
            }
        }
    }
    input_ = input;

    return output_;
}

Tensor<double> MaxPool::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> input_gradient(input_.num_dims, input_.dims);
    input_gradient.zero();

    for (int i = 0; i < input_.dims[0]; ++i) { // for each batch image
        for (int j = 0; j < input_.dims[1]; ++j) { // for each image channel
            for (int k = 0; k < output_.dims[2]; ++k) { // for each output y
                for (int l = 0; l < output_.dims[3]; ++l) { // for each output x
                    double chain_grad = chainGradient.get(i, j, k, l);
                    int index = indexes.get(i, j, k, l);
                    int m = index / size_;
                    int n = index % size_;
                    int input_y = k * stride_ + m;
                    int input_x = l * stride_ + n;
                    input_gradient.set(i, j, input_y, input_x, chain_grad);
                }
            }
        }
    }

    return input_gradient;
}

void MaxPool::load(FILE *file_model) {

}

void MaxPool::save(FILE *file_model) {

}
