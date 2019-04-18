//
// Created by lucas on 14/04/19.
//

#include "../include/Conv2d.h"

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    int kernel_dims[] = {out_channels, in_channels, kernel_size, kernel_size};
    kernels = Tensor<double>(4, kernel_dims);
    kernels.randn(generator, distribution, sqrt(2.0 / (kernel_size * kernel_size * out_channels)));

    int bias_dims[] = {out_channels};
    bias = Tensor<double>(1, bias_dims);
    bias.randn(generator, distribution, 0);

    this->stride = stride;
    this->padding = padding;
}

Tensor<double> &Conv2d::forward(Tensor<double> &input) {
    input_ = input;
    product_ = input.convolve2d(kernels, stride, padding, bias);

    return product_;
}

Tensor<double> Conv2d::backprop(Tensor<double> chain_gradient, double learning_rate) {
    Tensor<double> kernels_gradient(kernels.num_dims, kernels.dims);
    Tensor<double> input_gradient(input_.num_dims, input_.dims);
    Tensor<double> bias_gradient(1, bias.dims);
    kernels_gradient.zero();
    input_gradient.zero();
    bias_gradient.zero();

    // backprop convolution -- not using Tensor.convolve2d for efficiency
    for (int i = 0; i < input_.dims[0]; ++i) { // for each batch img
        for (int f = 0; f < kernels.dims[0]; f++) { // for each filter
            int x = -padding;
            for (int cx = 0; cx < chain_gradient.dims[2]; x += stride, cx++) { // for each x in the chain gradient
                int y = -padding;
                for (int cy = 0; cy < chain_gradient.dims[3]; y += stride, cy++) { // for each y in the chain gradient
                    double chain_grad = chain_gradient.get(i, f, cx, cy);
                    for (int fx = 0; fx < kernels.dims[2]; fx++) { // for each x in the filter
                        int ix = x + fx; // input x
                        if (ix >= 0 && ix < input_.dims[2]) {
                            for (int fy = 0; fy < kernels.dims[3]; fy++) { // for each y in the filter
                                int iy = y + fy; // input y
                                if (iy >= 0 && iy < input_.dims[3]) {
                                    for (int fc = 0; fc < kernels.dims[1]; fc++) { // for each channel in the filter
                                        kernels_gradient.add(f, fc, fx, fy, input_.get(i, fc, ix, iy) * chain_grad);
                                        input_gradient.add(i, fc, ix, iy, kernels.get(f, fc, fx, fy) * chain_grad);

                                    }
                                }
                            }
                        }
                    }
                    bias_gradient.add(f, chain_grad);
                }
            }
        }
    }
    kernels -= kernels_gradient * learning_rate;
    bias -= bias_gradient * learning_rate;

    return input_gradient;
}

void Conv2d::load(FILE *file_model) {
    double value;
    for (int i = 0; i < kernels.dims[0]; ++i) {
        for (int j = 0; j < kernels.dims[1]; ++j) {
            for (int k = 0; k < kernels.dims[2]; ++k) {
                for (int l = 0; l < kernels.dims[3]; ++l) {
                    int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
                    if (read != 1) throw std::runtime_error("Invalid model file");
                    kernels.set(i, j, k, l, value);
                }
            }
        }
    }
    for (int m = 0; m < bias.dims[0]; ++m) {
        int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
        if (read != 1) throw std::runtime_error("Invalid model file");
        bias.set(m, value);
    }
}

void Conv2d::save(FILE *file_model) {
    for (int i = 0; i < kernels.dims[0]; ++i) {
        for (int j = 0; j < kernels.dims[1]; ++j) {
            for (int k = 0; k < kernels.dims[2]; ++k) {
                for (int l = 0; l < kernels.dims[3]; ++l) {
                    fprintf(file_model, "%.18lf ", kernels.get(i, j, k, l));
                }
            }
        }
    }
    for (int m = 0; m < bias.dims[0]; ++m) {
        fprintf(file_model, "%.18lf ", bias.get(m));
    }
}
