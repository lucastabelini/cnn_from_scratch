//
// Created by lucas on 12/04/19.
//

#include "../include/Tensor.h"

template
class Tensor<float>;

template
class Tensor<double>;

template<typename T>
T Tensor<T>::get(int i, int j) {
    assert(num_dims == 2);
    return data_[j + i * dims[1]];
}

template<typename T>
T Tensor<T>::get(int i) {
    assert(num_dims == 1);
    return data_[i];
}

template<typename T>
void Tensor<T>::set(int i, int j, T value) {
    assert(num_dims == 2);
    data_[j + i * dims[1]] = value;
}

template<typename T>
void Tensor<T>::set(int i, T value) {
    data_[i] = value;
}

template<typename T>
void Tensor<T>::view(int num_dims, int *dims) {
    assert(num_dims > 0 && num_dims <= 4);
    this->num_dims = num_dims;
    std::copy(dims, dims + 4, this->dims);
}

template<typename T>
Tensor<T>::Tensor(int num_dims, int *dims) {
    assert(num_dims > 0 && num_dims <= 4);
    int size = 1;
    for (int i = 0; i < num_dims; ++i) {
        size *= dims[i];
        this->dims[i] = dims[i];
    }
    size_ = size;
//    std::shared_ptr<T[]> data_sp(new T[size_]);
    T *data_sp = new T[size_];
    data_ = data_sp;
    this->num_dims = num_dims;
}

template<typename T>
T Tensor<T>::get(int i, int j, int k) {
    assert(num_dims == 3);
    return data_[k + j * dims[2] + i * dims[1] * dims[2]];
}

template<typename T>
T Tensor<T>::get(int i, int j, int k, int l) {
    assert(num_dims == 4);
    return data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]];
}

template<typename T>
void Tensor<T>::set(int i, int j, int k, T value) {
    assert(num_dims == 3);
    data_[k + j * dims[2] + i * dims[1] * dims[2]] = value;
}

template<typename T>
void Tensor<T>::set(int i, int j, int k, int l, T value) {
    assert(num_dims == 4);
    data_[l + k * dims[3] + j * dims[2] * dims[3] + i * dims[1] * dims[2] * dims[3]] = value;
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other) : size_(other.size_), num_dims(other.num_dims),
                                            data_(new T[other.size_]) {
    std::copy(other.data_, other.data_ + size_, data_);
    std::copy(other.dims, other.dims + 4, dims);
}

template<typename T>
Tensor<T>::~Tensor() {
    delete[] data_;
}


template<typename T>
Tensor<T> Tensor<T>::matmul(Tensor<T> other) {
    assert(num_dims == 2 && other.num_dims == 2);
    assert(dims[1] == other.dims[0]);

    int new_dims[] = {dims[0], other.dims[1]};
    Tensor<T> product(2, new_dims);
    for (int i = 0; i < this->dims[0]; ++i) {
        for (int j = 0; j < other.dims[1]; ++j) {
            T value = 0;
            for (int k = 0; k < other.dims[0]; ++k) {
                value += this->get(i, k) * other.get(k, j);
            }
            product.set(i, j, value);
        }
    }
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::matrixTranspose() {
    assert(num_dims == 2);
    int new_dims[] = {dims[1], dims[0]};
    Tensor<T> transpose(num_dims, new_dims);
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            transpose.set(j, i, get(i, j));
        }
    }

    return transpose;
}


template<typename T>
Tensor<T> Tensor<T>::relu() {
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        T x = data_[i];
        result.data_[i] = x > 0 ? x : 0;
    }

    return result;
}

template<typename T>
T sigmoid(T x) {
    return 1.0 / (1.0 + exp(-x));
};

template<typename T>
Tensor<T> Tensor<T>::sigmoid() {
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        T x = data_[i];
        result.data_[i] = ::sigmoid(x);
    }

    return result;
}

template<typename T>
T sigmoidPrime(T x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
};

template<typename T>
Tensor<T> Tensor<T>::sigmoidPrime() {
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        T x = data_[i];
        result.data_[i] = ::sigmoidPrime(x);
    }

    return result;
}

template<typename T>
T Tensor<T>::sum() {
    T total = 0;
    for (int i = 0; i < size_; ++i) {
        total += data_[i];
    }
    return 0;
}

template<typename T>
Tensor<T> Tensor<T>::softmax() {
    assert(num_dims == 2);
    //Softmax with max trick to avoid overflows
    int rows = dims[0], cols = dims[1];
    Tensor<T> probabilities(2, dims);
    for (int i = 0; i < rows; ++i) {
        T row_max = -1; // useless value so my IDE stops screaming at me, will always be replaced
        for (int j = 0; j < cols; ++j) {
            if (j == 0 || get(i, j) > row_max) {
                row_max = get(i, j);
            }
        }

        T denominator = 0;
        for (int j = 0; j < cols; ++j) {
            T x = get(i, j);
            denominator += exp(get(i, j) - row_max);
        }


        for (int j = 0; j < cols; ++j) {
            probabilities.set(i, j, exp(get(i, j) - row_max) / denominator);
        }

    }
    return probabilities;
}

template<typename T>
Tensor<T> Tensor<T>::reluPrime() {
    assert(num_dims == 2);
    Tensor<T> prime(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        prime.data_[i] = data_[i] > 0 ? 1 : 0;
    }
    return prime;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(Tensor<T> &other) {
    if (other.num_dims == 1 && other.size_ == this->dims[1] && num_dims == 2) {
        // if other is a 1d tensor and this is a 2d tensor
        Tensor<T> sum(num_dims, dims);
        for (int k = 0; k < this->dims[0]; ++k) {
            for (int j = 0; j < this->dims[1]; ++j) {
                sum.set(k, j, get(k, j) + other.get(j));
            }
        }


        return sum;
    } else if (other.num_dims == num_dims && other.size_ == size_) {
        Tensor<T> sum(num_dims, dims);
        for (int i = 0; i < size_; ++i) {
            sum.data_[i] = data_[i] + other.data_[i];
        }
        return sum;
    }
    throw std::logic_error("Undefined sum");
}


template<typename T>
Tensor<T> Tensor<T>::operator*(Tensor<T> other) {
    assert(size_ == other.size_);
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        product.data_[i] = data_[i] * other.data_[i];
    }
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T multiplier) {
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        product.data_[i] = data_[i] * multiplier;
    }
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(T divisor) {
    Tensor<T> quotient(num_dims, dims);
    for (int i = 0; i < size_; ++i) {
        quotient.data_[i] = data_[i] / divisor;
    }
    return quotient;
}

template<typename T>
Tensor<T> Tensor<T>::operator-=(Tensor<T> difference) {
    assert(size_ == difference.size_);
    for (int i = 0; i < size_; ++i) {
        data_[i] = data_[i] - difference.data_[i];
    }
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::columnWiseSum() {
    assert(num_dims == 2);
    int rows = dims[0], cols = dims[1];
    int dims[] = {cols};
    Tensor<T> mean(1, dims);
    for (int i = 0; i < cols; ++i) {
        T total = 0;
        for (int j = 0; j < rows; ++j) {
            total += get(j, i);
        }
        mean.set(i, total);
    }
    return mean;
}

template<typename T>
void
Tensor<T>::randn(std::default_random_engine generator, std::normal_distribution<T> distribution, double multiplier) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = distribution(generator) * multiplier;
    }
}

template<typename T>
void Tensor<T>::print() {
    if (num_dims == 2) {
        int rows = dims[0], cols = dims[1];
        std::cout << "Tensor2D (" << rows << ", " << cols << ")\n[";
        for (int i = 0; i < rows; ++i) {
            if (i != 0) std::cout << " ";
            std::cout << "[";
            for (int j = 0; j < cols; ++j) {
                if (j == (cols - 1)) {
                    printf("%.18lf", get(i, j));
                } else {
                    printf("%.18lf ", get(i, j));
                }

            }
            if (i == (rows - 1)) {
                std::cout << "]]\n";
            } else {
                std::cout << "]\n";
            }
        }
    }
}

template<typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
    if (this != &other) {
        T *new_data = new T[other.size_];
        std::copy(other.data_, other.data_ + other.size_, new_data);
        if (size_ != -1) {
            delete[] data_;
        }
        size_ = other.size_;
        std::copy(other.dims, other.dims + 4, dims);
        num_dims = other.num_dims;
        data_ = new_data;
    }

    return *this;
}

template<typename T>
void Tensor<T>::dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = (distribution(generator) < p) / p;
    }
}