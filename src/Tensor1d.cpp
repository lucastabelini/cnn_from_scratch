//
// Created by lucas on 07/04/19.
//

#include "../include/Tensor1d.h"


template
class Tensor1d<float>;

template
class Tensor1d<double>;

template<typename T>
void Tensor1d<T>::randn(std::default_random_engine generator, std::normal_distribution<T> distribution, double multiplier) {
    for (int i = 0; i < length; ++i) {
        data_[i] = distribution(generator) * multiplier;
    }
}

template<typename T>
T Tensor1d<T>::operator[](int i) {
    return data_[i];
}

template<typename T>
Tensor1d<T> Tensor1d<T>::operator-=(Tensor1d<T> &difference) {
    Tensor1d<T> result(length);
    for (int i = 0; i < length; ++i) {
        result.set(i, data_[i] - difference[i]);
    }

    return result;
}

template<typename T>
Tensor1d<T>::Tensor1d(int length) {
    this->length = length;

    data_ = new T[length];
}

template<typename T>
void Tensor1d<T>::set(int i, T value) {
    data_[i] = value;
}

template<typename T>
Tensor1d<T> Tensor1d<T>::operator*(T multiplier) {
    Tensor1d<T> product(length);
    for (int i = 0; i < length; ++i) {
        product.set(i, data_[i] * multiplier);
    }

    return product;
}

template<typename T>
Tensor1d<T> &Tensor1d<T>::operator=(const Tensor1d<T> &other) {
    if (this != &other) {
        T *new_data = new T[other.length];
        std::copy(other.data_, other.data_ + other.length, new_data);

        if (length != 0) {
            delete[] data_;
        }

        data_ = new_data;
        length = other.length;
    }

    return *this;
}

template<typename T>
Tensor1d<T>::~Tensor1d() {
    delete[] data_;
};
