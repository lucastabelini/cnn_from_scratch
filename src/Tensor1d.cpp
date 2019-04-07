//
// Created by lucas on 07/04/19.
//

#include "../include/Tensor1d.h"


template
class Tensor1d<float>;

template
class Tensor1d<double>;

template<typename T>
void Tensor1d<T>::randn(std::default_random_engine generator, std::normal_distribution<T> distribution) {
    for (int i = 0; i < length; ++i) {
        data_[i] = distribution(generator);
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
};
