//
// Created by lucas on 04/04/19.
//

#include "../include/Tensor2d.h"

template
class Tensor2d<float>;

template
class Tensor2d<double>;

template<typename T>
Tensor2d<T>::Tensor2d() = default;

template<typename T>
Tensor2d<T>::Tensor2d(int rows, int cols) {
    assert(rows > 0);
    assert(cols > 0);
    this->rows = rows;
    this->cols = cols;
    this->data_ = new T[rows * cols];
}

template<typename T>
void
Tensor2d<T>::randn(std::default_random_engine generator, std::normal_distribution<T> distribution, double multiplier) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            set(i, j, distribution(generator) * multiplier);
        }
    }
}

template<typename T>
T Tensor2d<T>::get(int i, int j) {
    return data_[i * cols + j];
}

template<typename T>
T Tensor2d<T>::set(int i, int j, T value) {
    data_[i * cols + j] = value;
}

template<typename T>
Tensor2d<T> Tensor2d<T>::transpose() {
    Tensor2d<T> t(cols, rows);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            t.set(j, i, this->get(i, j));
        }
    }
    return t;
}

//template<typename T>
//void Tensor2d<T>::full(T value) {
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols_; ++j) {
//            this->set(i, j, value);
//        }
//    }
//}

template<typename T>
Tensor2d<T> Tensor2d<T>::matmul(Tensor2d<T> other) {
    assert(this->cols == other.rows);
    Tensor2d<T> product(this->rows, other.cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            T value = 0;
            for (int k = 0; k < other.rows; ++k) {
                value += this->get(i, k) * other.get(k, j);
            }
            product.set(i, j, value);
        }
    }
    return product;
}


//template<typename T>
//Tensor2d<T> Tensor2d<T>::relu() {
//    Tensor2d<T> reluResult(this->rows, this->cols_);
//    for (int j = 0; j < this->rows; ++j) {
//        for (int k = 0; k < this->cols_; ++k) {
//            T x = this->get(j, k);
//            reluResult.set(j, k, x > 0 ? x : 0);
//        }
//    }
//    return reluResult;
//}

template<typename T>
T sigmoid(T x) {
    return 1.0 / (1.0 + exp(-x));
};

template<typename T>
Tensor2d<T> Tensor2d<T>::sigmoid() {
    Tensor2d<T> result(this->rows, this->cols);
    for (int j = 0; j < this->rows; ++j) {
        for (int k = 0; k < this->cols; ++k) {
            T x = this->get(j, k);
            result.set(j, k, 1.0 / (1.0 + exp(-x)));
        }
    }
    return result;
}

template<typename T>
T sigmoidPrime(T x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
};

template<typename T>
Tensor2d<T> Tensor2d<T>::sigmoidPrime() {
    Tensor2d<T> result(this->rows, this->cols);
    for (int j = 0; j < this->rows; ++j) {
        for (int k = 0; k < this->cols; ++k) {
            T x = this->get(j, k);
            result.set(j, k, ::sigmoidPrime(x));
        }
    }
    return result;
}



//template<typename T>
//Tensor2d<T> Tensor2d<T>::reluPrime(Tensor2d<T> &x) {
//    Tensor2d<T> prime(x.rows, x.cols_);
//    for (int i = 0; i < x.rows; ++i) {
//        for (int j = 0; j < x.cols_; ++j) {
//            prime.set(i, j, x.get(i, j) > 0 ? 1 : 0);
//        }
//    }
//
//    return prime;
//}

template<typename T>
Tensor2d<T> Tensor2d<T>::softmax() {
    //Softmax with max trick to avoid overflows
    Tensor2d<T> probabilities(rows, cols);
    for (int j = 0; j < cols; ++j) {
        T columnMax = -1; // useless value so my IDE stops screaming at me, will always be replaced
        for (int i = 0; i < rows; ++i) {
            if (i == 0 || get(i, j) > columnMax) {
                columnMax = get(i, j);
            }
        }

        T denominator = 0;
        for (int i = 0; i < rows; ++i) {
            T x = get(i, j);
            denominator += exp(get(i, j) - columnMax);
        }


        for (int i = 0; i < rows; ++i) {
            probabilities.set(i, j, exp(get(i, j) - columnMax) / denominator);
        }

    }
    return probabilities;
}

template<typename T>
Tensor1d<T> Tensor2d<T>::rowWiseSum() {
    Tensor1d<T> mean(rows);
    for (int i = 0; i < rows; ++i) {
        T total = 0;
        for (int j = 0; j < cols; ++j) {
            total += get(i, j);
        }
        mean.set(i, total);
    }
    return mean;
}

template<typename T>
Tensor2d<T> Tensor2d<T>::operator+(Tensor1d<T> &bias) {
    assert(bias.length == this->rows);
    Tensor2d<T> sum(this->rows, this->cols);
    for (int k = 0; k < this->cols; ++k) {
        for (int j = 0; j < this->rows; ++j) {
            sum.set(j, k, this->get(j, k) + bias[j]);
        }
    }

    return sum;
}

template<typename T>
Tensor2d<T> Tensor2d<T>::operator*(Tensor2d<T> multiplier) {
    assert(rows == multiplier.rows);
    assert(cols == multiplier.cols);
    Tensor2d<T> product(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            product.set(i, j, get(i, j) * multiplier.get(i, j));
        }
    }

    return product;
}

template<typename T>
Tensor2d<T> Tensor2d<T>::operator*(T multiplier) {
    Tensor2d<T> product(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            product.set(i, j, get(i, j) * multiplier);
        }
    }

    return product;
}

template<typename T>
Tensor2d<T> Tensor2d<T>::operator/(T divisor) {
    Tensor2d<T> quotient(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            quotient.set(i, j, get(i, j) / divisor);
        }
    }

    return quotient;
}

template<typename T>
Tensor2d<T> &Tensor2d<T>::operator-=(Tensor2d<T> difference) {
    assert(rows == difference.rows);
    assert(cols == difference.cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            set(i, j, get(i, j) - difference.get(i, j));
        }
    }

    return *this;
}

template<typename T>
Tensor2d<T>::~Tensor2d() {
    delete[] data_;
}

template<typename T>
Tensor2d<T> &Tensor2d<T>::operator=(const Tensor2d<T> &other) {
    if (this != &other) {
        T *new_data = new T[other.rows * other.cols];
        std::copy(other.data_, other.data_ + other.rows * other.cols, new_data);

        if (rows * cols != 0) {
            delete[] data_;
        }

        data_ = new_data;
        rows = other.rows;
        cols = other.cols;
    }

    return *this;
}

template<typename T>
Tensor2d<T>::Tensor2d(const Tensor2d<T> &other) : rows(other.rows), cols(other.cols),
                                                  data_(new T[other.rows * other.cols]) {
    std::copy(other.data_, other.data_ + other.rows * other.cols, data_);
}

template<typename T>
void Tensor2d<T>::print() {
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

template<typename T>
double Tensor2d<T>::sum() {
    double total = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            total += get(i, j);
        }
    }
    return total;
}

template<typename T>
Tensor2d<T> Tensor2d<T>::operator+(Tensor2d<T> other) {
    assert(rows == other.rows);
    assert(cols == other.cols);
    Tensor2d<T> sum(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            sum.set(i, j, get(i, j) + other.get(i, j));
        }
    }

    return sum;
}
