//
// Created by lucas on 05/04/19.
//

#include "../include/MNISTDataLoader.h"


MNISTDataLoader::MNISTDataLoader(std::string const &imagesPath, std::string const &labelsPath,
                                 unsigned int batch_size) {
    this->batch_size_ = batch_size;
    loadImages(imagesPath);
    loadLabels(labelsPath);
}

unsigned int MNISTDataLoader::bytesToUInt(const char *bytes) {
    return ((unsigned char) bytes[0] << 24) | ((unsigned char) bytes[1] << 16) |
           ((unsigned char) bytes[2] << 8) | ((unsigned char) bytes[3] << 0);
}

void MNISTDataLoader::loadImages(std::string const &path) {
    // Info about the dataset's file format can be found at http://yann.lecun.com/exdb/mnist/
    std::ifstream file(path, std::ios::binary | std::ios::in);
    if (!file) {
        std::cerr << "Error: " << strerror(errno);
        exit(1);
    }
    file.clear();
    char bytes[4];
    file.read(bytes, 4); // magic number
    file.read(bytes, 4);
    num_images_ = bytesToUInt(bytes);
    file.read(bytes, 4);
    rows_ = bytesToUInt(bytes);
    file.read(bytes, 4);
    cols_ = bytesToUInt(bytes);

    images_.resize(num_images_);
    char byte;
    for (int i = 0; i < num_images_; ++i) {
        images_[i].resize(rows_);
        for (int j = 0; j < rows_; ++j) {
            images_[i][j].resize(cols_);
            for (int k = 0; k < cols_; ++k) {
                file.read(&byte, 1);
                images_[i][j][k] = (unsigned char) (byte & 0xff);
            }
        }
    }
}

int MNISTDataLoader::getNumBatches() {
    if (num_images_ % batch_size_ == 0) {
        return num_images_ / batch_size_;
    } else {
        return (num_images_ / batch_size_) + 1;
    }
}

void MNISTDataLoader::loadLabels(std::string const &path) {
    std::ifstream file(path, std::ios::binary | std::ios::in);
    if (!file) {
        std::cerr << "Error: " << strerror(errno);
    }
    file.clear();
    char bytes[4];
    file.read(bytes, 4); // magic number
    file.read(bytes, 4);
    num_images_ = bytesToUInt(bytes);

    labels_.resize(num_images_);
    char byte;
    for (int i = 0; i < num_images_; ++i) {
        file.read(&byte, 1);
        labels_[i] = (byte & 0xff);
    }
}

//    void MNISTDataLoader::printImage(int idx) {
//        for (int i = 0; i < rows_; ++i) {
//            for (int j = 0; j < cols_; ++j) {
//                if (images_[idx][i][j] > 127) {
//                    printf("%c", 219);
//                } else {
//                    printf(" ");
//                }
//            }
//            printf("\n");
//        }
//        printf("Label: %d\n", labels_[idx]);
//    }

std::pair<Tensor2d<double>, std::vector<int> > MNISTDataLoader::nextBatch() {
    std::pair<Tensor2d<double>, std::vector<int> > batchXY;
    int imgsMissing = num_images_ - batch_idx_;
    int size = imgsMissing > batch_size_ ? batch_size_ : imgsMissing;
    Tensor2d<double> tensorImgs(rows_ * cols_, size);
    std::vector<int> vecLabels;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < rows_; ++j) {
            for (int k = 0; k < cols_; ++k) {
                tensorImgs.set(j * cols_ + k, i, ((double) (images_[batch_idx_ + i][j][k])) / 255.0);
            }
        }
        vecLabels.push_back(labels_[batch_idx_ + i]);
    }
    batch_idx_ += size;
    if (batch_idx_ == num_images_) {
        batch_idx_ = 0;
    }
    batchXY.first = tensorImgs;
    batchXY.second = vecLabels;
    return batchXY;
}