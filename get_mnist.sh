echo "Downloading MNIST data set..."
mkdir data

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "Unpacking..."
gunzip train-images-idx3-ubyte.gz && mv train-images-idx3-ubyte data/
gunzip train-labels-idx1-ubyte.gz && mv train-labels-idx1-ubyte data/
gunzip t10k-images-idx3-ubyte.gz && mv t10k-images-idx3-ubyte data/
gunzip t10k-labels-idx1-ubyte.gz && mv t10k-labels-idx1-ubyte data/


