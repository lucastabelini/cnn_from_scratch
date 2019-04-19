# Neural Network in Pure C++

Simple modular implementation of a neural network in C++ using only the STL. 

### Installation
Get the MNIST data set:

```sh
bash get_mnist.sh
```
Generate your Makefile:
```sh
cmake -DCMAKE_BUILD_TYPE=Release
```
Make the code:
```sh
make
```
Run:
```sh
./neural_net_in_cpp data
```
The training should take about a minute and achieve ~97% accuracy.

### Todos
 - [x] Fully connected;
 - [x] Sigmoid;
 - [x] Dropout;
 - [x] ReLU;
 - [ ] Tanh;
 - [ ] Leaky ReLU;
 - [ ] Batch normalization;
 - [x] Convolutional layers;
 - [x] Max pooling;
 - [ ] Other optimizers (Adam, RMSProp, etc);
 - [x] Learning rate scheduler;
 - [ ] Plots;
 - [ ] Filter visualization
 - [ ] CUDA?

License
----

MIT
