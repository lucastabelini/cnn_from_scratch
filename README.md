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
The training should take less than 10s and achieve +90% accuracy.
### Todos
 - [x] Dropout;
 - [x] ReLU;
 - [ ] Tanh;
 - [ ] Leaky ReLU;
 - [ ] Batch normalization;
 - [ ] Convolutional layers;
 - [ ] Max pooling;
 - [ ] Other optimizers (Adam, RMSProp, etc);
 - [ ] Learning rate scheduler;
 - [ ] CUDA?

License
----

MIT
