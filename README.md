# Neural Network in Pure C++


Simple implementation of a neural network in C++ using only the STL. 


### Installation
Get the MNIST data set:

```sh
bash get_mnist.sh
```
Generate your Makefile:
```sh
cmake
```
Make the code:
```sh
make
```
Run:
```sh
./neural_net_in_cpp data
```
The training should take less than a minute and achieve +90% accuracy.
### Todos

 - [x] Gradient check;
 - [x] Regularization;
 - [ ] Dropout;
 - [ ] Batch normalization;
 - [ ] Convolutional layers;

License
----

MIT
