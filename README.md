# Neural Network in Pure C++


Simple implementation of a neural network in C++.  
Those are NOT this project's goals:
- speed performance;
- use of GPU;
- state-of-the-art performance.

This project's goals:
- not to use extern libraries (i.e. use only STL);
- simple code.

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

 - Gradient check;
 - Regularization;
 - Dropout;
 - Batch normalization;
 - Convolutional layers;

License
----

MIT