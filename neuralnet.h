#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

// Typedefs for clarity
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Helper functions
Matrix zero_init(int rows, int cols);
Matrix random_init(int rows, int cols);

// SoftMaxCrossEntropy Class
class SoftMaxCrossEntropy {
public:
    Vector softmax(const Vector& z);
    double cross_entropy(int y, const Vector& y_hat);
    std::tuple<Vector, double> forward(const Vector& z, int y);
    Vector backward(int y, const Vector& y_hat);
};

// Sigmoid Activation Class
class Sigmoid {
public:
public:
    Vector z;  // Cache for backward pass
    Vector forward(const Vector& x);
    Vector backward(const Vector& dz);
};

// Linear Layer Class
class Linear {
public:
    Matrix weights;  // Weights and biases
    Matrix gradients;  // Gradients for backpropagation
    double learning_rate;
    Vector input_with_bias;  // Cache for forward pass
    Linear(int input_size, int output_size, Matrix (init_fn)(int, int), double lr);
    Vector forward(const Vector& x);
    Vector backward(const Vector& dz);
    void step();
};

// Neural Network Class
class NN {
public:
    Linear linear1;
    Sigmoid sigmoid;
    Linear linear2;
    SoftMaxCrossEntropy softmax;
    NN(int input_size, int hidden_size, int output_size, Matrix (init_fn)(int, int), double lr);
    std::tuple<Vector, double> forward(const Vector& x, int y);
    Vector forward_1(const Vector& x, int y);
    std::tuple<Vector, double> forward_2(const Vector& z, int y);
    void backward(int y, const Vector& y_hat);
    Vector backward_1(int y, const Vector& y_hat);
    void backward_2(const Vector& y_hat);
    void step();
    double compute_loss(const Matrix& X, const Vector& y);
    std::tuple<std::vector<double>, std::vector<double>> train(
        const Matrix& X_train, const Vector& y_train,
        const Matrix& X_test, const Vector& y_test,
        int epochs);
    std::tuple<std::vector<double>, std::vector<double>> train_data(
        const Matrix& X_train, const Vector& y_train,
        const Matrix& X_test, const Vector& y_test,
        int epochs, int batch_size, int nproc, int pid);
    std::tuple<std::vector<double>, std::vector<double>> train_model(
        const Matrix& X_train, const Vector& y_train,
        const Matrix& X_test, const Vector& y_test,
        int epochs, int batch_size, int nproc, int pid);
    std::tuple<Vector, double> test(const Matrix& X, const Vector& y);
};

#endif // NEURALNET_H