// neuralnet.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

using Vector = VectorXd;
using Matrix = MatrixXd;

// Shuffle function for SGD
void shuffle(Matrix& X, Vector& y, int epoch) {
    mt19937 gen(epoch);
    vector<int> indices(X.rows());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);

    Matrix X_shuffled = X;
    Vector y_shuffled = y;

    for (size_t i = 0; i < indices.size(); ++i) {
        X_shuffled.row(i) = X.row(indices[i]);
        y_shuffled(i) = y(indices[i]);
    }

    X = X_shuffled;
    y = y_shuffled;
}

// Random initialization
Matrix random_init(int rows, int cols) {
    mt19937 gen(rows * cols);
    uniform_real_distribution<> dis(-0.1, 0.1);
    Matrix mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = dis(gen);
    return mat;
}

// Zero initialization
Matrix zero_init(int rows, int cols) {
    return Matrix::Zero(rows, cols);
}

// Sigmoid activation
class Sigmoid {
public:
    Vector z;

    Vector forward(const Vector& x) {
        z = (1 / (1 + (-x.array()).exp())).matrix();
        return z;
    }

    Vector backward(const Vector& dz) {
        return dz.array() * z.array() * (1 - z.array());
    }
};

// Softmax with CrossEntropy
class SoftMaxCrossEntropy {
public:
    Vector softmax(const Vector& z) {
        Vector exp_z = z.array().exp();
        return exp_z / exp_z.sum();
    }

    double cross_entropy(int y, const Vector& y_hat) {
        return -log(y_hat(y));
    }

    pair<Vector, double> forward(const Vector& z, int y) {
        Vector y_hat = softmax(z);
        double loss = cross_entropy(y, y_hat);
        return {y_hat, loss};
    }

    Vector backward(int y, const Vector& y_hat) {
        Vector grad = y_hat;
        grad(y) -= 1;
        return grad;
    }
};

// Linear Layer
class Linear {
public:
    Matrix w;
    Matrix dw;
    Vector x_with_bias;
    double lr;

    Linear(int input_size, int output_size, function<Matrix(int, int)> init_fn, double learning_rate)
        : lr(learning_rate) {
        w = init_fn(output_size, input_size + 1); // +1 for bias
        dw = Matrix::Zero(output_size, input_size + 1);
    }

    Vector forward(const Vector& x) {
        x_with_bias = Vector(x.size() + 1);
        x_with_bias(0) = 1;
        x_with_bias.tail(x.size()) = x;
        return w * x_with_bias;
    }

    Vector backward(const Vector& dz) {
        dw = dz * x_with_bias.transpose();
        return (w.rightCols(w.cols() - 1).transpose() * dz);
    }

    void step() {
        w -= lr * dw;
    }
};

class NN {
private:
    Linear* linear1;
    Sigmoid* sigmoid;
    Linear* linear2;
    SoftMaxCrossEntropy* softmax;

public:
    NN(int input_size, int hidden_size, int output_size,
       function<Matrix(int, int)> weight_init_fn, double learning_rate) {
        // Initialize layers
        linear1 = new Linear(input_size, hidden_size, weight_init_fn, learning_rate);
        sigmoid = new Sigmoid();
        linear2 = new Linear(hidden_size, output_size, weight_init_fn, learning_rate);
        softmax = new SoftMaxCrossEntropy();
    }

    pair<Vector, double> forward(const Vector& x, int y) {
        // Forward pass through the network
        Vector a = linear1->forward(x);
        Vector z = sigmoid->forward(a);
        Vector b = linear2->forward(z);
        auto [y_hat, loss] = softmax->forward(b, y);
        return {y_hat, loss};
    }

    void backward(int y, const Vector& y_hat) {
        // Backward pass through the network
        Vector db = softmax->backward(y, y_hat);
        Vector dz = linear2->backward(db);
        Vector da = sigmoid->backward(dz);
        linear1->backward(da);
    }

    void step() {
        // Update weights for layers
        linear1->step();
        linear2->step();
    }

    double compute_loss(const Matrix& X, const Vector& y) {
        // Compute average cross-entropy loss over dataset
        double total_loss = 0.0;
        for (int i = 0; i < X.rows(); ++i) {
            auto [_, loss] = forward(X.row(i), y(i));
            total_loss += loss;
        }
        return total_loss / X.rows();
    }

    tuple<vector<double>, vector<double>> train(const Matrix& X_tr, const Vector& y_tr,
                                                const Matrix& X_test, const Vector& y_test,
                                                int n_epochs) {
        // Train the network using SGD
        vector<double> train_losses, test_losses;

        for (int epoch = 0; epoch < n_epochs; ++epoch) {
            // Shuffle training data
            Matrix X_tr_shuffled = X_tr;
            Vector y_tr_shuffled = y_tr;
            shuffle(X_tr_shuffled, y_tr_shuffled, epoch);

            for (int i = 0; i < X_tr_shuffled.rows(); ++i) {
                auto [y_hat, loss] = forward(X_tr_shuffled.row(i), y_tr_shuffled(i));
                backward(y_tr_shuffled(i), y_hat);
                step();
            }

            double train_loss = compute_loss(X_tr_shuffled, y_tr_shuffled);
            double test_loss = compute_loss(X_test, y_test);
            train_losses.push_back(train_loss);
            test_losses.push_back(test_loss);
        }

        return {train_losses, test_losses};
    }

    pair<Vector, double> test(const Matrix& X, const Vector& y) {
        // Test the network and compute error rate
        Vector labels(X.rows());
        int correct = 0;

        for (int i = 0; i < X.rows(); ++i) {
            auto [y_hat, _] = forward(X.row(i), y(i));
            int pred = distance(y_hat.data(), max_element(y_hat.data(), y_hat.data() + y_hat.size()));
            labels(i) = pred;
            if (pred == y(i)) {
                correct++;
            }
        }

        double error_rate = 1.0 - static_cast<double>(correct) / X.rows();
        return {labels, error_rate};
    }
};

/ Function prototypes
Matrix load_csv(const string& file_path, Vector& labels);
tuple<Matrix, Vector, Matrix, Vector, string, string, string, int, int, int, double>
parse_args(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    // Parse command-line arguments and load data
    auto [X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr] = parse_args(argc, argv);

    // Define labels (corresponding to class indices)
    vector<string> labels = {"a", "e", "g", "i", "l", "n", "o", "r", "t", "u"};

    // Initialize the neural network
    NN nn(
        X_tr.cols(),   // input_size
        n_hid,         // hidden_size
        labels.size(), // output_size
        init_flag == 2 ? zero_init : random_init, // weight_init_fn
        lr             // learning_rate
    );

    // Train the model
    auto [train_losses, test_losses] = nn.train(X_tr, y_tr, X_test, y_test, n_epochs);

    // Test the model
    auto [train_labels, train_error_rate] = nn.test(X_tr, y_tr);
    auto [test_labels, test_error_rate] = nn.test(X_test, y_test);

    // Write predictions for training data
    ofstream train_out(out_tr);
    for (int i = 0; i < train_labels.size(); ++i) {
        train_out << train_labels[i] << "\n";
    }
    train_out.close();

    // Write predictions for test data
    ofstream test_out(out_te);
    for (int i = 0; i < test_labels.size(); ++i) {
        test_out << test_labels[i] << "\n";
    }
    test_out.close();

    // Write metrics
    ofstream metrics_out(out_metrics);
    for (size_t i = 0; i < train_losses.size(); ++i) {
        int cur_epoch = i + 1;
        metrics_out << "epoch=" << cur_epoch << " crossentropy(train): " << train_losses[i] << "\n";
        metrics_out << "epoch=" << cur_epoch << " crossentropy(validation): " << test_losses[i] << "\n";
    }
    metrics_out << "error(train): " << train_error_rate << "\n";
    metrics_out << "error(validation): " << test_error_rate << "\n";
    metrics_out.close();

    // Print error rates to console
    cout << "Train Error Rate: " << train_error_rate << endl;
    cout << "Test Error Rate: " << test_error_rate << endl;

    return 0;
}

// Utility functions
Matrix load_csv(const string& file_path, Vector& labels) {
    // Load a CSV file and separate the labels (assume the first column contains labels).
    ifstream file(file_path);
    vector<vector<double>> data;
    vector<double> label_vec;

    string line;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string value;
        bool is_label = true;

        while (getline(ss, value, ',')) {
            if (is_label) {
                label_vec.push_back(stoi(value));
                is_label = false;
            } else {
                row.push_back(stod(value));
            }
        }
        data.push_back(row);
    }

    Matrix X(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            X(i, j) = data[i][j];
        }
    }

    labels = Vector::Map(label_vec.data(), label_vec.size());
    return X;
}

tuple<Matrix, Vector, Matrix, Vector, string, string, string, int, int, int, double>
parse_args(int argc, char* argv[]) {
    if (argc != 11) {
        cerr << "Usage: " << argv[0]
             << " <train_input> <validation_input> <train_out> <validation_out>"
                " <metrics_out> <num_epoch> <hidden_units> <init_flag> <learning_rate>"
             << endl;
        exit(1);
    }

    string train_input = argv[1];
    string validation_input = argv[2];
    string train_out = argv[3];
    string validation_out = argv[4];
    string metrics_out = argv[5];
    int num_epoch = stoi(argv[6]);
    int hidden_units = stoi(argv[7]);
    int init_flag = stoi(argv[8]);
    double learning_rate = stod(argv[9]);

    Vector y_tr, y_test;
    Matrix X_tr = load_csv(train_input, y_tr);
    Matrix X_test = load_csv(validation_input, y_test);

    return {X_tr, y_tr, X_test, y_test, train_out, validation_out, metrics_out,
            num_epoch, hidden_units, init_flag, learning_rate};
}