#include "neuralnet.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <chrono>

#include <mpi.h>

// Helper Functions
Matrix zero_init(int rows, int cols) {
    return Matrix(rows, Vector(cols, 0.0));
}

Matrix random_init(int rows, int cols) {
    std::mt19937 gen(rows * cols);  // Seed with product of dimensions
    std::uniform_real_distribution<> dist(-0.1, 0.1);
    Matrix matrix(rows, Vector(cols));
    for (auto& row : matrix) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
    return matrix;
}

// SoftMaxCrossEntropy Implementation
Vector SoftMaxCrossEntropy::softmax(const Vector& z) {
    Vector exp_z(z.size());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        exp_z[i] = std::exp(z[i]);
        sum += exp_z[i];
    }
    for (auto& val : exp_z) {
        val /= sum;
    }
    return exp_z;
}

double SoftMaxCrossEntropy::cross_entropy(int y, const Vector& y_hat) {
    return -std::log(y_hat[y]);
}

std::tuple<Vector, double> SoftMaxCrossEntropy::forward(const Vector& z, int y) {
    Vector y_hat = softmax(z);
    double loss = cross_entropy(y, y_hat);
    return {y_hat, loss};
}

Vector SoftMaxCrossEntropy::backward(int y, const Vector& y_hat) {
    Vector grad = y_hat;
    grad[y] -= 1.0;
    return grad;
}

// Sigmoid Implementation
Vector Sigmoid::forward(const Vector& x) {
    z.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        z[i] = 1.0 / (1.0 + std::exp(-x[i]));
    }
    return z;
}

Vector Sigmoid::backward(const Vector& dz) {
    Vector grad(z.size());
    for (size_t i = 0; i < z.size(); ++i) {
        grad[i] = dz[i] * z[i] * (1.0 - z[i]);
    }
    return grad;
}

// Linear Implementation
Linear::Linear(int input_size, int output_size, Matrix (*init_fn)(int, int), double lr)
    : learning_rate(lr) {
    weights = init_fn(output_size, input_size + 1);  // Including bias
    gradients = zero_init(output_size, input_size + 1);
}

Vector Linear::forward(const Vector& x) {
    input_with_bias = x;
    input_with_bias.insert(input_with_bias.begin(), 1.0);  // Add bias term
    Vector output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            output[i] += weights[i][j] * input_with_bias[j];
        }
    }
    return output;
}

Vector Linear::backward(const Vector& dz) {
    for (size_t i = 0; i < gradients.size(); ++i) {
        for (size_t j = 0; j < gradients[i].size(); ++j) {
            gradients[i][j] = dz[i] * input_with_bias[j];
        }
    }
    Vector dx(weights[0].size() - 1, 0.0);  // Exclude bias
    for (size_t j = 1; j < weights[0].size(); ++j) {
        for (size_t i = 0; i < weights.size(); ++i) {
            dx[j - 1] += weights[i][j] * dz[i];
        }
    }
    return dx;
}

void Linear::step() {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learning_rate * gradients[i][j];
        }
    }
}

// Neural Network Implementation
NN::NN(int input_size, int hidden_size, int output_size, Matrix (*init_fn)(int, int), double lr)
    : linear1(input_size, hidden_size, init_fn, lr),
      sigmoid(),
      linear2(hidden_size, output_size, init_fn, lr),
      softmax() {}

std::tuple<Vector, double> NN::forward(const Vector& x, int y) {
    Vector a = linear1.forward(x);
    Vector z = sigmoid.forward(a);
    Vector b = linear2.forward(z);
    return softmax.forward(b, y);
    // Vector l1 = linear1.forward(x);
    // Vector z1 = sigmoid.forward(l1);
    // Vector l2 = linear2.forward(z1);
    // Vector z2 = sigmoid.forward(l2);
    // Vector l3 = linear3.forward(z2);
    // Vector z3 = sigmoid.forward(l3);
    // Vector l4 = linear4.forward(z3);
    // Vector z4 = sigmoid.forward(l4);
    // Vector l5 = linear5.forward(z4);
    // Vector z5 = sigmoid.forward(l5);
    // Vector l6 = linear6.forward(z5);
    // Vector z6 = sigmoid.forward(l6);
    // Vector l7 = linear7.forward(z6);
    // Vector z7 = sigmoid.forward(l7);
    // Vector l8 = linear8.forward(z7);
    // return softmax.forward(l8, y);
}

void NN::backward(int y, const Vector& y_hat) {
    Vector db = softmax.backward(y, y_hat);
    Vector dz = linear2.backward(db);
    Vector da = sigmoid.backward(dz);
    linear1.backward(da);
}

Vector NN::backward_1(int y, const Vector& y_hat) {
    Vector db = softmax.backward(y, y_hat);
    Vector dz = linear2.backward(db);
    return dz;
}

void NN::backward_2(const Vector& dz) {
    Vector da = sigmoid.backward(dz);
    linear1.backward(da);
}

void NN::step() {
    linear2.step();
    linear1.step();
}

double NN::compute_loss(const Matrix& X, const Vector& y) {
    double total_loss = 0.0;

    for (size_t i = 0; i < X.size(); ++i) {
        auto [_, loss] = forward(X[i], y[i]);  // Perform forward pass
        total_loss += loss;
    }

    return total_loss / X.size();  // Average loss
}

std::tuple<std::vector<double>, std::vector<double>> NN::train_data(
    const Matrix& X_train, const Vector& y_train,
    const Matrix& X_test, const Vector& y_test,
    int epochs, int batch_size, int nproc, int pid) {

    std::vector<double> train_losses;
    std::vector<double> test_losses;

    int total_size = X_train.size();
    std::vector<int> counts(nproc);
    std::vector<int> displacements(nproc);
    int rem = X_train.size() % nproc;
    int total_counts = 0;

    for (int k = 0; k < nproc; k++) {
        counts[k] = total_size / nproc;
        if (rem > 0) {
            counts[k]+=1;
            rem--;
        }
        displacements[k] = total_counts;
        total_counts += counts[k];
    }

    int batch_round = counts[0] / batch_size;
    batch_round = (counts[0] % batch_size == 0) ? batch_round : batch_round + 1;
    int rows1 = linear1.weights.size();
    int cols1 = linear1.weights[0].size();
    int rows2 = linear2.weights.size();
    int cols2 = linear2.weights[0].size();
    std::vector<double> send_buffer1(rows1 * cols1);
    std::vector<double> send_buffer2(rows2 * cols2);
    std::vector<double> rec_buffer1(rows1 * cols1);
    std::vector<double> rec_buffer2(rows2 * cols2);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle the training data for each epoch
        Matrix X_shuffled = X_train;
        Vector y_shuffled = y_train;

        // Shuffle data using a random permutation
        std::vector<size_t> indices(X_train.size());
        std::iota(indices.begin(), indices.end(), 0);  // Generate indices
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

        // inefficient
        for (size_t i = 0; i < indices.size(); ++i) {
            X_shuffled[i] = X_train[indices[i]];
            y_shuffled[i] = y_train[indices[i]];
        }

        // Train on each data point
        for (size_t i = 0; i < batch_round; ++i) {
            // printf("batch_round is %d \n", batch_round);
            int curr_start = i * batch_size + displacements[pid];
            int size = (curr_start + batch_size) >= counts[pid] + displacements[pid] ? displacements[pid] + counts[pid] - curr_start : batch_size;
            // if (pid == 0) {
            //     printf("Curr start is %d and expected end is %d, correct end is %d\n", curr_start, curr_start + batch_size, counts[pid] + displacements[pid]);
            // }
            // printf("Size is %d\n", size);
            // printf("Count for proc %d is %d\n", pid, counts[pid]);
            for (size_t j = 0; j < size; ++j) {
                // if (pid == 0) {
                //     printf("Curr idx is %d\n", j + curr_start);
                // }
                auto [y_hat, loss] = forward(X_shuffled[j + curr_start], 
                                            y_shuffled[j + curr_start]);  // Forward pass
                backward(y_shuffled[j + curr_start], y_hat);  // Backpropagation
                step();  // Update weights using gradients
            }

            for(int row_i = 0; row_i < rows1; row_i++) {
                for(int col_j = 0; col_j < cols1; col_j ++) {
                    send_buffer1[row_i * cols1 + col_j] = linear1.weights[row_i][col_j];
                }
            }
            for(int row_i = 0; row_i < rows2; row_i++) {
                for(int col_j = 0; col_j < cols2; col_j ++) {
                    send_buffer2[row_i * cols2 + col_j] = linear2.weights[row_i][col_j];
                }
            }

            // printf("----------------------- before MPI_Allreduce -----------------------------\n");
            //auto start = std::chrono::high_resolution_clock::now();

            MPI_Allreduce(send_buffer1.data(), rec_buffer1.data(), rows1 * cols1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(send_buffer2.data(), rec_buffer2.data(), rows2 * cols2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // if (pid == 0 && i == 1) {
            //     auto end = std::chrono::high_resolution_clock::now();
            //     std::chrono::duration<double> elapsed_time = end - start;

            //     // Print the result
            //     std::cout << "Elapsed time: " << elapsed_time.count() << " seconds\n";
            // }

            //After loop completes divide weights by nproc to get average
            for(int row_i = 0; row_i < rows1; row_i++) {
                for(int col_j = 0; col_j < cols1; col_j ++) {
                    linear1.weights[row_i][col_j] = rec_buffer1[row_i * cols1 + col_j] / nproc;
                }
            }
            for(int row_i = 0; row_i < rows2; row_i++) {
                for(int col_j = 0; col_j < cols2; col_j ++) {
                    linear2.weights[row_i][col_j] = rec_buffer2[row_i * cols2 + col_j] / nproc;
                }
            }
            // printf("----------------------- after MPI_Allreduce -----------------------------\n");
        }

        // Compute training and test losses after the epoch
        double train_loss = compute_loss(X_shuffled, y_shuffled);
        double test_loss = compute_loss(X_test, y_test);

        train_losses.push_back(train_loss);
        test_losses.push_back(test_loss);
        
        // std::cout << "Epoch " << (epoch + 1) << ": Train Loss = " << train_loss
        //           << ", Test Loss = " << test_loss << std::endl;
    }

    return {train_losses, test_losses};
}

std::tuple<std::vector<double>, std::vector<double>> NN::train(
    const Matrix& X_train, const Vector& y_train,
    const Matrix& X_test, const Vector& y_test,
    int epochs, int batch_size, int nproc, int pid) {

    std::vector<double> train_losses;
    std::vector<double> test_losses;

    int total_size = X_train.size();
    std::vector<int> counts(nproc);
    std::vector<int> displacements(nproc);
    int rem = X_train.size() % nproc;

    for (int k = 0; k < nproc; k++) {
        counts[k] = total_size / nproc;
        counts[k] = rem > k ? counts[k] + 1 : counts[k];
        displacements[k] = (k == 0) ? 0 : displacements[k-1] + counts[k-1];
    }

    int batch_round = counts[0] / batch_size;
    batch_round = (counts[0] % batch_size == 0) ? batch_round : batch_round + 1;
    int rows1 = linear1.weights.size();
    int cols1 = linear1.weights[0].size();
    int rows2 = linear2.weights.size();
    int cols2 = linear2.weights[0].size();
    std::vector<double> send_buffer1(rows1 * cols1);
    std::vector<double> send_buffer2(rows2 * cols2);
    std::vector<double> rec_buffer1(rows1 * cols1);
    std::vector<double> rec_buffer2(rows2 * cols2);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle the training data for each epoch
        Matrix X_shuffled = X_train;
        Vector y_shuffled = y_train;

        // Shuffle data using a random permutation
        std::vector<size_t> indices(X_train.size());
        std::iota(indices.begin(), indices.end(), 0);  // Generate indices
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

        // inefficient
        for (size_t i = 0; i < indices.size(); ++i) {
            X_shuffled[i] = X_train[indices[i]];
            y_shuffled[i] = y_train[indices[i]];
        }

        // Train on each data point
        for (size_t i = 0; i < batch_round; ++i) {
            // printf("batch_round is %d \n", batch_round);
            int curr_start = i * batch_size + displacements[pid];
            //printf("Curr start is %d and expected end is %d, correct end is %d\n", curr_start, curr_start + batch_size, counts[pid] + displacements[pid]);
            int size = (curr_start + batch_size) >= counts[pid] + displacements[pid] ? displacements[pid] + counts[pid] - curr_start : batch_size;
            // printf("Size is %d\n", size);
            // printf("Count for proc %d is %d\n", pid, counts[pid]);
            for (size_t j = 0; j < size; ++j) {
                auto [y_hat, loss] = forward(X_shuffled[j + curr_start], 
                                            y_shuffled[j + curr_start]);  // Forward pass
                backward(y_shuffled[j + curr_start], y_hat);  // Backpropagation
                step();  // Update weights using gradients
            }

            for(int row_i = 0; row_i < rows1; row_i++) {
                for(int col_j = 0; col_j < cols1; col_j ++) {
                    send_buffer1[row_i * cols1 + col_j] = linear1.weights[row_i][col_j];
                }
            }
            for(int row_i = 0; row_i < rows2; row_i++) {
                for(int col_j = 0; col_j < cols2; col_j ++) {
                    send_buffer2[row_i * cols2 + col_j] = linear2.weights[row_i][col_j];
                }
            }

            // printf("----------------------- before MPI_Allreduce -----------------------------\n");
            //RING METHOD
                //Within ring method add weights each time
            MPI_Allreduce(send_buffer1.data(), rec_buffer1.data(), rows1 * cols1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(send_buffer2.data(), rec_buffer2.data(), rows2 * cols2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            //After loop completes divide weights by nproc to get average
            for(int row_i = 0; row_i < rows1; row_i++) {
                for(int col_j = 0; col_j < cols1; col_j ++) {
                    linear1.weights[row_i][col_j] = rec_buffer1[row_i * cols1 + col_j] / nproc;
                }
            }
            for(int row_i = 0; row_i < rows2; row_i++) {
                for(int col_j = 0; col_j < cols2; col_j ++) {
                    linear2.weights[row_i][col_j] = rec_buffer2[row_i * cols2 + col_j] / nproc;
                }
            }
            // printf("----------------------- after MPI_Allreduce -----------------------------\n");
        }

        // Compute training and test losses after the epoch
        double train_loss = compute_loss(X_shuffled, y_shuffled);
        double test_loss = compute_loss(X_test, y_test);

        train_losses.push_back(train_loss);
        test_losses.push_back(test_loss);

        // std::cout << "Epoch " << (epoch + 1) << ": Train Loss = " << train_loss
        //           << ", Test Loss = " << test_loss << std::endl;
    }

    return {train_losses, test_losses};
}

// std::tuple<std::vector<double>, std::vector<double>> NN::train_model(
//     const Matrix& X_train, const Vector& y_train,
//     const Matrix& X_test, const Vector& y_test,
//     int epochs) {

//     std::vector<double> train_losses;
//     std::vector<double> test_losses;

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         // Shuffle the training data for each epoch
//         Matrix X_shuffled = X_train;
//         Vector y_shuffled = y_train;

//         // Shuffle data using a random permutation
//         std::vector<size_t> indices(X_train.size());
//         std::iota(indices.begin(), indices.end(), 0);  // Generate indices
//         std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

//         for (size_t i = 0; i < indices.size(); ++i) {
//             X_shuffled[i] = X_train[indices[i]];
//             y_shuffled[i] = y_train[indices[i]];
//         }

//         // Train on each data point
//         for (size_t i = 0; i < X_shuffled.size(); ++i) {
//             auto [y_hat, loss] = forward(X_shuffled[i], y_shuffled[i]);  // Forward pass
//             backward(y_shuffled[i], y_hat);  // Backpropagation
//             step();  // Update weights using gradients
//         }

//         // Compute training and test losses after the epoch
//         double train_loss = compute_loss(X_shuffled, y_shuffled);
//         double test_loss = compute_loss(X_test, y_test);

//         train_losses.push_back(train_loss);
//         test_losses.push_back(test_loss);

//         std::cout << "Epoch " << (epoch + 1) << ": Train Loss = " << train_loss
//                   << ", Test Loss = " << test_loss << std::endl;
//     }
// }

std::tuple<Vector, double> NN::test(const Matrix& X, const Vector& y) {
    Vector predictions;  // Store predicted labels
    int correct = 0;     // Count correct predictions

    for (size_t i = 0; i < X.size(); ++i) {
        auto [y_hat, _] = forward(X[i], y[i]);  // Perform forward pass
        int predicted_label = std::distance(y_hat.begin(), std::max_element(y_hat.begin(), y_hat.end()));
        predictions.push_back(predicted_label);

        // Check if prediction matches the true label
        if (predicted_label == y[i]) {
            ++correct;
        }
    }

    // Calculate error rate: (1 - correct/total)
    double error_rate = 1.0 - (static_cast<double>(correct) / X.size());

    return {predictions, error_rate};
}

// Helper function to load data from a CSV file
std::tuple<Matrix, Vector> load_csv(const std::string& file_path) {
    std::ifstream file(file_path);
    Matrix data;
    Vector labels;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        Vector row;

        // Extract the label (first column)
        if (std::getline(ss, value, ',')) {
            labels.push_back(std::stoi(value));
        }

        // Extract the features
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }

    return {data, labels};
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <train_input> <validation_input> <train_out> <validation_out>"
                     " <metrics_out> <num_epochs> <hidden_units> <init_flag> <learning_rate>"
                     " <nproc> <batch_size>"
                  << std::endl;
        return 1;
    }

    int pid;
    int nproc;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Status status;

    // printf("Curr proc is %d\n", pid);

    // Parse command-line arguments
    std::string train_input = argv[1];
    std::string validation_input = argv[2];
    std::string train_out = argv[3];
    std::string validation_out = argv[4];
    std::string metrics_out = argv[5];
    int num_epochs = std::stoi(argv[6]);
    int hidden_units = std::stoi(argv[7]);
    int init_flag = std::stoi(argv[8]);
    double learning_rate = std::stod(argv[9]);
    int batch_size = std::stod(argv[10]);

    try {
        // Load training and validation data
        auto [X_train, y_train] = load_csv(train_input);
        auto [X_val, y_val] = load_csv(validation_input);
        
        int count = 0;
        for (auto& temp : X_train) {
            count ++;
        }

        //printf("Num train = %d\n", count);

        // Initialize the neural network
        NN nn(
            X_train[0].size(),  // Input size (number of features)
            hidden_units,       // Hidden layer size
            10,                 // Output size (number of classes, e.g., 10 for digits)
            init_flag == 1 ? random_init : zero_init,  // Initialization function
            learning_rate       // Learning rate
        );

        // Train the network
        auto start = std::chrono::high_resolution_clock::now();

        auto [train_losses, val_losses] = nn.train_data(X_train, y_train, X_val, y_val, num_epochs, batch_size, nproc, pid, batch_size, nproc, pid);
        //auto [train_losses, val_losses] = nn.train(X_train, y_train, X_val, y_val, num_epochs);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end - start;

        // Print the result
        std::cout << "Elapsed time: " << elapsed_time.count() << " seconds\n";

        // Test the network on training and validation sets
        auto [train_predictions, train_error] = nn.test(X_train, y_train);
        auto [val_predictions, val_error] = nn.test(X_val, y_val);

        // Save predictions to output files
        std::ofstream train_out_file(train_out);
        for (const auto& pred : train_predictions) {
            train_out_file << pred << "\n";
        }

        std::ofstream val_out_file(validation_out);
        for (const auto& pred : val_predictions) {
            val_out_file << pred << "\n";
        }

        // Save metrics to output file
        std::ofstream metrics_file(metrics_out);
        for (size_t epoch = 0; epoch < train_losses.size(); ++epoch) {
            metrics_file << "epoch=" << (epoch + 1)
                         << " crossentropy(train): " << train_losses[epoch]
                         << "\n";
            metrics_file << "epoch=" << (epoch + 1)
                         << " crossentropy(validation): " << val_losses[epoch]
                         << "\n";
        }
        metrics_file << "error(train): " << train_error << "\n";
        metrics_file << "error(validation): " << val_error << "\n";

        std::cout << "Training error: " << train_error << std::endl;
        std::cout << "Validation error: " << val_error << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}