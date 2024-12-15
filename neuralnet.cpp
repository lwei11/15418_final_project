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
      linear2(hidden_size, hidden_size, init_fn, lr),
      linear3(hidden_size, hidden_size, init_fn, lr),
      linear4(hidden_size, hidden_size, init_fn, lr),
      linear5(hidden_size, hidden_size, init_fn, lr),
      linear6(hidden_size, output_size, init_fn, lr),
      softmax() {}

std::tuple<Vector, double> NN::forward(const Vector& x, int y) {
    Vector a = linear1.forward(x);
    Vector z = sigmoid.forward(a);
    Vector b = linear2.forward(z);
    Vector l3 = linear3.forward(b);
    Vector l4 = linear4.forward(l3);
    Vector l5 = linear5.forward(l4);
    Vector l6 = linear6.forward(l5);
    return softmax.forward(l6, y);
}

Vector NN::forward_1(const Vector& x) {
    Vector a = linear1.forward(x);
    Vector z = sigmoid.forward(a);
    Vector b = linear2.forward(z);
    Vector l3 = linear3.forward(b);
    return l3;
}

std::tuple<Vector, double> NN::forward_2(const Vector& z, int y) {
    Vector l4 = linear4.forward(z);
    Vector l5 = linear5.forward(l4);
    Vector l6 = linear6.forward(l5);
    return softmax.forward(l6, y);
}

void NN::backward(int y, const Vector& y_hat) {
    Vector db = softmax.backward(y, y_hat);
    Vector l4 = linear6.backward(db);
    Vector l5 = linear5.backward(l4);
    Vector l6 = linear4.backward(l5);
    Vector l7 = linear3.backward(l6);
    Vector l8 = linear2.backward(l7);
    Vector da = sigmoid.backward(l8);
    linear1.backward(da);
}

Vector NN::backward_2(int y, const Vector& y_hat) {
    Vector db = softmax.backward(y, y_hat);
    Vector l4 = linear6.backward(db);
    Vector l5 = linear5.backward(l4);
    Vector l6 = linear4.backward(l5);
    return l6;
}

void NN::backward_1(const Vector& dz) {
    Vector l7 = linear3.backward(dz);
    Vector l8 = linear2.backward(l7);
    Vector da = sigmoid.backward(l8);
    linear1.backward(da);
}

void NN::step() {
    linear6.step();
    linear5.step();
    linear4.step();
    linear3.step();
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

        for (size_t i = 0; i < indices.size(); ++i) {
            X_shuffled[i] = X_train[indices[i]];
            y_shuffled[i] = y_train[indices[i]];
        }

        // Train on each data point
        for (int i = 0; i < batch_round; ++i) {
            int curr_start = i * batch_size + displacements[pid];
            int size = (curr_start + batch_size) >= counts[pid] + displacements[pid] ? displacements[pid] + counts[pid] - curr_start : batch_size;
            for (int j = 0; j < size; ++j) {
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
            
            //Aggregate gradients
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
        }

        // Compute training and test losses after the epoch
        double train_loss = compute_loss(X_shuffled, y_shuffled);
        double test_loss = compute_loss(X_test, y_test);

        train_losses.push_back(train_loss);
        test_losses.push_back(test_loss);
        
        //If we want to see per epoch stats
        // if (pid == 0) {
        //     std::cout << "Epoch " << (epoch + 1) << ": Train Loss = " << train_loss
        //             << ", Test Loss = " << test_loss << std::endl;
        // }
    }

    return {train_losses, test_losses};
}

std::tuple<std::vector<double>, std::vector<double>> NN::train(
    const Matrix& X_train, const Vector& y_train,
    const Matrix& X_test, const Vector& y_test,
    int epochs) {

    std::vector<double> train_losses;
    std::vector<double> test_losses;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle the training data for each epoch
        Matrix X_shuffled = X_train;
        Vector y_shuffled = y_train;

        // Shuffle data using a random permutation
        std::vector<size_t> indices(X_train.size());
        std::iota(indices.begin(), indices.end(), 0);  // Generate indices
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

        for (size_t i = 0; i < indices.size(); ++i) {
            X_shuffled[i] = X_train[indices[i]];
            y_shuffled[i] = y_train[indices[i]];
        }

        // Train on each data point
        for (size_t i = 0; i < X_shuffled.size(); ++i) {
            auto [y_hat, loss] = forward(X_shuffled[i], y_shuffled[i]);  // Forward pass
            backward(y_shuffled[i], y_hat);  // Backpropagation
            step();  // Update weights using gradients
        }

        // Compute training and test losses after the epoch
        double train_loss = compute_loss(X_shuffled, y_shuffled);
        double test_loss = compute_loss(X_test, y_test);

        train_losses.push_back(train_loss);
        test_losses.push_back(test_loss);
        
        //If we want to see per epoch stats
        // std::cout << "Epoch " << (epoch + 1) << ": Train Loss = " << train_loss
        //           << ", Test Loss = " << test_loss << std::endl;
    }

    return {train_losses, test_losses};
}

std::tuple<std::vector<double>, std::vector<double>> NN::train_model_2(
    const Matrix& X_train, const Vector& y_train,
    const Matrix& X_test, const Vector& y_test,
    int epochs, int batch_size, int pid) {

    std::vector<double> train_losses;
    std::vector<double> test_losses;

    Matrix X_shuffled = X_train;
    Vector y_shuffled = y_train;

    // Shuffle data using a random permutation
    int total_samples = X_train.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle the data
        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

        for (size_t i = 0; i < indices.size(); ++i) {
            X_shuffled[i] = X_train[indices[i]];
            y_shuffled[i] = y_train[indices[i]];
        }

        // Iterate over batches
        for (int start = 0; start < total_samples; start += batch_size) {
            int end = std::min(start + batch_size, total_samples);
            int current_batch_size = end - start;

            // Extract batch
            Matrix X_batch(current_batch_size);
            Vector y_batch(current_batch_size);
            for (int j = 0; j < current_batch_size; ++j) {
                X_batch[j] = X_shuffled[start + j];
                y_batch[j] = y_shuffled[start + j];
            }

            if (pid == 0) {
                // Forward at stage 1
                std::vector<Vector> A1_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A1_batch[b] = forward_1(X_batch[b]);
                }

                // Flatten A1_batch
                int A1_dim = (int)A1_batch[0].size();
                std::vector<double> A1_flat(current_batch_size * A1_dim);
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(A1_batch[b].begin(), A1_batch[b].end(),
                              A1_flat.begin() + b * A1_dim);
                }

                // Send batch metadata and flattened arrays to Rank 1
                MPI_Send(&current_batch_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                MPI_Send(&A1_dim, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                MPI_Send(A1_flat.data(), current_batch_size * A1_dim, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
                MPI_Send(y_batch.data(), current_batch_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

                // Receive y_hat results from Rank 1
                int y_hat_dim;
                MPI_Recv(&y_hat_dim, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<double> y_hat_flat(current_batch_size * y_hat_dim);
                MPI_Recv(y_hat_flat.data(), current_batch_size * y_hat_dim, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Unflatten y_hat_batch
                std::vector<Vector> y_hat_batch(current_batch_size, Vector(y_hat_dim));
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(y_hat_flat.begin() + b * y_hat_dim,
                              y_hat_flat.begin() + (b + 1) * y_hat_dim,
                              y_hat_batch[b].begin());
                }

                // Receive dA1 from Rank 1
                int dA1_dim;
                MPI_Recv(&dA1_dim, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<double> dA1_flat(current_batch_size * dA1_dim);
                MPI_Recv(dA1_flat.data(), current_batch_size * dA1_dim, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<Vector> dA1_batch(current_batch_size, Vector(dA1_dim));
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(dA1_flat.begin() + b * dA1_dim,
                              dA1_flat.begin() + (b + 1) * dA1_dim,
                              dA1_batch[b].begin());
                }

                // Backward at stage 0 for each sample, then step
                for (int b = 0; b < current_batch_size; ++b) {
                    backward_1(dA1_batch[b]);
                }
                step();

            } else if (pid == 1) {
                // Receive batch metadata and flattened arrays from Rank 0
                int current_batch_size;
                int A1_dim;
                MPI_Recv(&current_batch_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&A1_dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<double> A1_flat(current_batch_size * A1_dim);
                MPI_Recv(A1_flat.data(), current_batch_size * A1_dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<double> y_batch_recv(current_batch_size);
                MPI_Recv(y_batch_recv.data(), current_batch_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Unflatten A1_batch
                std::vector<Vector> A1_batch(current_batch_size, Vector(A1_dim));
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(A1_flat.begin() + b * A1_dim,
                              A1_flat.begin() + (b + 1) * A1_dim,
                              A1_batch[b].begin());
                }

                // Forward at stage 2
                std::vector<Vector> y_hat_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    auto [y_hat, loss] = forward_2(A1_batch[b], y_batch_recv[b]);
                    y_hat_batch[b] = y_hat;
                }

                int y_hat_dim = (int)y_hat_batch[0].size();
                // Flatten y_hat_batch
                std::vector<double> y_hat_flat(current_batch_size * y_hat_dim);
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(y_hat_batch[b].begin(), y_hat_batch[b].end(),
                              y_hat_flat.begin() + b * y_hat_dim);
                }

                // Send y_hat back to Rank 0
                MPI_Send(&y_hat_dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(y_hat_flat.data(), current_batch_size * y_hat_dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

                // Backward at stage 1
                std::vector<Vector> dA1_batch(current_batch_size, Vector(A1_dim));
                for (int b = 0; b < current_batch_size; ++b) {
                    dA1_batch[b] = backward_2(y_batch_recv[b], y_hat_batch[b]);
                }

                // Flatten dA1_batch
                std::vector<double> dA1_flat(current_batch_size * A1_dim);
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(dA1_batch[b].begin(), dA1_batch[b].end(),
                              dA1_flat.begin() + b * A1_dim);
                }

                // Send dA1 back to Rank 0
                MPI_Send(&A1_dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(dA1_flat.data(), current_batch_size * A1_dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
        // Compute training and test losses after the epoch
        double train_loss = compute_loss(X_shuffled, y_shuffled);
        double test_loss = compute_loss(X_test, y_test);

        train_losses.push_back(train_loss);
        test_losses.push_back(test_loss);
        
        //If we want to see per epoch stats
        // if (pid == 0) {
        //     std::cout << "Epoch " << (epoch + 1) << ": Train Loss = " << train_loss
        //             << ", Test Loss = " << test_loss << std::endl;
        // }
    }
    return {train_losses, test_losses};
}

std::tuple<std::vector<double>, std::vector<double>> NN::train_model_8(
    const Matrix& X_train, const Vector& y_train,
    const Matrix& X_test, const Vector& y_test,
    int epochs, int batch_size, int pid) 
{
    std::vector<double> train_losses;
    std::vector<double> test_losses;

    Matrix X_shuffled = X_train;
    Vector y_shuffled = y_train;
    int total_samples = (int)X_train.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));
        for (size_t i = 0; i < indices.size(); ++i) {
            X_shuffled[i] = X_train[indices[i]];
            y_shuffled[i] = y_train[indices[i]];
        }


        for (int start = 0; start < total_samples; start += batch_size) {
            int end = std::min(start + batch_size, total_samples);
            int current_batch_size = end - start;

            // Extract batch (on Rank 0)
            Matrix X_batch;
            Vector y_batch;
            if (pid == 0) {
                X_batch.resize(current_batch_size);
                y_batch.resize(current_batch_size);
                for (int j = 0; j < current_batch_size; ++j) {
                    X_batch[j] = X_shuffled[start + j];
                    y_batch[j] = y_shuffled[start + j];
                }
            }

            // Forward/Backward logic for each rank
            auto send_batch = [&](int dest, const std::vector<Vector>& A_batch, const Vector& y_batch) {
                int A_dim = (int)A_batch[0].size();
                // Flatten A_batch
                std::vector<double> A_flat(current_batch_size * A_dim);
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(A_batch[b].begin(), A_batch[b].end(), A_flat.begin() + b * A_dim);
                }
                MPI_Send(&current_batch_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(&A_dim, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(A_flat.data(), current_batch_size * A_dim, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                MPI_Send(y_batch.data(), current_batch_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            };

            auto recv_batch = [&](int src, std::vector<Vector>& A_batch, Vector& y_batch) {
                int current_batch_size_recv;
                int A_dim;
                MPI_Recv(&current_batch_size_recv, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&A_dim, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<double> A_flat(current_batch_size_recv * A_dim);
                MPI_Recv(A_flat.data(), current_batch_size_recv * A_dim, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<double> y_batch_recv(current_batch_size_recv);
                MPI_Recv(y_batch_recv.data(), current_batch_size_recv, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                A_batch.resize(current_batch_size_recv, Vector(A_dim));
                for (int b = 0; b < current_batch_size_recv; ++b) {
                    std::copy(A_flat.begin() + b * A_dim,
                              A_flat.begin() + (b + 1) * A_dim,
                              A_batch[b].begin());
                }
                y_batch = y_batch_recv;
            };

            //Send helper function
            auto send_grads = [&](int dest, const std::vector<Vector>& dA_batch) {
                int dA_dim = (int)dA_batch[0].size();
                std::vector<double> dA_flat(current_batch_size * dA_dim);
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(dA_batch[b].begin(), dA_batch[b].end(), dA_flat.begin() + b * dA_dim);
                }
                MPI_Send(&dA_dim, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(dA_flat.data(), current_batch_size * dA_dim, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            };

            //Recv helper function
            auto recv_grads = [&](int src, std::vector<Vector>& dA_batch) {
                int dA_dim;
                MPI_Recv(&dA_dim, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<double> dA_flat(current_batch_size * dA_dim);
                MPI_Recv(dA_flat.data(), current_batch_size * dA_dim, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                dA_batch.resize(current_batch_size, Vector(dA_dim));
                for (int b = 0; b < current_batch_size; ++b) {
                    std::copy(dA_flat.begin() + b * dA_dim,
                              dA_flat.begin() + (b + 1) * dA_dim,
                              dA_batch[b].begin());
                }
            };

            // Each rank does forward_i and backward_(9-i) following a pipeline format:
            // Forward direction: 0->1->2->3->4->5->6->7
            // Backward direction: 7->6->5->4->3->2->1->0

            if (pid == 0) {
                // Forward at stage 1
                std::vector<Vector> A1_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A1_batch[b] = linear1.forward(X_batch[b]);
                }

                // Send A1, y to Rank 1
                send_batch(1, A1_batch, y_batch);

                // Receive dA0 (gradients) from Rank 1
                std::vector<Vector> dA0_batch;
                recv_grads(1, dA0_batch);

                // Backward at stage 1
                for (int b = 0; b < current_batch_size; ++b) {
                    linear1.backward(dA0_batch[b]);
                }
                step();

            } else if (pid == 1) {
                // Receive A1,y from Rank 0
                std::vector<Vector> A1_batch; Vector y_batch;
                recv_batch(0, A1_batch, y_batch);

                // Forward at stage 2
                std::vector<Vector> A2_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A2_batch[b] = sigmoid.forward(A1_batch[b]);
                }

                // Send A2,y to Rank 2
                send_batch(2, A2_batch, y_batch);

                // Receive dA1 from Rank 2
                std::vector<Vector> dA1_batch;
                recv_grads(2, dA1_batch);

                // Backward at stage 2
                std::vector<Vector> dA0_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    dA0_batch[b] = sigmoid.backward(dA1_batch[b]);
                }

                // Send dA0 back to Rank 0
                send_grads(0, dA0_batch);

            } else if (pid == 2) {
                // Receive A2,y from Rank 1
                std::vector<Vector> A2_batch; Vector y_batch;
                recv_batch(1, A2_batch, y_batch);

                // Forward at stage 3
                std::vector<Vector> A3_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A3_batch[b] = linear2.forward(A2_batch[b]);
                }

                // Send A3,y to Rank 3
                send_batch(3, A3_batch, y_batch);

                // Receive dA2 from Rank 3
                std::vector<Vector> dA2_batch;
                recv_grads(3, dA2_batch);

                // Backward at stage 3
                std::vector<Vector> dA1_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    dA1_batch[b] = linear2.backward(dA2_batch[b]);
                }

                // Send dA1 to Rank 1
                send_grads(1, dA1_batch);

            } else if (pid == 3) {
                std::vector<Vector> A3_batch; Vector y_batch;
                recv_batch(2, A3_batch, y_batch);

                // Forward at stage 4
                std::vector<Vector> A4_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A4_batch[b] = linear3.forward(A3_batch[b]);
                }

                send_batch(4, A4_batch, y_batch);

                std::vector<Vector> dA3_batch;
                recv_grads(4, dA3_batch);

                // Backward at stage 4
                std::vector<Vector> dA2_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    dA2_batch[b] = linear3.backward(dA3_batch[b]);
                }

                send_grads(2, dA2_batch);

            } else if (pid == 4) {
                std::vector<Vector> A4_batch; Vector y_batch;
                recv_batch(3, A4_batch, y_batch);

                // Forward at stage 5
                std::vector<Vector> A5_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A5_batch[b] = linear4.forward(A4_batch[b]);
                }

                send_batch(5, A5_batch, y_batch);

                std::vector<Vector> dA4_batch;
                recv_grads(5, dA4_batch);

                // Backward at stage 5
                std::vector<Vector> dA3_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    dA3_batch[b] = linear4.backward(dA4_batch[b]);
                }

                send_grads(3, dA3_batch);

            } else if (pid == 5) {
                std::vector<Vector> A5_batch; Vector y_batch;
                recv_batch(4, A5_batch, y_batch);

                // Forward at stage 6
                std::vector<Vector> A6_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A6_batch[b] = linear5.forward(A5_batch[b]);
                }

                send_batch(6, A6_batch, y_batch);

                std::vector<Vector> dA5_batch;
                recv_grads(6, dA5_batch);

                // Backward at stage 6
                std::vector<Vector> dA4_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    dA4_batch[b] = linear5.backward(dA5_batch[b]);
                }

                send_grads(4, dA4_batch);

            } else if (pid == 6) {
                std::vector<Vector> A6_batch; Vector y_batch;
                recv_batch(5, A6_batch, y_batch);

                // Forward at stage 7
                std::vector<Vector> A7_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    A7_batch[b] = linear6.forward(A6_batch[b]);
                }

                send_batch(7, A7_batch, y_batch);

                std::vector<Vector> dA6_batch;
                recv_grads(7, dA6_batch);

                // Backward at stage 7
                std::vector<Vector> dA5_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    dA5_batch[b] = linear6.backward(dA6_batch[b]);
                }

                send_grads(5, dA5_batch);

            } else if (pid == 7) {
                // Receive A7,y from Rank 6
                std::vector<Vector> A7_batch; Vector y_batch;
                recv_batch(6, A7_batch, y_batch);

                // Forward at stage 8 produces y_hat
                std::vector<Vector> y_hat_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    auto [y_hat, loss] = softmax.forward(A7_batch[b], y_batch[b]);
                    y_hat_batch[b] = y_hat;
                }

                // Backward at final stage
                std::vector<Vector> dA7_batch(current_batch_size);
                for (int b = 0; b < current_batch_size; ++b) {
                    dA7_batch[b] = softmax.backward(y_batch[b], y_hat_batch[b]);
                }

                // Send dA7 back to Rank 6
                send_grads(6, dA7_batch);
            }
        }

        // Compute training and test losses after the epoch
        double train_loss = 0.0;
        double test_loss = 0.0;
        train_loss = compute_loss(X_shuffled, y_shuffled);
        test_loss = compute_loss(X_test, y_test);

        train_losses.push_back(train_loss);
        test_losses.push_back(test_loss);

        //If we want to see per epoch stats
        // if (pid == 0) {  
        //     std::cout << "Epoch " << (epoch + 1) << ": Train Loss = " << train_loss
        //               << ", Test Loss = " << test_loss << std::endl;
        // }
    }

    return {train_losses, test_losses};
}

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

        //auto [train_losses, val_losses] = nn.train_model_2(X_train, y_train, X_val, y_val, num_epochs, batch_size, pid);
        auto [train_losses, val_losses] = nn.train_model_8(X_train, y_train, X_val, y_val, num_epochs, batch_size, pid);
        //auto [train_losses, val_losses] = nn.train_data(X_train, y_train, X_val, y_val, num_epochs, batch_size, nproc, pid);
        //auto [train_losses, val_losses] = nn.train(X_train, y_train, X_val, y_val, num_epochs);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end - start;

        // Print the result
        if (pid == 0) {
            std::cout << "Elapsed time: " << elapsed_time.count() << " seconds\n";
        }

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
        
        if (pid == 0) {
            std::cout << "Training error: " << train_error << std::endl;
            std::cout << "Validation error: " << val_error << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}