// #include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include "neuralnet.h"  // Header file for neural network implementation

constexpr double TOLERANCE = 1e-4;

// Helper function for assert_allclose
bool assert_allclose(const std::vector<double>& a, const std::vector<double>& b, double tol = TOLERANCE) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// Helper function for 2D vector comparison
bool assert_allclose_2d(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, double tol = TOLERANCE) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!assert_allclose(a[i], b[i], tol)) return false;
    }
    return true;
}

// Test case for random initialization
class TestRandomInit : public ::testing::Test {
protected:
    RandomInit random_init;
};

TEST_F(TestRandomInit, TestShape) {
    auto matrix = random_init.initialize(10, 5);
    ASSERT_EQ(matrix.size(), 10);
    ASSERT_EQ(matrix[0].size(), 5);
}

TEST_F(TestRandomInit, TestBounds) {
    auto matrix = random_init.initialize(5, 7);
    for (const auto& row : matrix) {
        for (double val : row) {
            ASSERT_GE(val, -0.1);
            ASSERT_LE(val, 0.1);
        }
    }
}

TEST_F(TestRandomInit, TestVariationRow) {
    auto matrix = random_init.initialize(20, 1);
    std::unordered_set<double> unique_values;
    for (const auto& row : matrix) {
        unique_values.insert(row[0]);
    }
    ASSERT_GT(unique_values.size(), 1);
}

TEST_F(TestRandomInit, TestVariationColumn) {
    auto matrix = random_init.initialize(1, 20);
    std::unordered_set<double> unique_values(matrix[0].begin(), matrix[0].end());
    ASSERT_GT(unique_values.size(), 1);
}

// Test case for the Linear layer
class TestLinear : public ::testing::Test {
protected:
    Linear layer;

    void SetUp() override {
        layer = Linear(5, 10, ZeroInit(), 0.0);  // Replace with appropriate setup
    }
};

TEST_F(TestLinear, TestForward) {
    std::vector<std::vector<double>> X = {...}; // Input
    std::vector<double> w = {...};             // Weights
    std::vector<double> solution = {...};      // Expected output

    layer.set_weights(w);
    auto result = layer.forward(X);
    ASSERT_TRUE(assert_allclose(result, solution));
}

TEST_F(TestLinear, TestBiasZeroInit) {
    Linear zero_linear(5, 10, ZeroInit(), 1.0);
    std::vector<double> x(5, 0.0);
    auto result = zero_linear.forward(x);
    ASSERT_EQ(std::count_if(result.begin(), result.end(), [](double v) { return v != 0; }), 0);
}

TEST_F(TestLinear, TestBiasRandomInit) {
    Linear random_linear(5, 10, RandomInit(), 1.0);
    std::vector<double> x(5, 0.0);
    auto result = random_linear.forward(x);
    ASSERT_EQ(std::count_if(result.begin(), result.end(), [](double v) { return v != 0; }), 0);
}

TEST_F(TestLinear, TestBackward) {
    std::vector<std::vector<double>> X = {...};
    std::vector<double> w = {...};
    std::vector<double> dx_solution = {...};
    std::vector<double> dw_solution = {...};

    layer.set_weights(w);
    auto z = layer.forward(X);
    std::vector<double> dz(z.size(), 1.0);
    auto dx = layer.backward(dz);
    auto dw = layer.get_dw();
    ASSERT_TRUE(assert_allclose(dx, dx_solution));
    ASSERT_TRUE(assert_allclose(dw, dw_solution));
}

// Test case for the Sigmoid layer
class TestSigmoid : public ::testing::Test {};

TEST_F(TestSigmoid, TestForward1) {
    std::vector<double> a = {...};
    std::vector<double> solution = {...};

    Sigmoid sigmoid;
    auto z = sigmoid.forward(a);
    ASSERT_TRUE(assert_allclose(z, solution));
}

TEST_F(TestSigmoid, TestBackward) {
    std::vector<double> z = {...};
    std::vector<double> solution = {...};

    Sigmoid sigmoid;
    sigmoid.forward(z);  // Cache forward result
    auto dz = sigmoid.backward(1.0);
    ASSERT_TRUE(assert_allclose(dz, solution));
}

// Test case for SoftMaxCrossEntropy
class TestSoftmax : public ::testing::Test {};

TEST_F(TestSoftmax, TestSoftmaxForward) {
    std::vector<double> z = {...};
    std::vector<double> solution = {...};

    SoftMaxCrossEntropy softmax;
    auto yh = softmax._softmax(z);
    ASSERT_TRUE(assert_allclose(yh, solution));
}

TEST_F(TestSoftmax, TestCrossEntropy) {
    std::vector<double> yh = {...};
    std::vector<double> y = {...};
    double solution = ...;

    SoftMaxCrossEntropy softmax;
    auto loss = softmax._cross_entropy(y, yh);
    ASSERT_NEAR(loss, solution, TOLERANCE);
}

TEST_F(TestSoftmax, TestBackward) {
    std::vector<double> y = {...};
    std::vector<double> yh = {...};
    std::vector<double> solution = {...};

    SoftMaxCrossEntropy softmax;
    auto db = softmax.backward(y, yh);
    ASSERT_TRUE(assert_allclose(db, solution));
}

// Test case for the full neural network (NN)
class TestNN : public ::testing::Test {
protected:
    NN nn;

    void SetUp() override {
        nn = NN(5, 4, 10, 1.0, ZeroInit());  // Replace with appropriate setup
    }
};

TEST_F(TestNN, TestForward) {
    std::vector<double> x = {...};
    int y = ...;
    std::vector<double> solution = {...};

    auto [yh, loss] = nn.forward(x, y);
    ASSERT_TRUE(assert_allclose(yh, solution));
}

TEST_F(TestNN, TestBackward) {
    std::vector<double> x = {...};
    int y = ...;

    nn.forward(x, y);  // Forward pass
    nn.backward(y, nn.get_yh());  // Backward pass

    std::vector<std::vector<double>> d_w1 = {...};  // Expected gradients
    std::vector<std::vector<double>> d_w2 = {...};

    ASSERT_TRUE(assert_allclose_2d(nn.get_linear1_dw(), d_w1));
    ASSERT_TRUE(assert_allclose_2d(nn.get_linear2_dw(), d_w2));
}
