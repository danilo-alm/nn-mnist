#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

constexpr double EULERS_NUMBER = 2.718281828;

struct MNISTSample {
    int label;
    std::vector<double> pixels;
};

void ReLU(std::vector<double>& arr);
void softmax(std::vector<double>& arr);
std::vector<MNISTSample> loadMNISTCSV(const std::string& filename);

int main() {
    std::cout << "Reading train dataset" << std::endl;
    std::vector<MNISTSample> train = loadMNISTCSV("./MNIST_CSV/mnist_train.csv");

    std::cout << "Reading test dataset" << std::endl;
    std::vector<MNISTSample> test = loadMNISTCSV("./MNIST_CSV/mnist_test.csv");

    return 0;
}

void ReLU(std::vector<double>& arr) {
    for (auto& val : arr) {
        val = std::max(0.0, val);
    }
}

void softmax(std::vector<double>& arr) {
    double sum = 0.0;
    for (auto& val : arr) {
        val = std::pow(EULERS_NUMBER, val);
        sum += val;
    }
    for (auto& val : arr) {
        val /= sum;
    }
}

std::vector<MNISTSample> loadMNISTCSV(const std::string& filename) {
    std::vector<MNISTSample> dataset;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;

        MNISTSample sample;

        std::getline(ss, value, ',');
        sample.label = std::stoi(value);

        while (std::getline(ss, value, ',')) {
            sample.pixels.push_back(std::stod(value));
        }

        dataset.push_back(sample);
    }

    return dataset;
}

