#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

constexpr double EULERS_NUMBER = 2.718281828;

struct MNISTSample {
    int label;
    std::vector<int> pixels;
};

void load_dataset(const std::string& path, std::vector<std::vector<int>>& x, std::vector<int>& y);
std::vector<MNISTSample> loadMNISTCSV(const std::string& filename);
void split_x_and_y(std::vector<MNISTSample>& dataset, std::vector<std::vector<int>>& x, std::vector<int>& y);
void ReLU(std::vector<double>& arr);
void softmax(std::vector<double>& arr);

int main() {
    const std::string train_csv = "./MNIST_CSV/mnist_train.csv";
    const std::string test_csv = "./MNIST_CSV/mnist_test.csv";

    std::vector<std::vector<int>> x_train, x_test;
    std::vector<int> y_train, y_test;

    std::cout << "Reading train dataset" << std::endl;
    load_dataset(train_csv, x_train, y_train);

    std::cout << "Reading test dataset" << std::endl;
    load_dataset(test_csv, x_test, y_test);

    return 0;
}

void load_dataset(const std::string& path, std::vector<std::vector<int>>& x, std::vector<int>& y) {
    std::vector<MNISTSample> dataset = loadMNISTCSV(path);
    split_x_and_y(dataset, x, y);
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
            sample.pixels.push_back(std::stoi(value));
        }

        dataset.push_back(sample);
    }

    return dataset;
}

void split_x_and_y(std::vector<MNISTSample>& dataset, std::vector<std::vector<int>>& x, std::vector<int>& y) {
    for (auto& s : dataset) {
        x.push_back(s.pixels);
        y.push_back(s.label);
    }
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

