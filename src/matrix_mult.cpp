#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <string>
#include <sstream>
#include <cstdlib>

std::vector<std::vector<double>> readMatrix(const std::string& filename, int& size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue; 
        std::vector<double> row;
        std::stringstream ss(line);
        double val;
        while (ss >> val) {
            row.push_back(val);
        }
        matrix.push_back(row);
    }

    if (matrix.empty()) {
        std::cerr << "Файл пуст: " << filename << std::endl;
        exit(1);
    }

    size = matrix.size();
    return matrix;
}

void writeMatrix(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось создать файл: " << filename << std::endl;
        exit(1);
    }
    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j] << (j == row.size() - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    // Значения по умолчанию
    int N = 1000;
    int num_threads = omp_get_max_threads(); 

    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) num_threads = std::atoi(argv[2]);

    omp_set_num_threads(num_threads);

    std::string fileA = "data/matrix_A_" + std::to_string(N) + ".txt";
    std::string fileB = "data/matrix_B_" + std::to_string(N) + ".txt";
    std::string fileC = "results/matrix_C_" + std::to_string(N) + "_" + std::to_string(num_threads) + ".txt";

    int N_A = 0, N_B = 0;
    auto A = readMatrix(fileA, N_A);
    auto B = readMatrix(fileB, N_B);

    if (N_A != N_B) {
        std::cerr << "Размеры матриц не совпадают! A: " << N_A << ", B: " << N_B << std::endl;
        return 1;
    }

    N = N_A;

    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    double start_time = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a_ik = A[i][k];
            for (int j = 0; j < N; ++j) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }

    double end_time = omp_get_wtime();
    double duration = end_time - start_time;

    long long operations = (long long)N * N * N;
    std::cout << N << "," << num_threads << "," << duration << "," << operations << std::endl;

    writeMatrix(fileC, C);

    return 0;
}