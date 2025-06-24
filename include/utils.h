#pragma once

#include <chrono>

struct Matrix {
	int width;
	int height;
	float* elements;
};


void fill_random(Matrix& matrix);
bool compare(const Matrix& mat1, const Matrix& mat2);
double time_in_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);
Matrix createMatrix(int width, int height);
