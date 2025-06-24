#include "cpu_matmul.h"
#include "gpu_matmul.h"
#include <cuda_runtime.h>
#include "utils.h"
#include <iostream>
#include <vector>
#include <chrono>


int main() {

	int a, b, c;

	a = 1024;
	b = 1024;
	c = 1024;

	std::cout << "Matrix Multiplicatoin Benchmark with 1024 x 1024 Matrices \n";

	Matrix mat1 = createMatrix(a, b);
	Matrix mat2 = createMatrix(b, c);
	Matrix gpuSingleRes = createMatrix(a, c);
	Matrix cpuSingleRes = createMatrix(a, c);

	fill_random(mat1);
	fill_random(mat2);

	auto t1 = std::chrono::high_resolution_clock::now();
	cpu_matmul(mat1, mat2, cpuSingleRes);
	auto t2 = std::chrono::high_resolution_clock::now();
	double cpuTime = time_in_ms(t1, t2);
	std::cout << "CPU Time: " << cpuTime << " ms\n";



	t1 = std::chrono::high_resolution_clock::now();
	matmul_kernel<<<1, 1>>> (mat1,mat2, gpuSingleRes);
	cudaDeviceSynchronize();
	t2 = std::chrono::high_resolution_clock::now();
	double gpuSingleTime = time_in_ms(t1, t2);
	std::cout << "GPU Single Time: " << gpuSingleTime << " ms\n";

	std::cout << "Comparison: " << (compare(gpuSingleRes, cpuSingleRes) ? "CPU result = GPU Naive result" : "CPU result != GPU Naive Result") << "\n";


	std::cout << "Speedup CPU -> GPU: " << cpuSingleTime / gpuSingleRes << "\n";

}
