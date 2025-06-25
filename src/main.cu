#include "cpu_matmul.h"
#include "gpu_matmul.h"
#include <cuda_runtime.h>
#include "utils.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>


void computeBenchmark(int a, int b, int c, std::ofstream& outfile, bool verbose) {

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

	Matrix d_mat1, d_mat2, d_res_mat;
    d_mat1.width = mat1.width; d_mat1.height = mat1.height; 
    d_mat2.width = mat2.width; d_mat2.height = mat2.height;
    d_res_mat.width = gpuSingleRes.width; d_res_mat.height = gpuSingleRes.height;


	cudaMalloc(&d_mat1.elements, mat1.width * mat1.height * sizeof(float));
	cudaMalloc(&d_mat2.elements, mat2.width * mat2.height * sizeof(float));
	cudaMalloc(&d_res_mat.elements, gpuSingleRes.width * gpuSingleRes.height * sizeof(float));

	cudaMemcpy(d_mat1.elements, mat1.elements, mat1.width * mat1.height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat2.elements, mat2.elements, mat2.width * mat2.height* sizeof(float), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	t1 = std::chrono::high_resolution_clock::now();
	matmul_kernel<<<1, 1>>> (d_mat1,d_mat2, d_res_mat);
	cudaDeviceSynchronize();
	t2 = std::chrono::high_resolution_clock::now();


	cudaMemcpy(gpuSingleRes.elements, d_res_mat.elements, gpuSingleRes.width * gpuSingleRes.height * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_mat1.elements);
	cudaFree(d_mat2.elements);
	cudaFree(d_res_mat.elements);

	double gpuSingleTime = time_in_ms(t1, t2);

	if(verbose) {
		outfile << cpuTime << "," << gpuSingleTime << "," << gpuSingleTime / cpuTime << "," << a << "," << compare(gpuSingleRes, cpuSingleRes) << "\n";	
	}
}

int main() {
	std::ofstream outfile("results.csv");
	
	if(!outfile) {
		std::cerr << "Error could not open results.csv\n";
		return 1;
	}
	outfile << "cpuTime,gpuTime,cpuSpeedUp,matrixSize,compare\n";

	int a, b, c;
	a = b = c = 128;

	for(int i = 32; i <= 1024; i *=2){
		a = b = c = i;
		for(int j = 0; j < 13; j++) {
			j < 3 ? computeBenchmark(a,  b, c, outfile, false) : computeBenchmark(a, b, c, outfile, true);
		}
	}

}
