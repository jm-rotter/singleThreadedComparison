#pragma once
#include "utils.h"

__global__ void matmul_kernel(const Matrix mat1, const Matrix mat2, Matrix res_mat);
