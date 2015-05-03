#include <cuda.h> 
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.5\common\inc\helper_cuda.h"
#include "convGPU.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define   ROWS_RESULT_STEPS 32 
#define   ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(
	float *d_Dst,
	float *d_Src,
	float *kernel,
	int imageW,
	int imageH,
	int pitch
	)
{
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	//Load main data
#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
	}

	//Load left halo
#pragma unroll

	for (int i = 0; i < ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Load right halo
#pragma unroll

	for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Compute and store results
	__syncthreads();
#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		float sum = 0;

#pragma unroll

		for (int j = -KERNEL_RADIUS + 2; j <= KERNEL_RADIUS; j++)
		{
			sum += kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
		}

		d_Dst[i * ROWS_BLOCKDIM_X] = sum;
	}
}

void convolutionRowsGPU(
	float *d_Dst,
	float *d_Src,
	float *kernel,
	int imageW,
	int imageH
	)
{
	assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
	//assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
	//assert(imageH % ROWS_BLOCKDIM_Y == 0);

	dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
	//cout << "blocks" << blocks.x << "\t" << blocks.y << "\t" << blocks.z << endl;
	//cout << threads.x << "\t" << threads.y << "\t" << threads.z << endl;
	convolutionRowsKernel << <blocks, threads >> >(
		d_Dst,
		d_Src,
		kernel,
		imageW,
		imageH,
		imageW
		);
	getLastCudaError("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(
	float *d_Dst,
	float *d_Src,
	float *kernel,
	int imageW,
	int imageH,
	int pitch
	)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	//Main data
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
	}

	//Upper halo
#pragma unroll

	for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Lower halo
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Compute and store results
	__syncthreads();
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		float sum = 0;
#pragma unroll

		for (int j = -KERNEL_RADIUS + 2; j <= KERNEL_RADIUS; j++)
		{
			sum += kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
		}

		d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
	}
}

void convolutionColumnsGPU(
	float *d_Dst,
	float *d_Src,
	float *kernel,
	int imageW,
	int imageH
	)
{
	assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
	//assert(imageW % COLUMNS_BLOCKDIM_X == 0);
	//assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

	dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
	dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	convolutionColumnsKernel << <blocks, threads >> >(
		d_Dst,
		d_Src,
		kernel,
		imageW,
		imageH,
		imageW
		);
	getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

__global__ void getDescrKernel(float *resDescr, float *d_Output, int baseX, int baseY, int imWidth)
{
	*(resDescr + threadIdx.y + blockIdx.y * 32 + (threadIdx.x + blockIdx.x * 32) * imWidth) = d_Output[(threadIdx.y + blockIdx.y * 32 + baseX) * 1 +
		(threadIdx.x + blockIdx.x * 32 + baseY) * imWidth];
}

void getDescr(float *resDescr, float *d_Output, int outHeight, int outWidth, int baseX, int baseY, int imWidth)
{
	dim3 blocks(outHeight / 32, outWidth / 32);
	dim3 threads(32, 32);
	getDescrKernel << <blocks, threads >> >(resDescr, d_Output, baseX, baseY, imWidth);
	getLastCudaError("convolutionRowsKernel() execution failed\n");
}

