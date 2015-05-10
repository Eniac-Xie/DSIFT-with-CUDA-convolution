#ifndef CONVGPU_H
#define CONVGPU_H

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS - 1)

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////

void convolutionRowsGPU(
	float *d_Dst,
	float *d_Src,
	float *kernel,
	int imageW,
	int imageH
	);

void convolutionColumnsGPU(
	float *d_Dst,
	float *d_Src,
	float *kernel,
	int imageW,
	int imageH
	);

void getDescr(
	float *resDescr, 
	float *d_Output, 
	int outHeight, 
	int outWidth, 
	int baseX, 
	int baseY, 
	int imWidth
	);

#endif
