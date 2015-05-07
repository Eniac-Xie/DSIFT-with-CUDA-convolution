#ifndef DSIFT_CUDA_H
#define DSIFT_CUDA_H

#define TRANSPOSE         (0x1 << 2) /**< @brief Transpose result. */
#define PAD_BY_CONTINUITY (0x1 << 0) /**< @brief Pad by continuity. */
#define PAD_MASK          (0x3)      /**< @brief Padding field selector. */
#define PAD_BY_ZERO       (0x0 << 0) /**< @brief Pad with zeroes. */
#define MAX(x,y) (((x)>(y))?(x):(y))
#define EPSILON_F 1.19209290E-07F

#include <iostream>
#include <windows.h>
#include <fstream>
#include <cuda_runtime.h>
#include "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.5\common\inc\helper_functions.h"
#include "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.5\common\inc\helper_cuda.h"
#include "convGPU.h"
#include "fastMath.h"

using namespace std;
/* Dense SIFT keypoint */
typedef struct DsiftKeypoint_
{
	double x; /**< x coordinate */
	double y; /**< y coordinate */
	double s; /**< scale */
	double norm; /**< SIFT descriptor norm */
} DsiftKeypoint;

/* Dense SIFT descriptor geometry */
typedef struct DsiftDescriptorGeometry_
{
	int numBinT;  /**< number of orientation bins */
	int numBinX;  /**< number of bins along X */
	int numBinY;  /**< number of bins along Y */
	int binSizeX; /**< size of bins along X */
	int binSizeY; /**< size of bins along Y */
} DsiftDescriptorGeometry;

/*  Dense SIFT filter */
typedef struct DsiftFilter_
{
	int imWidth;            /**< @internal @brief image width */
	int imHeight;           /**< @internal @brief image height */

	int stepX;              /**< frame sampling step X */
	int stepY;              /**< frame sampling step Y */

	int boundMinX;          /**< frame bounding box min X */
	int boundMinY;          /**< frame bounding box min Y */
	int boundMaxX;          /**< frame bounding box max X */
	int boundMaxY;          /**< frame bounding box max Y */

	/** descriptor parameters */
	DsiftDescriptorGeometry geom;

	double windowSize;      /**< size of the Gaussian window */

	int numFrames;          /**< number of sampled frames */
	int descrSize;          /**< size of a descriptor */
	DsiftKeypoint *frames; /**< frame buffer */
	float *descrs;          /**< descriptor buffer */

	int numBinAlloc;        /**< buffer allocated: descriptor size */
	int numFrameAlloc;      /**< buffer allocated: number of frames  */
	int numGradAlloc;       /**< buffer allocated: number of orientations */

	float **grads;          /**< gradient buffer */
	float *convTmp1;        /**< temporary buffer */
	float *convTmp2;        /**< temporary buffer */
}  DsiftFilter;

int dsift_get_descriptor_size(DsiftFilter const *self)
{
	return self->descrSize;
};

DsiftFilter * init_dsift_filter(int imWidth, int imHeight, DsiftDescriptorGeometry* geom, int *step)
{
	DsiftFilter * self = new DsiftFilter;
	self->imWidth = imWidth;
	self->imHeight = imHeight;

	self->stepX = step[0];
	self->stepY = step[1];

	self->boundMinX = 0;
	self->boundMinY = 0;
	self->boundMaxX = imWidth - 1;
	self->boundMaxY = imHeight - 1;

	self->geom = *geom;

	self->windowSize = 2.0;

	self->convTmp1 = new float[self->imWidth * self->imHeight];
	self->convTmp2 = new float[self->imWidth * self->imHeight];

	self->numBinAlloc = 0;
	self->numFrameAlloc = 0;
	self->numGradAlloc = 0;

	self->descrSize = 0;
	self->numFrames = 0;
	self->grads = NULL;
	self->frames = NULL;
	self->descrs = NULL;

	int x1 = self->boundMinX;
	int x2 = self->boundMaxX;
	int y1 = self->boundMinY;
	int y2 = self->boundMaxY;

	int rangeX = x2 - x1 - (self->geom.numBinX - 1) * self->geom.binSizeX;
	int rangeY = y2 - y1 - (self->geom.numBinY - 1) * self->geom.binSizeY;

	int numFramesX = (rangeX >= 0) ? rangeX / self->stepX + 1 : 0;
	int numFramesY = (rangeY >= 0) ? rangeY / self->stepY + 1 : 0;

	self->numFrames = numFramesX * numFramesY;
	self->descrSize = self->geom.numBinT *
		self->geom.numBinX *
		self->geom.numBinY;

	return self;
};

DsiftDescriptorGeometry* init_dsift_geom(int numBinX, int numBinY, int numBinT, int binSizeX, int binSizeY)
{
	DsiftDescriptorGeometry *geom = new DsiftDescriptorGeometry;
	geom->numBinX = 4;
	geom->numBinY = 4;
	geom->numBinT = 8;
	geom->binSizeX = 8;
	geom->binSizeY = 8;
	return geom;
};

float at(int x, int y, IplImage *srcImage)
{
	uchar* ptr = (uchar*)(srcImage->imageData + y * srcImage->widthStep);
	return ptr[x] / 255.f;
}
void compute_grad(DsiftFilter * dsift, IplImage *srcImage)
{
	int imageHeight = dsift->imHeight;
	int imageWidth = dsift->imWidth;
	/* clear integral images */
	for (int t = 0; t < dsift->geom.numBinT; ++t)
		memset(dsift->grads[t], 0,
		sizeof(float)* dsift->imWidth * dsift->imHeight);

	/* Compute gradients, their norm, and their angle */
	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < imageWidth; x++)
		{
			float gx, gy;
			float angle, mod, nt, rbint;
			int bint;

			/* y derivative */
			if (y == 0) {
				gy = at(x, y + 1, srcImage) - at(x, y, srcImage);
			}
			else if (y == imageHeight - 1) {
				gy = at(x, y, srcImage) - at(x, y - 1, srcImage);
			}
			else {
				gy = 0.5F * (at(x, y + 1, srcImage) - at(x, y - 1, srcImage));
			}

			/* x derivative */
			if (x == 0) {
				gx = at(x + 1, y, srcImage) - at(x, y, srcImage);
			}
			else if (x == imageWidth - 1) {
				gx = at(x, y, srcImage) - at(x - 1, y, srcImage);
			}
			else {
				gx = 0.5F * (at(x + 1, y, srcImage) - at(x - 1, y, srcImage));
			}

			/* angle and modulus */
			angle = fast_atan2_f(gy, gx);
			mod = fast_sqrt_f(gx*gx + gy*gy);

			/* quantize angle */
			nt = mod_2pi_f(angle) * (dsift->geom.numBinT / (2 * DSIFT_PI));
			bint = (int)floor_f(nt);
			rbint = nt - bint;

			/* write it back, 8 direction per pixle*/
			dsift->grads[(bint) % dsift->geom.numBinT][x + y * dsift->imWidth] = (1 - rbint) * mod;
			dsift->grads[(bint + 1) % dsift->geom.numBinT][x + y * dsift->imWidth] = (rbint)* mod;
		}
	}
};

void dsift_alloc_buffers(DsiftFilter* self)
{
	int numFrameAlloc = self->numFrames;
	int numBinAlloc = self->descrSize;
	int numGradAlloc = self->geom.numBinT;

	self->frames = new DsiftKeypoint[numFrameAlloc];
	self->descrs = new float[numBinAlloc * numFrameAlloc];
	self->grads = new float*[numGradAlloc];
	for (int t = 0; t < numGradAlloc; ++t) {
		self->grads[t] = new float[self->imWidth * self->imHeight];
	}
};

float *
dsift_new_kernel(int binSize, int numBins, int binIndex, double windowSize)
{
	int filtLen = 2 * binSize - 1;
	float * ker = new float[filtLen];
	float * kerIter = ker;
	float delta = binSize * (binIndex - 0.5F * (numBins - 1));
	float sigma = (float)binSize * (float)windowSize;
	int x;

	for (x = -binSize + 1; x <= +binSize - 1; ++x) {
		float z = (x - delta) / sigma;
		*kerIter++ = (1.0F - fabsf(x) / binSize) *
			((binIndex >= 0) ? expf(-0.5F * z * z) : 1.0F);
	}
	return ker;
};

void imconvcol_vf(float* dst, int dst_stride,
	float const* src,
	int src_width, int src_height, int src_stride,
	float const* filt, int filt_begin, int filt_end,
	int step, unsigned int flags)
{
	int x = 0;
	int y;
	int dheight = (src_height - 1) / step + 1;
	int transp = flags & TRANSPOSE;
	int zeropad = (flags & PAD_MASK) == PAD_BY_ZERO;

	/* let filt point to the last sample of the filter */
	filt += filt_end - filt_begin;

	while (x < (signed)src_width) {
		/* Calculate dest[x,y] = sum_p image[x,p] filt[y - p]
		* where supp(filt) = [filt_begin, filt_end] = [fb,fe].
		*
		* CHUNK_A: y - fe <= p < 0
		*          completes VL_MAX(fe - y, 0) samples
		* CHUNK_B: VL_MAX(y - fe, 0) <= p < VL_MIN(y - fb, height - 1)
		*          completes fe - VL_MAX(fb, height - y) + 1 samples
		* CHUNK_C: completes all samples
		*/
		float const *filti;
		int stop;

		for (y = 0; y < (signed)src_height; y += step) {
			float acc = 0;
			float v = 0, c;
			float const* srci;

			filti = filt;
			stop = filt_end - y;
			srci = src + x - stop * src_stride;
			int res = srci - src;
			/*stop > 0, pick up values outside the image boundary*/
			if (stop > 0) {
				if (zeropad) {
					v = 0;
				}
				else {
					v = *(src + x);
				}
				while (filti > filt - stop) {
					c = *filti--;
					acc += v * c;
					srci += src_stride;
				}
			}

			stop = filt_end - MAX(filt_begin, y - (signed)src_height + 1) + 1;
			while (filti > filt - stop) {
				v = *srci;
				c = *filti--;
				acc += v * c;
				srci += src_stride;
			}

			if (zeropad) v = 0;

			stop = filt_end - filt_begin + 1;
			while (filti > filt - stop) {
				c = *filti--;
				acc += v * c;
			}

			if (transp) {
				*dst = acc; dst += 1;
			}
			else {
				*dst = acc; dst += dst_stride;
			}
		} /* next y */
		if (transp) {
			dst += 1 * dst_stride - dheight * 1;
		}
		else {
			dst += 1 * 1 - dheight * dst_stride;
		}
		x += 1;
	} /* next x */
};

void dsift_with_gaussian_window(DsiftFilter * self)
{
	int binx, biny, bint;
	int framex, framey;
	float *yker;

	int Wx = self->geom.binSizeX - 1;
	int Wy = self->geom.binSizeY - 1;
	int loop = 0;

	float *d_Buffer, *d_Output, *xkerGPU, *ykerGPU;
	float **d_Input, **xker;
	d_Input = new float *[self->geom.numBinT];
	xker = new float*[self->geom.numBinX];
	for (bint = 0; bint < self->geom.numBinT; bint++)
	{
		checkCudaErrors(cudaMalloc(&d_Input[bint], self->imWidth * self->imHeight * sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_Input[bint], self->grads[bint], self->imWidth * self->imHeight * sizeof(float), cudaMemcpyHostToDevice));
	}
	for (binx = 0; binx < self->geom.numBinX; ++binx)
	{
		xker[binx] = dsift_new_kernel(self->geom.binSizeX,
			self->geom.numBinX,
			binx,
			self->windowSize);
	}

	checkCudaErrors(cudaMalloc(&d_Output, self->imWidth * self->imHeight * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_Buffer, self->imWidth * self->imHeight * sizeof(float)));
	checkCudaErrors(cudaMalloc(&xkerGPU, (2 * self->geom.binSizeX - 1) * sizeof(float)));
	checkCudaErrors(cudaMalloc(&ykerGPU, (2 * self->geom.binSizeY - 1) * sizeof(float)));

	int frameSizeX = self->geom.binSizeX * (self->geom.numBinX - 1) + 1;
	int frameSizeY = self->geom.binSizeY * (self->geom.numBinY - 1) + 1;
	int descrSize = dsift_get_descriptor_size(self);
	float * resDescr;
	checkCudaErrors(cudaMalloc(&resDescr, (self->boundMaxY - frameSizeY + 2) * (self->boundMaxX - frameSizeX + 2) * sizeof(float)));


	for (biny = 0; biny < self->geom.numBinY; ++biny) {

		yker = dsift_new_kernel(self->geom.binSizeY,
			self->geom.numBinY,
			biny,
			self->windowSize);
		checkCudaErrors(cudaMemcpy(ykerGPU, yker, (2 * self->geom.binSizeY - 1) * sizeof(float), cudaMemcpyHostToDevice));
		for (binx = 0; binx < self->geom.numBinX; ++binx) {
			checkCudaErrors(cudaMemcpy(xkerGPU, xker[binx], (2 * self->geom.binSizeX - 1) * sizeof(float), cudaMemcpyHostToDevice));
			for (bint = 0; bint < self->geom.numBinT; ++bint) {

				//LARGE_INTEGER t1, t2, t3, tc;
				//QueryPerformanceFrequency(&tc);
				//QueryPerformanceCounter(&t1);
				convolutionRowsGPU(
					d_Buffer,
					d_Input[bint],
					xkerGPU,
					self->imWidth,
					self->imHeight
					);
				convolutionColumnsGPU(
					d_Output,
					d_Buffer,
					ykerGPU,
					self->imWidth,
					self->imHeight
					);
				cudaDeviceSynchronize();

				//QueryPerformanceCounter(&t2);
				//printf("convolution Time:%f\n", (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);

				getDescr( resDescr, 
					d_Output, 
					self->boundMaxY - frameSizeY + 2, 
					self->boundMaxX - frameSizeX + 2,
					binx * self->geom.binSizeX,
					biny * self->geom.binSizeY,
					self->imWidth);
				float *dst = self->descrs + (bint + binx * self->geom.numBinT + biny * (self->geom.numBinX * self->geom.numBinT)) * (self->boundMaxY - frameSizeY + 2) * (self->boundMaxX - frameSizeX + 2);
				checkCudaErrors(cudaMemcpy(dst, resDescr, (self->boundMaxY - frameSizeY + 2) * (self->boundMaxX - frameSizeX + 2) * sizeof(float), cudaMemcpyDeviceToHost));
				//QueryPerformanceCounter(&t3);
				//printf("Copy Time:%f\n", (t3.QuadPart - t2.QuadPart)*1.0 / tc.QuadPart);
			} /* for bint */
			
		} /* for binx */
		delete[]yker;
	} /* for biny */
	delete[]xker;
	for (bint = 0; bint < self->geom.numBinT; bint++)
	{
		cudaFree(d_Input[bint]);
	}
	cudaFree(d_Buffer);
	cudaFree(d_Output);
	cudaFree(xkerGPU);
	cudaFree(ykerGPU);
	cudaFree(resDescr);
};

inline float dsift_normalize_histogram(float * begin, float * end)
{
	float * iter;
	float  norm = 0.0F;

	for (iter = begin; iter < end; ++iter) {
		norm += (*iter) * (*iter);
	}
	norm = fast_sqrt_f(norm) + EPSILON_F;

	for (iter = begin; iter < end; ++iter) {
		*iter /= norm;
	}
	return norm;
};

inline void
dsift_transpose_descriptor(float* dst,
float const* src,
int numBinT,
int numBinX,
int numBinY)
{
	int t, x, y;

	for (y = 0; y < numBinY; ++y) {
		for (x = 0; x < numBinX; ++x) {
			int offset = numBinT * (x + y * numBinX);
			int offsetT = numBinT * (y + x * numBinY);

			for (t = 0; t < numBinT; ++t) {
				int tT = numBinT / 4 - t;
				dst[offsetT + (tT + numBinT) % numBinT] = src[offset + t];
			}
		}
	}
};

#endif