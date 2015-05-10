#include <iostream>
#include <fstream>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <cuda.h> 
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "DSIFT_CUDA.cuh"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "No image input or too many image input. " << endl;
		exit(-1);
	}

	/* parameter of image */
	IplImage *srcImage = NULL;
	char *srcPictureName = argv[1];
	int imageWidth = 0, imageHeight = 0;

	/* parameter of descripter */
	int numBinX = 4, numBinY = 4, numBinT = 8, binSizeX = 8, binSizeY = 8;

	/* sample step of X and Y */
	int step[2] = { 1, 1 };

	srcImage = cvLoadImage(srcPictureName, 1);
	if (srcImage == NULL)
	{
		cout << "Image not found" << endl;
		exit(-2);
	}
	imageWidth = srcImage->width;
	imageHeight = srcImage->height;

	/* program use gray scale image only */
	IplImage * grayImage = cvCreateImage(cvSize(srcImage->width, srcImage->height), srcImage->depth, 1);
	cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);

	
	LARGE_INTEGER t1, t3, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	/*initialize Dsift Filter*/
	DsiftDescriptorGeometry* geom = init_dsift_geom(numBinX, numBinY, numBinT, binSizeX, binSizeY);
	DsiftFilter* self = init_dsift_filter(imageWidth, imageHeight, geom, step);
	dsift_alloc_buffers(self);
	float *srcGPU, *destGPU;
	checkCudaErrors(cudaMalloc(&srcGPU, sizeof(float)* self->numFrames * self->descrSize));
	checkCudaErrors(cudaMalloc(&destGPU, sizeof(float)* self->numFrames * self->descrSize));
	compute_grad(self, grayImage);
	
	/* do dense sift, dsift descripter saved in srcGPU*/
	dsift_with_gaussian_window(self, srcGPU);

	dim3 threads(self->descrSize, 4);
	dim3 blocks(64, 64, self->numFrames / 16384);
	reverse << <blocks, threads >> >(srcGPU, destGPU, self->numFrames, self->descrSize);
	checkCudaErrors(cudaMemcpy(self->descrs, destGPU, sizeof(float)* self->numFrames * self->descrSize, cudaMemcpyDeviceToHost));

	int framex, framey, bint;
	int frameSizeX = self->geom.binSizeX * (self->geom.numBinX - 1) + 1;
	int frameSizeY = self->geom.binSizeY * (self->geom.numBinY - 1) + 1;
	int descrSize = dsift_get_descriptor_size(self);
	float * descrIter = self->descrs;
	float deltaCenterX = 0.5F * self->geom.binSizeX * (self->geom.numBinX - 1);
	float deltaCenterY = 0.5F * self->geom.binSizeY * (self->geom.numBinY - 1);

	DsiftKeypoint* frameIter = self->frames;
	for (framey = self->boundMinY;
		framey <= self->boundMaxY - frameSizeY + 1;
		framey += self->stepY) 
	{
		for (framex = self->boundMinX;
			framex <= self->boundMaxX - frameSizeX + 1;
			framex += self->stepX) 
		{
			frameIter->x = framex + deltaCenterX;
			frameIter->y = framey + deltaCenterY;
			frameIter++;
		} /* for framex */
	} /* for framey */

	/*use openMP to accelerate normalize*/
#pragma omp parallel for
	for (int i = 0; i < self->numFrames; i++)
	{
		/* L2 normalize */
		dsift_normalize_histogram(descrIter, descrIter + descrSize);

		/* clamp */
		for (bint = 0; bint < descrSize; ++bint)
		if (descrIter[bint] > 0.2F) descrIter[bint] = 0.2F;

		/* L2 normalize */
		dsift_normalize_histogram(descrIter, descrIter + descrSize);

		descrIter += descrSize;
	}

	QueryPerformanceCounter(&t3);
	cout << "Use Time: " << (t3.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << endl;

	ofstream out("descriptor.txt");
	ofstream outFrames("frames.txt");

	DsiftKeypoint const *frames = self->frames;
	for (int i = 0; i < self->numFrames; i++)
	{
		outFrames << frames[i].y << "\t" << frames[i].x << "\t";
		outFrames << endl;
		float *tmpDescr = self->descrs + descrSize * i;
		for (int j = 0; j < descrSize; ++j) 
		{
			unsigned char res = (unsigned char)(512.0F * tmpDescr[j] < 255.0F ? (512.0F * tmpDescr[j]) : 255.0F);
			out << (unsigned int)res << "\t";
		}
		out << endl;
	}
	out.close();
	outFrames.close();

	cvNamedWindow("srcImage", 0);
	cvShowImage("srcImage", srcImage);
	cvWaitKey(0);
	return 0;
}