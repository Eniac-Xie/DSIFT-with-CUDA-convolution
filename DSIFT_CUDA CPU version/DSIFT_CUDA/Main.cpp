#include <iostream>
#include <string>
#include <fstream>
#include <windows.h>
#include <Mmsystem.h>
#include <opencv2/opencv.hpp>
#include "fastMath.h"
#include "DSIFT_CUDA.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "error parameter. program take only one parameter." << endl;
		exit(-1);
	}
	IplImage *srcImage = 0;
	char *srcPictureName = argv[1];
	int imageWidth = 0, imageHeight = 0;

	/* parameter of descripter */
	int numBinX = 4, numBinY = 4, numBinT = 8, binSizeX = 8, binSizeY = 8;

	/* step of X and Y */
	int step[2] = { 1, 1 };

	srcImage = cvLoadImage(srcPictureName, 1); 
	if (srcImage == NULL)
	{
		cout << "image: " << srcPictureName << "not found" << endl;
		exit(-2);
	}
	imageWidth = srcImage->width;
	imageHeight = srcImage->height;

	/*initialize Dsift Filter*/
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	DsiftDescriptorGeometry* geom = init_dsift_geom(numBinX, numBinY, numBinT, binSizeX, binSizeY);
	DsiftFilter* self = init_dsift_filter(imageWidth, imageHeight, geom, step);
	dsift_alloc_buffers(self);
	IplImage * grayImage = cvCreateImage(cvSize(srcImage->width, srcImage->height), srcImage->depth, 1);
	cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
	compute_grad(self, grayImage);
	dsift_with_gaussian_window(self);

	DsiftKeypoint* frameIter = self->frames;
	float * descrIter = self->descrs;
	int framex, framey, bint;

	int frameSizeX = self->geom.binSizeX * (self->geom.numBinX - 1) + 1;
	int frameSizeY = self->geom.binSizeY * (self->geom.numBinY - 1) + 1;
	int descrSize = dsift_get_descriptor_size(self);

	float deltaCenterX = 0.5F * self->geom.binSizeX * (self->geom.numBinX - 1);
	float deltaCenterY = 0.5F * self->geom.binSizeY * (self->geom.numBinY - 1);

	float normConstant = frameSizeX * frameSizeY;

	for (framey = self->boundMinY;
		framey <= self->boundMaxY - frameSizeY + 1;
		framey += self->stepY) {
		
		for (framex = self->boundMinX;
			framex <= self->boundMaxX - frameSizeX + 1;
			framex += self->stepX) {

			frameIter->x = framex + deltaCenterX;
			frameIter->y = framey + deltaCenterY;

			/* mass */
			{
				float mass = 0;
				for (bint = 0; bint < descrSize; ++bint)
					mass += descrIter[bint];
				mass /= normConstant;
				frameIter->norm = mass;
			}

			/* L2 normalize */
			dsift_normalize_histogram(descrIter, descrIter + descrSize);

			/* clamp */
			for (bint = 0; bint < descrSize; ++bint)
			if (descrIter[bint] > 0.2F) descrIter[bint] = 0.2F;

			/* L2 normalize */
			dsift_normalize_histogram(descrIter, descrIter + descrSize);

			frameIter++;
			descrIter += descrSize;
		} /* for framex */
	} /* for framey */
	QueryPerformanceCounter(&t2);
	printf("Use Time:%f\n", (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
	ofstream out("output.txt");
	DsiftKeypoint const *frames = self->frames;
	ofstream outFrames("Frames.txt");
	float * outDescrIter = self->descrs;
	for (int i = 0; i < self->numFrames; i++)
	{
		outFrames << frames[i].y << "\t" << frames[i].x << "\t";
		outFrames << endl;
		float *tmpDescr = self->descrs + descrSize * i;
		for (int j = 0; j < descrSize; ++j) {
			unsigned char res = (unsigned char)(512.0F * tmpDescr[j] < 255.0F ? (512.0F * tmpDescr[j]) : 255.0F);
			out << (unsigned int)res  << "\t";
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