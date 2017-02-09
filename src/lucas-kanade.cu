#include <algorithm>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <system_error>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// cuBLAS includes
#include "cublas_v2.h"

/**************************************************/

// Function headers
float* cuda_lucaskanade(float* frame0, float* frame1, int w, int h);

// CUDA error handler
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		std::stringstream ss;
		ss << file << "(" << line << ")" << std::endl << cudaGetErrorName(code) << " " << cudaGetErrorString(code);
		
		std::string err_str;
		ss >> err_str;
		throw std::system_error(std::error_code(code, std::generic_category()), err_str.c_str());
	}
}

// MIPMAP GENERATION

uint32_t getMipMapLevels(cudaExtent size) {
	uint32_t res = static_cast<uint32_t>(1 + std::floor(std::log2(std::max({ size.width, size.height, size.depth }))));
	return res;
}

__global__ void d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, unsigned int imageW, unsigned int imageH) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float px = 1.0 / float(imageW);
	float py = 1.0 / float(imageH);


	if ((x < imageW) && (y < imageH))
	{
		// we are using the normalized access to make sure non-power-of-two textures
		// behave well when downsized.

		// Sample a 2x2 block next to current pixel
		float color =
			(tex2D<float>(mipInput, (x + 0) * px, (y + 0) * py)) +
			(tex2D<float>(mipInput, (x + 1) * px, (y + 0) * py)) +
			(tex2D<float>(mipInput, (x + 1) * px, (y + 1) * py)) +
			(tex2D<float>(mipInput, (x + 0) * px, (y + 1) * py));
		color /= 4.0;

		// Sample weighted 3x3 block centered at current pixel
		// This one seems to shift the image so I would avoid it
		//float color =
		//	0.25   * (tex2D<float>(mipInput, (x + 0) * px, (y + 0) * py)) +
		//	0.125  * (tex2D<float>(mipInput, (x + 1) * px, (y + 0) * py)) +
		//	0.125  * (tex2D<float>(mipInput, (x + 0) * px, (y + 1) * py)) +
		//	0.125  * (tex2D<float>(mipInput, (x + 0) * px, (y - 1) * py)) +
		//	0.125  * (tex2D<float>(mipInput, (x - 1) * px, (y + 0) * py)) +
		//	0.0625 * (tex2D<float>(mipInput, (x + 1) * px, (y + 1) * py)) +
		//	0.0625 * (tex2D<float>(mipInput, (x + 1) * px, (y - 1) * py)) +
		//	0.0625 * (tex2D<float>(mipInput, (x - 1) * px, (y + 1) * py)) +
		//	0.0625 * (tex2D<float>(mipInput, (x - 1) * px, (y - 1) * py));

		color = min(color, 1.0);

		surf2Dwrite(color, mipOutput, x * sizeof(float), y);
	}
}

void generateMipMaps(cudaMipmappedArray_t mipmapArray, cudaExtent extent) {
	size_t width = extent.width;
	size_t height = extent.height;

	unsigned int level = 0;

	while (width != 1 || height != 1)
	{
		width /= 2;
		width = std::max((size_t)1, width);
		height /= 2;
		height = std::max((size_t)1, height);

		cudaArray_t levelFrom;
		CUDA_CALL(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
		cudaArray_t levelTo;
		CUDA_CALL(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

		cudaExtent  levelToSize;
		CUDA_CALL(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
		assert(levelToSize.width == width);
		assert(levelToSize.height == height);
		assert(levelToSize.depth == 0);

		// generate texture object for reading
		cudaTextureObject_t         texInput;
		cudaResourceDesc            texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));

		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = levelFrom;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));

		texDesc.normalizedCoords = 1;
		texDesc.filterMode = cudaFilterModeLinear;

		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;

		texDesc.readMode = cudaReadModeElementType;

		CUDA_CALL(cudaCreateTextureObject(&texInput, &texRes, &texDesc, NULL));

		// generate surface object for writing
		cudaSurfaceObject_t surfOutput;
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = levelTo;

		CUDA_CALL(cudaCreateSurfaceObject(&surfOutput, &surfRes));

		// run mipmap kernel
		dim3 blockSize(16, 16, 1);
		dim3 gridSize(((unsigned int)width + blockSize.x - 1) / blockSize.x, ((unsigned int)height + blockSize.y - 1) / blockSize.y, 1);

		d_mipmap<<<gridSize, blockSize>>>(surfOutput, texInput, (unsigned int)width, (unsigned int)height);

		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaGetLastError());

		CUDA_CALL(cudaDestroySurfaceObject(surfOutput));

		CUDA_CALL(cudaDestroyTextureObject(texInput));

		level++;
	}
}

void setImageData(cudaMipmappedArray_t mipmapArray, float* src, cudaExtent extent) {
	size_t width = extent.width;
	size_t height = extent.height;

	// Upload level 0
	cudaArray_t baseLevel;
	cudaGetMipmappedArrayLevel(&baseLevel, mipmapArray, 0);
	cudaMemcpyToArray(baseLevel, 0, 0, src, sizeof(float)*width*height, cudaMemcpyHostToDevice);

	// Compute remaining mipmap levels
	generateMipMaps(mipmapArray, extent);
}

/***************************************************/

__global__ void lkKernel(float* result, cudaTextureObject_t frame0, cudaTextureObject_t frame1, const int width, const int height, int level)
{
	int windowSize = 5;

	float px = 1.0 / float(width);
	float py = 1.0 / float(height);

	unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y);
	unsigned int idx = y*width + x;

	float x0 = x >> level;
	float y0 = y >> level;

	result[idx * 2 + 0] = tex2DLod<float>(frame0, (x0) * px, (y0) * py, level);
	result[idx * 2 + 1] = tex2DLod<float>(frame1, (x0) * px, (y0) * py, level);

	return;

	if (x > width - 1 || y > height - 1) return;

	float det, D;

	float sum_Ixx = 0.0f;
	float sum_Ixy = 0.0f;
	float sum_Iyy = 0.0f;
	float Ix, Iy, It; // Image gradients

	level = 0;

	// Calculate spatial gradient
	for (int yy = -windowSize; yy <= windowSize; yy++) {
		for (int xx = -windowSize; xx <= windowSize; xx++) {
			Ix = tex2DLod<float>(frame0, (x0 + xx + 1) * px, (y0 + yy + 0) * py, level) - tex2DLod<float>(frame0, (x0 + xx - 1) * px, (y0 + yy + 0) * py, level);
			Iy = tex2DLod<float>(frame0, (x0 + xx + 0) * px, (y0 + yy + 1) * py, level) - tex2DLod<float>(frame0, (x0 + xx + 0) * px, (y0 + yy - 1) * py, level);
			
			sum_Ixx += Ix*Ix;
			sum_Ixy += Ix*Iy;
			sum_Iyy += Iy*Iy;
		}
	}

	det = sum_Ixx*sum_Iyy - sum_Ixy*sum_Ixy;

	if (det < 0.00001f) return;

	D = 1 / det;

	// Iterations
	float Vx = result[idx * 2 + 0];
	float Vy = result[idx * 2 + 1];

	float x1 = x + Vx;
	float y1 = y + Vy;

	float I, J;

	float sum_Ixt;
	float sum_Iyt;

	for (int iter = 0; iter < 5; iter++) {
		if (x1 < 0 || x1 > width - 1 || y1 < 0 || y1 > height - 1) return;

		sum_Ixt = 0.0f;
		sum_Iyt = 0.0f;

		for (int yy = -windowSize; yy <= windowSize; yy++) {
			for (int xx = -windowSize; xx <= windowSize; xx++) {
				I = tex2DLod<float>(frame0, (x0 + xx) * px, (y0 + yy) * py, level);
				J = tex2DLod<float>(frame1, (x1 + xx) * px, (y1 + yy) * py, level);

				Ix = tex2DLod<float>(frame0, (x0 + xx + 1) * px, (y0 + yy + 0) * py, level) - tex2DLod<float>(frame0, (x0 + xx - 1) * px, (y0 + yy + 0) * py, level);
				Iy = tex2DLod<float>(frame0, (x0 + xx + 0) * px, (y0 + yy + 1) * py, level) - tex2DLod<float>(frame0, (x0 + xx + 0) * px, (y0 + yy - 1) * py, level);

				It = J - I;

				sum_Ixt += Ix*It;
				sum_Iyt += Iy*It;
			}
		}

		float vx = D*(-sum_Iyy*sum_Ixt + sum_Ixy*sum_Iyt);
		float vy = D*( sum_Ixy*sum_Ixt - sum_Ixx*sum_Iyt);

		Vx += vx;
		Vy += vy;
		x1 += vx;
		y1 += vy;

		// Stop if movement is sufficiently small
		if (fabsf(vx) < 0.01f && fabsf(vy) < 0.01f) break;
	}

	if (level != 0) {
		Vx += Vx;
		Vy += Vy;
	}

	result[idx * 2 + 0] = I;// Vx;
	result[idx * 2 + 1] = J;// Vy;
}

float* cuda_lucaskanade(float* frame0, float* frame1, int w, int h) {
	float* result = new float[w*h*2];
	
	try {
		// Set CUDA device
		CUDA_CALL(cudaSetDevice(0));

		// Allocate image buffers on GPU
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaExtent imageExtent;
		imageExtent.width  = w;
		imageExtent.height = h;
		imageExtent.depth  = 0;

		cudaMipmappedArray_t d_frame0;
		cudaMipmappedArray_t d_frame1;

		CUDA_CALL(cudaMallocMipmappedArray(&d_frame0, &channelDesc, imageExtent, getMipMapLevels(imageExtent)));
		CUDA_CALL(cudaMallocMipmappedArray(&d_frame1, &channelDesc, imageExtent, getMipMapLevels(imageExtent)));

		// Set image data
		setImageData(d_frame0, frame0, imageExtent);
		setImageData(d_frame1, frame1, imageExtent);

		// Create the texture objects
		cudaTextureObject_t d_tex0;
		cudaTextureObject_t d_tex1;

		cudaResourceDesc resDesc0;
		memset(&resDesc0, 0, sizeof(cudaResourceDesc));

		resDesc0.resType = cudaResourceTypeMipmappedArray;
		resDesc0.res.mipmap.mipmap = d_frame0;

		cudaResourceDesc resDesc1;
		memset(&resDesc1, 0, sizeof(cudaResourceDesc));

		resDesc1.resType = cudaResourceTypeMipmappedArray;
		resDesc1.res.mipmap.mipmap = d_frame1;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));

		texDesc.normalizedCoords = 1;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.mipmapFilterMode = cudaFilterModeLinear;

		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;

		texDesc.maxMipmapLevelClamp = float(getMipMapLevels(imageExtent) - 1);

		texDesc.readMode = cudaReadModeElementType;

		CUDA_CALL(cudaCreateTextureObject(&d_tex0, &resDesc0, &texDesc, NULL));
		CUDA_CALL(cudaCreateTextureObject(&d_tex1, &resDesc1, &texDesc, NULL));

		// Create intermediate resources
		float* d_dx;
		float* d_dy;
		CUDA_CALL(cudaMalloc(&d_dx, sizeof(float)*w*h));
		CUDA_CALL(cudaMalloc(&d_dy, sizeof(float)*w*h));

		CUDA_CALL(cudaMemset(d_dx, 0, sizeof(float)*w*h));
		CUDA_CALL(cudaMemset(d_dy, 0, sizeof(float)*w*h));

		// Create result resources
		float* d_result;
		CUDA_CALL(cudaMalloc(&d_result, sizeof(float)*w*h*2));

		// Launch the kernel
		dim3 blockSize(16, 16, 1);
		dim3 gridSize(((unsigned int)w + blockSize.x - 1) / blockSize.x, ((unsigned int)h + blockSize.y - 1) / blockSize.y, 1);
		//for (int l = getMipMapLevels(imageExtent) - 1; l >= 0; l--) {
		//	lkKernel<<<gridSize, blockSize>>>(d_result, d_tex0, d_tex1, std::max(w>>l,1), std::max(h>>l,1), l);
		//	cudaThreadSynchronize();
		//}
		int l = 0;
		lkKernel << <gridSize, blockSize >> >(d_result, d_tex0, d_tex1, std::max(w >> l, 1), std::max(h >> l, 1), l);
		cudaThreadSynchronize();
		CUDA_CALL(cudaPeekAtLastError());

		// Wait for kernel to finish
		CUDA_CALL(cudaDeviceSynchronize());
		 
		// Retrieve output image from GPU
		CUDA_CALL(cudaMemcpy(result, d_result, sizeof(float)*w*h*2, cudaMemcpyDeviceToHost));

		// Free the buffers
		//CUDA_CALL(cudaFree(d_frame0));
		//CUDA_CALL(cudaFree(d_frame1));
		//CUDA_CALL(cudaFree(d_result));
		//system("pause");
	}
	catch (std::system_error& e) {
		std::cerr << "CUDA ERROR " << e.code().value() << ": " << e.what() << std::endl;
		system("pause");
	}

	return result;
}