#include <algorithm>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "colorcode.h"

#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

// Convert color image to grayscale
float* to_grayscale(float* src_image, int w, int h, int comp) {
	float* result = new float[w*h];

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int idx = (y * w + x) * comp;
			result[y*w + x] = 0.2989f * src_image[idx + 0] + 0.5870f * src_image[idx + 1] + 0.1140f * src_image[idx + 2];
		}
	}

	return result;
}

// Horizontal convolution across 3 pixels
float* conv1x3(float* src_image, int w, int h, int comp, float w0, float w1, float w2) {
	float* result = new float[w*h*comp];

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int idx = (y*w + x)*comp;

			int prev_x = std::max(0, x - 1);
			int next_x = std::min(w - 1, x + 1);
			int prev_x_idx = (y*w + prev_x)*comp;
			int next_x_idx = (y*w + next_x)*comp;

			float dxr, dxg, dxb;
			dxr = std::min(1.0f, std::max(0.0f, w0 * src_image[next_x_idx + 0] + w1 * src_image[idx + 0] + w2 * src_image[prev_x_idx + 0]));
			dxg = std::min(1.0f, std::max(0.0f, w0 * src_image[next_x_idx + 1] + w1 * src_image[idx + 1] + w2 * src_image[prev_x_idx + 1]));
			dxb = std::min(1.0f, std::max(0.0f, w0 * src_image[next_x_idx + 2] + w1 * src_image[idx + 2] + w2 * src_image[prev_x_idx + 2]));

			float r = dxr;
			float g = dxg;
			float b = dxb;

			result[idx + 0] = r;
			result[idx + 1] = g;
			result[idx + 2] = b;
		}
	}

	return result;
}

// Vertical convolution across 3 pixels
float* conv3x1(float* src_image, int w, int h, int comp, float w0, float w1, float w2) {
	float* result = new float[w*h*comp];

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int idx = (y*w + x)*comp;

			int prev_y = std::max(0, y - 1);
			int next_y = std::min(h - 1, y + 1);
			int prev_y_idx = (prev_y*w + x)*comp;
			int next_y_idx = (next_y*w + x)*comp;

			float dyr, dyg, dyb;
			dyr = std::min(1.0f, std::max(0.0f, w0 * src_image[next_y_idx + 0] + w1 * src_image[idx + 0] + w2 * src_image[prev_y_idx + 0]));
			dyg = std::min(1.0f, std::max(0.0f, w0 * src_image[next_y_idx + 1] + w1 * src_image[idx + 1] + w2 * src_image[prev_y_idx + 1]));
			dyb = std::min(1.0f, std::max(0.0f, w0 * src_image[next_y_idx + 2] + w1 * src_image[idx + 2] + w2 * src_image[prev_y_idx + 2]));

			float r = dyr;
			float g = dyg;
			float b = dyb;

			result[idx + 0] = r;
			result[idx + 1] = g;
			result[idx + 2] = b;
		}
	}

	return result;
}

// Naive sobel edge detection
float* naive_sobel(float* src_image, int w, int h, int comp) {
	float* result = new float[w*h*comp];

	// Gx
	float* intermediate_x = conv1x3(src_image, w, h, comp, -1, 0, 1);
	float* intermediate_gx = conv3x1(intermediate_x, w, h, comp, 1, 2, 1);
	delete intermediate_x;

	// Gy
	float *intermediate_y = conv1x3(src_image, w, h, comp, 1, 2, 1);
	float* intermediate_gy = conv3x1(intermediate_y, w, h, comp, -1, 0, 1);
	delete intermediate_y;

	// G = sqrt(Gx^2 + Gy^2)
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int idx = (y*w + x)*comp;

			float r = std::sqrt(intermediate_gx[idx + 0] * intermediate_gx[idx + 0] + intermediate_gy[idx + 0] * intermediate_gy[idx + 0]);
			float g = std::sqrt(intermediate_gx[idx + 1] * intermediate_gx[idx + 1] + intermediate_gy[idx + 1] * intermediate_gy[idx + 1]);
			float b = std::sqrt(intermediate_gx[idx + 2] * intermediate_gx[idx + 2] + intermediate_gy[idx + 2] * intermediate_gy[idx + 2]);

			result[idx + 0] = r;
			result[idx + 1] = g;
			result[idx + 2] = b;
		}
	}

	delete intermediate_gx;
	delete intermediate_gy;

	return result;
}

// Naive gaussian
float* naive_gaussian(float* src_image, int w, int h, int comp) {
	float* intermediate = conv1x3(src_image, w, h, comp, 0.27901f, 0.44198f, 0.27901f);
	float* result = conv3x1(intermediate, w, h, comp, 0.27901f, 0.44198f, 0.27901f);

	//delete intermediate;
	return result;
}

// Declare the cuda LK helper function
float* cuda_lucaskanade(float* frame0, float* frame1, int w, int h);

int main(int argc, char** argv) {
	std::string cur_frame_filename, next_frame_filename;
	std::string out_filename("out.png");

	// Parse command line args
	for (int n = 1; n < argc; n++) {
		std::string arg = argv[n];

		if (arg.compare("-f1") == 0) {
			cur_frame_filename = argv[++n];
		}
		else if (arg.compare("-f2") == 0) {
			next_frame_filename = argv[++n];
		}
	}

	// Read in images
	int f0_x, f0_y, f0_comp;
	float* frame0_g = stbi_loadf(cur_frame_filename.c_str(), &f0_x, &f0_y, &f0_comp, 1);

	int f1_x, f1_y, f1_comp;
	float* frame1_g = stbi_loadf(next_frame_filename.c_str(), &f1_x, &f1_y, &f1_comp, 1);

	// Generate output image
	int out_x, out_y, out_comp;
	float* out_frame;

	out_x = f0_x;
	out_y = f0_y;
	out_comp = 3;
	//out_frame = naive_gaussian(frame0_g, out_x, out_y, out_comp);
	out_frame = cuda_lucaskanade(frame0_g, frame1_g, out_x, out_y);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	if (cudaDeviceReset() != cudaSuccess) {
	    fprintf(stderr, "cudaDeviceReset failed!");
	}
	
	// Convert motion vector to color encoding
	float max_rad = -1.0f;
	for (int y = 0; y < out_y; y++) {
		for (int x = 0; x < out_x; x++) {
			int idx = y*out_x + x;

			float vx = out_frame[idx * 2 + 0];
			float vy = out_frame[idx * 2 + 1];

			// Check to make sure motion vector is reasonable
			if (isnan(vx) || isnan(vy)) continue;
			if (fabs(vx) > 1e9 || fabs(vy) > 1e9) continue;

			max_rad = std::max(max_rad, sqrt(vx * vx + vy * vy));
		}
	}

	max_rad = (max_rad == 0.0f) ? -1.0f : max_rad; // if flow == 0 everywhere

	// Write result to image
	unsigned char* write_frame = new unsigned char[out_x * out_y * out_comp];
	for (int y = 0; y < out_y; y++) {
		for (int x = 0; x < out_x; x++) {
			int idx = y*out_x + x;

			float vx = out_frame[idx * 2 + 0];
			float vy = out_frame[idx * 2 + 1];

#define DEBUG_FRAME 1
#if DEBUG_FRAME
			write_frame[idx * 3 + 0] = (unsigned char)out_frame[idx * 2 + 0] * 255;
			write_frame[idx * 3 + 1] = (unsigned char)out_frame[idx * 2 + 1] * 255;
			write_frame[idx * 3 + 2] = 0;
#else
			// Check to make sure motion vector is reasonable
			if (isnan(vx) || isnan(vy) || fabs(vx) > 1e9 || fabs(vy) > 1e9) {
				write_frame[idx * 3 + 0] = 0;
				write_frame[idx * 3 + 1] = 0;
				write_frame[idx * 3 + 2] = 0;
			}
			else {
				//// Debug color encoding
				//computeColor(2 * ((float)x / out_x - 0.5f) / sqrt(2), 2 * ((float)y / out_y - 0.5f) / sqrt(2), &write_frame[idx * 3]);

				// Translate motion vector to color encoding
				computeColor(vx / max_rad, vy / max_rad, &write_frame[idx * 3]);

				// Need to un-swizzle because for some reason computeColor outputs BGR
				unsigned char temp = write_frame[idx * 3 + 0];
				write_frame[idx * 3 + 0] = write_frame[idx * 3 + 2];
				write_frame[idx * 3 + 2] = temp;
			}
#endif
		}
	}
	stbi_write_png(out_filename.c_str(), out_x, out_y, out_comp, write_frame, out_comp*out_x);

	return 0;
}