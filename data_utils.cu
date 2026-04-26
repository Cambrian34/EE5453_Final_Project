#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data_utils.h"

void load_cifar10_batch(const char* filename, float* images, int* labels, int num_images) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening CIFAR-10 file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < num_images; i++) {
        unsigned char label;
        size_t got = fread(&label, 1, 1, file);
        if (got != 1) {
            fprintf(stderr, "Error reading label at image %d\n", i);
            exit(1);
        }
        labels[i] = (int)label;
        unsigned char pixels[IMG_CHANNELS * IMG_WIDTH * IMG_HEIGHT];
        got = fread(pixels, 1, sizeof(pixels), file);
        if (got != sizeof(pixels)) {
            fprintf(stderr, "Error reading image pixels at image %d\n", i);
            exit(1);
        }
        for (int j = 0; j < (int)sizeof(pixels); j++) {
            // Normalization: CIFAR pixels are 0-255. Map to [-0.5, 0.5] or [0, 1]
            images[i * sizeof(pixels) + j] = ((float)pixels[j] / 255.0f) - 0.5f;
        }
    }
    fclose(file);
    printf("Loaded %d images from %s\n", num_images, filename);
}

/**
 * Xavier (Glorot) Initialization
 * Formula: Uniform distribution in [-limit, limit]
 * where limit = sqrt(6 / (fan_in + fan_out))
 */
void init_weights_xavier(float* w, int size, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < size; i++) {
        w[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * limit - limit;
    }
}

// Keep old signature for compatibility if needed, but updated to use internal logic
void init_weights(float* w, int size) {
    // Default fallback to a small range if fan_in/out aren't known
    for (int i = 0; i < size; i++) {
        w[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
}