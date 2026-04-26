#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include "network_config.h"

void load_cifar10_batch(const char* filename, float* images, int* labels, int num_images);
void init_weights(float* w, int size);

#endif // DATA_UTILS_H
