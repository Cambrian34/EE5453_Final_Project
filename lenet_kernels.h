#ifndef LENET_KERNELS_H
#define LENET_KERNELS_H

#include "network_config.h"

__global__ void conv1_forward_kernel(float *input, float *weights, float *bias, float *output,
                                     int batch_size, int in_w, int in_h, int in_c,
                                     int out_w, int out_h, int num_filters, int filter_size);

__global__ void conv2_forward_kernel(float *input, float *weights, float *bias, float *output,
                                     int batch_size, int in_w, int in_h, int in_c,
                                     int out_w, int out_h, int num_filters, int filter_size);

__global__ void max_pool_forward_kernel(float *input, float *output,
                                        int batch_size, int in_w, int in_h, int in_c,
                                        int pool_size, int stride);

__global__ void fc_forward_kernel(float *input, float *weights, float *bias, float *output,
                                  int batch_size, int in_nodes, int out_nodes, bool apply_relu);

__global__ void softmax_cross_entropy_kernel(float *input, int *labels, float *probs, float *loss,
                                             int batch_size, int num_classes);

__global__ void output_error_kernel(float *probs, int *labels, float *d_out,
                                    int batch_size, int num_classes);

__global__ void fc_weight_grad_kernel(float *input, float *d_out, float *d_weights,
                                      int batch_size, int in_nodes, int out_nodes);

__global__ void bias_grad_kernel(float *d_out, float *d_bias,
                                 int batch_size, int out_nodes);

__global__ void conv_weight_grad_kernel(float *input, float *d_out, float *d_weights,
                                        int batch_size, int in_w, int in_h, int in_c,
                                        int out_w, int out_h, int num_filters, int filter_size);

__global__ void update_weights_kernel(float *weights, float *d_weights, float *bias, float *d_bias,
                                      int size, int out_nodes, float lr, int batch_size);

__global__ void fc_backward_kernel(float *d_out, float *weights, float *d_in, float *input,
                                   int batch_size, int in_nodes, int out_nodes, bool apply_relu);

__global__ void max_pool_backward_kernel(float *d_out, float *input, float *d_in,
                                         int batch_size, int in_w, int in_h, int in_c,
                                         int pool_size, int stride, bool apply_relu_deriv);

__global__ void conv_backward_kernel(float *d_out, float *weights, float *d_in,
                                     int batch_size, int in_w, int in_h, int in_c,
                                     int out_w, int out_h, int num_filters, int filter_size);

__global__ void conv_bias_grad_kernel(float *d_out, float *d_bias,
                                      int batch_size, int out_w, int out_h, int num_filters);
#endif // LENET_KERNELS_H
