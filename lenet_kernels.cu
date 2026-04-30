#include <math.h>
#include "lenet_kernels.h"

// BONUS: Optimized Conv1 Forward using Shared Memory caching for inputs
__global__ void conv1_forward_kernel(float *input, float *weights, float *bias, float *output,
                                     int batch_size, int in_w, int in_h, int in_c,
                                     int out_w, int out_h, int num_filters, int filter_size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + tx;
    int out_y = blockIdx.y * blockDim.y + ty;
    int f = blockIdx.z % num_filters;
    int b = blockIdx.z / num_filters;

    if (b >= batch_size)
        return;

    // Shared memory for the input tile (all channels)
    extern __shared__ float s_input[];

    int in_tile_w = blockDim.x + filter_size - 1;
    int in_tile_h = blockDim.y + filter_size - 1;
    int linear_tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;
    int total_elements_per_channel = in_tile_w * in_tile_h;

    int base_x = blockIdx.x * blockDim.x;
    int base_y = blockIdx.y * blockDim.y;

    // Cooperative loading of input pixels into shared memory
    for (int c = 0; c < in_c; c++)
    {
        for (int i = linear_tid; i < total_elements_per_channel; i += total_threads)
        {
            int sy = i / in_tile_w;
            int sx = i % in_tile_w;
            int ix = base_x + sx;
            int iy = base_y + sy;

            int s_idx = c * total_elements_per_channel + i;
            if (ix >= 0 && ix < in_w && iy >= 0 && iy < in_h)
            {
                s_input[s_idx] = input[b * (in_c * in_h * in_w) + c * (in_h * in_w) + iy * in_w + ix];
            }
            else
            {
                s_input[s_idx] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (out_x < out_w && out_y < out_h)
    {
        float val = 0.0f;
        for (int c = 0; c < in_c; c++)
        {
            for (int fy = 0; fy < filter_size; fy++)
            {
                for (int fx = 0; fx < filter_size; fx++)
                {
                    int s_y = ty + fy;
                    int s_x = tx + fx;
                    val += s_input[c * total_elements_per_channel + s_y * in_tile_w + s_x] * weights[f * (in_c * filter_size * filter_size) + c * (filter_size * filter_size) + fy * filter_size + fx];
                }
            }
        }
        val += bias[f];
        output[b * (num_filters * out_h * out_w) + f * (out_h * out_w) + out_y * out_w + out_x] = fmaxf(0.0f, val); // ReLU
    }
}
// Conv2 Forward (similar to conv1 but without shared memory for simplicity)
__global__ void conv2_forward_kernel(float *input, float *weights, float *bias, float *output,
                                     int batch_size, int in_w, int in_h, int in_c,
                                     int out_w, int out_h, int num_filters, int filter_size)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z % num_filters;
    int b = blockIdx.z / num_filters;

    if (b >= batch_size || out_x >= out_w || out_y >= out_h)
        return;

    float val = 0.0f;
    for (int c = 0; c < in_c; c++)
    {
        for (int fy = 0; fy < filter_size; fy++)
        {
            for (int fx = 0; fx < filter_size; fx++)
            {
                int in_x = out_x * 1 + fx; // stride 1
                int in_y = out_y * 1 + fy;
                if (in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h)
                {
                    int in_idx = b * (in_c * in_h * in_w) + c * (in_h * in_w) + in_y * in_w + in_x;
                    int w_idx = f * (in_c * filter_size * filter_size) + c * (filter_size * filter_size) + fy * filter_size + fx;
                    val += input[in_idx] * weights[w_idx];
                }
            }
        }
    }
    val += bias[f];
    val = fmaxf(0.0f, val);
    int out_idx = b * (num_filters * out_h * out_w) + f * (out_h * out_w) + out_y * out_w + out_x;
    output[out_idx] = val;
}

// Max Pooling Forward
__global__ void max_pool_forward_kernel(float *input, float *output,
                                        int batch_size, int in_w, int in_h, int in_c,
                                        int pool_size, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % in_c;
    int b = blockIdx.z / in_c;

    if (b >= batch_size || out_x >= (in_w / stride) || out_y >= (in_h / stride))
        return;

    float max_val = -INFINITY;
    for (int py = 0; py < pool_size; py++)
    {
        for (int px = 0; px < pool_size; px++)
        {
            int in_x = out_x * stride + px;
            int in_y = out_y * stride + py;
            if (in_x < in_w && in_y < in_h)
            {
                int in_idx = b * (in_c * in_h * in_w) + c * (in_h * in_w) + in_y * in_w + in_x;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }
    int out_idx = b * (in_c * (in_h / stride) * (in_w / stride)) + c * ((in_h / stride) * (in_w / stride)) + out_y * (in_w / stride) + out_x;
    output[out_idx] = max_val;
}

// Standard Fully Connected Forward Layer
__global__ void fc_forward_kernel(float *input, float *weights, float *bias, float *output,
                                  int batch_size, int in_nodes, int out_nodes, bool apply_relu)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (node < out_nodes && b < batch_size)
    {
        float val = 0.0f;
        for (int i = 0; i < in_nodes; i++)
        {
            val += input[b * in_nodes + i] * weights[node * in_nodes + i];
        }
        val += bias[node];
        if (apply_relu)
        {
            val = fmaxf(0.0f, val);
        }
        output[b * out_nodes + node] = val;
    }
}

// Softmax & Cross Entropy Loss calculation
__global__ void softmax_cross_entropy_kernel(float *input, int *labels, float *probs, float *loss, int batch_size, int num_classes)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size)
    {
        float max_val = input[b * num_classes];
        for (int i = 1; i < num_classes; i++)
        {
            if (input[b * num_classes + i] > max_val)
                max_val = input[b * num_classes + i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++)
        {
            probs[b * num_classes + i] = expf(input[b * num_classes + i] - max_val);
            sum_exp += probs[b * num_classes + i];
        }
        for (int i = 0; i < num_classes; i++)
        {
            probs[b * num_classes + i] /= sum_exp;
        }
        int label = labels[b];
        loss[b] = -logf(probs[b * num_classes + label] + 1e-7f);
    }
}

// Output Layer Error (Softmax Gradient)
__global__ void output_error_kernel(float *probs, int *labels, float *d_out, int batch_size, int num_classes)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size)
    {
        int label = labels[b];
        for (int i = 0; i < num_classes; i++)
        {
            d_out[b * num_classes + i] = probs[b * num_classes + i] - (i == label ? 1.0f : 0.0f);
        }
    }
}

__global__ void fc_weight_grad_kernel(float *input, float *d_out, float *d_weights,
                                      int batch_size, int in_nodes, int out_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = in_nodes * out_nodes;
    if (idx >= total_weights)
        return;
    int out_idx = idx / in_nodes;
    int in_idx = idx % in_nodes;
    float grad = 0.0f;
    for (int b = 0; b < batch_size; b++)
    {
        grad += d_out[b * out_nodes + out_idx] * input[b * in_nodes + in_idx];
    }
    d_weights[idx] = grad;
}

__global__ void bias_grad_kernel(float *d_out, float *d_bias,
                                 int batch_size, int out_nodes)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_nodes)
        return;
    float grad = 0.0f;
    for (int b = 0; b < batch_size; b++)
    {
        grad += d_out[b * out_nodes + out_idx];
    }
    d_bias[out_idx] = grad;
}

// SGD Weight Update Kernel
__global__ void update_weights_kernel(float *weights, float *d_weights, float *bias, float *d_bias,
                                      int size, int out_nodes, float lr, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Average gradient and apply update
        float grad = d_weights[idx] / (float)batch_size;
        weights[idx] -= lr * grad;
        d_weights[idx] = 0.0f; // Reset for next batch
    }

    // Bias update
    if (idx < out_nodes)
    {
        float b_grad = d_bias[idx] / (float)batch_size;
        bias[idx] -= lr * b_grad;
        d_bias[idx] = 0.0f;
    }
}

// ADAM Weight Update Kernel
__global__ void adam_update_weights_kernel(float *weights, float *d_weights, float *bias, float *d_bias,
                                           float *m_weights, float *v_weights, float *m_bias, float *v_bias,
                                           int size, int out_nodes, float lr, float beta1, float beta2, float epsilon, int t, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bias-correction terms
    float bias_correction1 = 1.0f - powf(beta1, (float)t);
    float bias_correction2 = 1.0f - powf(beta2, (float)t);
    float lr_corrected = lr * sqrtf(bias_correction2) / bias_correction1;

    // Weight update with ADAM
    if (idx < size)
    {
        float grad = d_weights[idx] / (float)batch_size;

        // Update biased first moment estimate (mean)
        m_weights[idx] = beta1 * m_weights[idx] + (1.0f - beta1) * grad;

        // Update biased second moment estimate (variance)
        v_weights[idx] = beta2 * v_weights[idx] + (1.0f - beta2) * grad * grad;

        // Bias-corrected first and second moment estimates
        float m_hat = m_weights[idx] / bias_correction1;
        float v_hat = v_weights[idx] / bias_correction2;

        // Update weight
        weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);

        d_weights[idx] = 0.0f; // Reset for next batch
    }

    // Bias update with ADAM
    if (idx < out_nodes)
    {
        float b_grad = d_bias[idx] / (float)batch_size;

        // Update biased first moment estimate (mean)
        m_bias[idx] = beta1 * m_bias[idx] + (1.0f - beta1) * b_grad;

        // Update biased second moment estimate (variance)
        v_bias[idx] = beta2 * v_bias[idx] + (1.0f - beta2) * b_grad * b_grad;

        // Bias-corrected first and second moment estimates
        float m_hat = m_bias[idx] / bias_correction1;
        float v_hat = v_bias[idx] / bias_correction2;

        // Update bias
        bias[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);

        d_bias[idx] = 0.0f;
    }
}

// Conv Weight Gradient Kernel
__global__ void conv_weight_grad_kernel(float *input, float *d_out, float *d_weights,
                                        int batch_size, int in_w, int in_h, int in_c,
                                        int out_w, int out_h, int num_filters, int filter_size)
{
    int f = blockIdx.z;
    int c = blockIdx.y;
    int fy = blockIdx.x / filter_size;
    int fx = blockIdx.x % filter_size;

    if (f >= num_filters || c >= in_c || fy >= filter_size || fx >= filter_size)
        return;

    float grad = 0.0f;
    for (int b = 0; b < batch_size; b++)
    {
        for (int oy = 0; oy < out_h; oy++)
        {
            for (int ox = 0; ox < out_w; ox++)
            {
                int in_x = ox + fx;
                int in_y = oy + fy;
                if (in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h)
                {
                    int in_idx = b * (in_c * in_h * in_w) + c * (in_h * in_w) + in_y * in_w + in_x;
                    int out_idx = b * (num_filters * out_h * out_w) + f * (out_h * out_w) + oy * out_w + ox;
                    grad += input[in_idx] * d_out[out_idx];
                }
            }
        }
    }
    int w_idx = f * (in_c * filter_size * filter_size) + c * (filter_size * filter_size) + fy * filter_size + fx;
    d_weights[w_idx] = grad;
}

__global__ void fc_backward_kernel(float *d_out, float *weights, float *d_in, float *input,
                                   int batch_size, int in_nodes, int out_nodes, bool apply_relu)
{
    int in_node = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (in_node < in_nodes && b < batch_size)
    {
        float grad = 0.0f;
        for (int out_node = 0; out_node < out_nodes; out_node++)
        {
            grad += d_out[b * out_nodes + out_node] * weights[out_node * in_nodes + in_node];
        }
        if (apply_relu)
        {
            // Derivative of ReLU
            grad = (input[b * in_nodes + in_node] > 0.0f) ? grad : 0.0f;
        }
        d_in[b * in_nodes + in_node] = grad;
    }
}

// Max Pool Backward: Routes the incoming gradient only to the pixel that had the max value
__global__ void max_pool_backward_kernel(float *d_out, float *input, float *d_in,
                                         int batch_size, int in_w, int in_h, int in_c,
                                         int pool_size, int stride, bool apply_relu_deriv)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % in_c;
    int b = blockIdx.z / in_c;

    int out_w = in_w / stride;
    int out_h = in_h / stride;

    if (b >= batch_size || out_x >= out_w || out_y >= out_h)
        return;

    // Find the max pixel index from the forward pass
    float max_val = -INFINITY;
    int max_idx_x = -1;
    int max_idx_y = -1;

    for (int py = 0; py < pool_size; py++)
    {
        for (int px = 0; px < pool_size; px++)
        {
            int in_x = out_x * stride + px;
            int in_y = out_y * stride + py;
            if (in_x < in_w && in_y < in_h)
            {
                int in_idx = b * (in_c * in_h * in_w) + c * (in_h * in_w) + in_y * in_w + in_x;
                float val = input[in_idx];
                if (val > max_val)
                {
                    max_val = val;
                    max_idx_x = in_x;
                    max_idx_y = in_y;
                }
            }
        }
    }

    // Assign gradient to the max pixel
    float grad = d_out[b * (in_c * out_h * out_w) + c * (out_h * out_w) + out_y * out_w + out_x];

    if (max_idx_x != -1 && max_idx_y != -1)
    {
        int max_in_idx = b * (in_c * in_h * in_w) + c * (in_h * in_w) + max_idx_y * in_w + max_idx_x;
        // Apply derivative of the convolution's ReLU here
        if (apply_relu_deriv && input[max_in_idx] <= 0.0f)
        {
            grad = 0.0f;
        }
        atomicAdd(&d_in[max_in_idx], grad); // Safely accumulate gradient
    }
}

// Convolution Backward: Calculates the error with respect to the input feature map
__global__ void conv_backward_kernel(float *d_out, float *weights, float *d_in,
                                     int batch_size, int in_w, int in_h, int in_c,
                                     int out_w, int out_h, int num_filters, int filter_size)
{
    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    int in_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % in_c;
    int b = blockIdx.z / in_c;

    if (b >= batch_size || in_x >= in_w || in_y >= in_h)
        return;

    float grad = 0.0f;
    for (int f = 0; f < num_filters; f++)
    {
        for (int fy = 0; fy < filter_size; fy++)
        {
            for (int fx = 0; fx < filter_size; fx++)
            {
                int out_x = in_x - fx;
                int out_y = in_y - fy;

                if (out_x >= 0 && out_x < out_w && out_y >= 0 && out_y < out_h)
                {
                    int out_idx = b * (num_filters * out_h * out_w) + f * (out_h * out_w) + out_y * out_w + out_x;
                    int w_idx = f * (in_c * filter_size * filter_size) + c * (filter_size * filter_size) + fy * filter_size + fx;
                    grad += d_out[out_idx] * weights[w_idx];
                }
            }
        }
    }

    d_in[b * (in_c * in_h * in_w) + c * (in_h * in_w) + in_y * in_w + in_x] = grad;
}

__global__ void conv_bias_grad_kernel(float *d_out, float *d_bias,
                                      int batch_size, int out_w, int out_h, int num_filters)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_filters)
        return;

    float grad = 0.0f;
    for (int b = 0; b < batch_size; b++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                int idx = b * (num_filters * out_h * out_w) + f * (out_h * out_w) + y * out_w + x;
                grad += d_out[idx];
            }
        }
    }
    d_bias[f] = grad;
}