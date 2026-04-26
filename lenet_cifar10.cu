#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "network_config.h"
#include "data_utils.h"
#include "lenet_kernels.h"

void init_weights_xavier(float *w, int size, int fan_in, int fan_out);

int main()
{
    FILE *output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        printf("Error: Could not open output.txt for writing\n");
        return 1;
    }

    fprintf(output_file, "Starting LeNet-5 CIFAR-10 Training on GPU (Fixed Conv Gradients)...\n");
    fprintf(stdout, "Starting LeNet-5 CIFAR-10 Training on GPU (Fixed Conv Gradients)...\n");

    int num_images = 10000;
    size_t img_mem_size = num_images * IMG_CHANNELS * IMG_WIDTH * IMG_HEIGHT * sizeof(float);
    size_t label_mem_size = num_images * sizeof(int);

    float *h_batch_images = (float *)malloc(img_mem_size);
    int *h_batch_labels = (int *)malloc(label_mem_size);
    float *h_test_images = (float *)malloc(img_mem_size);
    int *h_test_labels = (int *)malloc(label_mem_size);

    float *d_batch_images;
    int *d_batch_labels;
    cudaCheck(cudaMalloc((void **)&d_batch_images, img_mem_size));
    cudaCheck(cudaMalloc((void **)&d_batch_labels, label_mem_size));

    int conv1_w_size = C1_FILTERS * IMG_CHANNELS * C1_SIZE * C1_SIZE;
    int conv1_out_size = C1_FILTERS * C1_OUT_W * C1_OUT_H;
    int p1_out_size = C1_FILTERS * P1_OUT_W * P1_OUT_H;
    int conv2_w_size = C2_FILTERS * C1_FILTERS * C2_SIZE * C2_SIZE;
    int conv2_out_size = C2_FILTERS * C2_OUT_W * C2_OUT_H;
    int p2_out_size = C2_FILTERS * P2_OUT_W * P2_OUT_H;
    int fc1_w_size = FC1_SIZE * p2_out_size;
    int fc2_w_size = FC2_SIZE * FC1_SIZE;
    int fc3_w_size = NUM_CLASSES * FC2_SIZE;

    float *h_w_c1 = (float *)malloc(conv1_w_size * sizeof(float));
    init_weights_xavier(h_w_c1, conv1_w_size, IMG_CHANNELS * 5 * 5, C1_FILTERS * 5 * 5);
    float *h_b_c1 = (float *)calloc(C1_FILTERS, sizeof(float));

    float *h_w_c2 = (float *)malloc(conv2_w_size * sizeof(float));
    init_weights_xavier(h_w_c2, conv2_w_size, C1_FILTERS * 5 * 5, C2_FILTERS * 5 * 5);
    float *h_b_c2 = (float *)calloc(C2_FILTERS, sizeof(float));

    float *h_w_fc1 = (float *)malloc(fc1_w_size * sizeof(float));
    init_weights_xavier(h_w_fc1, fc1_w_size, p2_out_size, FC1_SIZE);
    float *h_b_fc1 = (float *)calloc(FC1_OUT, sizeof(float));

    float *h_w_fc2 = (float *)malloc(fc2_w_size * sizeof(float));
    init_weights_xavier(h_w_fc2, fc2_w_size, FC1_SIZE, FC2_SIZE);
    float *h_b_fc2 = (float *)calloc(FC2_OUT, sizeof(float));

    float *h_w_fc3 = (float *)malloc(fc3_w_size * sizeof(float));
    init_weights_xavier(h_w_fc3, fc3_w_size, FC2_SIZE, NUM_CLASSES);
    float *h_b_fc3 = (float *)calloc(NUM_CLASSES, sizeof(float));

    float *d_w_c1, *d_b_c1, *d_w_c2, *d_b_c2, *d_w_fc1, *d_b_fc1, *d_w_fc2, *d_b_fc2, *d_w_fc3, *d_b_fc3;
    float *d_conv1_out, *d_p1_out, *d_conv2_out, *d_p2_out, *d_fc1_out, *d_fc2_out, *d_fc3_out;
    float *d_probs, *d_loss, *d_d_fc3, *d_d_fc2, *d_d_fc1, *d_d_p2, *d_d_conv2, *d_d_p1, *d_d_conv1;
    float *d_d_w_c1, *d_d_b_c1, *d_d_w_c2, *d_d_b_c2, *d_d_w_fc1, *d_d_b_fc1, *d_d_w_fc2, *d_d_b_fc2, *d_d_w_fc3, *d_d_b_fc3;
    
    // ADAM optimizer state buffers - first moment (m) and second moment (v)
    float *m_w_c1, *v_w_c1, *m_b_c1, *v_b_c1;
    float *m_w_c2, *v_w_c2, *m_b_c2, *v_b_c2;
    float *m_w_fc1, *v_w_fc1, *m_b_fc1, *v_b_fc1;
    float *m_w_fc2, *v_w_fc2, *m_b_fc2, *v_b_fc2;
    float *m_w_fc3, *v_w_fc3, *m_b_fc3, *v_b_fc3;
    int adam_t = 0; // ADAM iteration counter

    cudaCheck(cudaMalloc(&d_w_c1, conv1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b_c1, C1_FILTERS * sizeof(float)));
    cudaCheck(cudaMalloc(&d_conv1_out, BATCH_SIZE * conv1_out_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_p1_out, BATCH_SIZE * p1_out_size * sizeof(float)));

    cudaCheck(cudaMalloc(&d_w_c2, conv2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b_c2, C2_FILTERS * sizeof(float)));
    cudaCheck(cudaMalloc(&d_conv2_out, BATCH_SIZE * conv2_out_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_p2_out, BATCH_SIZE * p2_out_size * sizeof(float)));

    cudaCheck(cudaMalloc(&d_w_fc1, fc1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b_fc1, FC1_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&d_fc1_out, BATCH_SIZE * FC1_OUT * sizeof(float)));

    cudaCheck(cudaMalloc(&d_w_fc2, fc2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b_fc2, FC2_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&d_fc2_out, BATCH_SIZE * FC2_OUT * sizeof(float)));

    cudaCheck(cudaMalloc(&d_w_fc3, fc3_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b_fc3, NUM_CLASSES * sizeof(float)));
    cudaCheck(cudaMalloc(&d_fc3_out, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    cudaCheck(cudaMalloc(&d_probs, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    cudaCheck(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));

    cudaCheck(cudaMalloc(&d_d_fc3, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_fc2, BATCH_SIZE * FC2_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_fc1, BATCH_SIZE * FC1_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_p2, BATCH_SIZE * p2_out_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_conv2, BATCH_SIZE * conv2_out_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_p1, BATCH_SIZE * p1_out_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_conv1, BATCH_SIZE * conv1_out_size * sizeof(float)));

    cudaCheck(cudaMalloc(&d_d_w_c1, conv1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_b_c1, C1_FILTERS * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_w_c2, conv2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_b_c2, C2_FILTERS * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_w_fc1, fc1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_b_fc1, FC1_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_w_fc2, fc2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_b_fc2, FC2_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_w_fc3, fc3_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_d_b_fc3, NUM_CLASSES * sizeof(float)));

    // Allocate ADAM optimizer state buffers (first and second moment estimates)
    // Conv1 layer
    cudaCheck(cudaMalloc(&m_w_c1, conv1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&v_w_c1, conv1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&m_b_c1, C1_FILTERS * sizeof(float)));
    cudaCheck(cudaMalloc(&v_b_c1, C1_FILTERS * sizeof(float)));
    
    // Conv2 layer
    cudaCheck(cudaMalloc(&m_w_c2, conv2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&v_w_c2, conv2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&m_b_c2, C2_FILTERS * sizeof(float)));
    cudaCheck(cudaMalloc(&v_b_c2, C2_FILTERS * sizeof(float)));
    
    // FC1 layer
    cudaCheck(cudaMalloc(&m_w_fc1, fc1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&v_w_fc1, fc1_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&m_b_fc1, FC1_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&v_b_fc1, FC1_OUT * sizeof(float)));
    
    // FC2 layer
    cudaCheck(cudaMalloc(&m_w_fc2, fc2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&v_w_fc2, fc2_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&m_b_fc2, FC2_OUT * sizeof(float)));
    cudaCheck(cudaMalloc(&v_b_fc2, FC2_OUT * sizeof(float)));
    
    // FC3 layer
    cudaCheck(cudaMalloc(&m_w_fc3, fc3_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&v_w_fc3, fc3_w_size * sizeof(float)));
    cudaCheck(cudaMalloc(&m_b_fc3, NUM_CLASSES * sizeof(float)));
    cudaCheck(cudaMalloc(&v_b_fc3, NUM_CLASSES * sizeof(float)));
    
    // Initialize ADAM state buffers to zero
    cudaCheck(cudaMemset(m_w_c1, 0, conv1_w_size * sizeof(float)));
    cudaCheck(cudaMemset(v_w_c1, 0, conv1_w_size * sizeof(float)));
    cudaCheck(cudaMemset(m_b_c1, 0, C1_FILTERS * sizeof(float)));
    cudaCheck(cudaMemset(v_b_c1, 0, C1_FILTERS * sizeof(float)));
    
    cudaCheck(cudaMemset(m_w_c2, 0, conv2_w_size * sizeof(float)));
    cudaCheck(cudaMemset(v_w_c2, 0, conv2_w_size * sizeof(float)));
    cudaCheck(cudaMemset(m_b_c2, 0, C2_FILTERS * sizeof(float)));
    cudaCheck(cudaMemset(v_b_c2, 0, C2_FILTERS * sizeof(float)));
    
    cudaCheck(cudaMemset(m_w_fc1, 0, fc1_w_size * sizeof(float)));
    cudaCheck(cudaMemset(v_w_fc1, 0, fc1_w_size * sizeof(float)));
    cudaCheck(cudaMemset(m_b_fc1, 0, FC1_OUT * sizeof(float)));
    cudaCheck(cudaMemset(v_b_fc1, 0, FC1_OUT * sizeof(float)));
    
    cudaCheck(cudaMemset(m_w_fc2, 0, fc2_w_size * sizeof(float)));
    cudaCheck(cudaMemset(v_w_fc2, 0, fc2_w_size * sizeof(float)));
    cudaCheck(cudaMemset(m_b_fc2, 0, FC2_OUT * sizeof(float)));
    cudaCheck(cudaMemset(v_b_fc2, 0, FC2_OUT * sizeof(float)));
    
    cudaCheck(cudaMemset(m_w_fc3, 0, fc3_w_size * sizeof(float)));
    cudaCheck(cudaMemset(v_w_fc3, 0, fc3_w_size * sizeof(float)));
    cudaCheck(cudaMemset(m_b_fc3, 0, NUM_CLASSES * sizeof(float)));
    cudaCheck(cudaMemset(v_b_fc3, 0, NUM_CLASSES * sizeof(float)));

    cudaCheck(cudaMemcpy(d_w_c1, h_w_c1, conv1_w_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b_c1, h_b_c1, C1_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w_c2, h_w_c2, conv2_w_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b_c2, h_b_c2, C2_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w_fc1, h_w_fc1, fc1_w_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b_fc1, h_b_fc1, FC1_OUT * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w_fc2, h_w_fc2, fc2_w_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b_fc2, h_b_fc2, FC2_OUT * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w_fc3, h_w_fc3, fc3_w_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b_fc3, h_b_fc3, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const char *train_files[NUM_TRAIN_BATCHES] = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"};

    // Load test batch before training for per-epoch evaluation
    fprintf(output_file, "Loading test batch for per-epoch evaluation...\n");
    fprintf(stdout, "Loading test batch for per-epoch evaluation...\n");
    load_cifar10_batch("test_batch.bin", h_test_images, h_test_labels, num_images);

    fprintf(output_file, "Starting Training Loop...\n");
    fprintf(stdout, "Starting Training Loop...\n");
    cudaEventRecord(start);

    // Pre-allocate test batch data
    float *h_probs_eval = (float *)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        for (int file_idx = 0; file_idx < NUM_TRAIN_BATCHES; file_idx++)
        {
            fprintf(output_file, "Epoch %d/%d - Training on %s\n", epoch + 1, EPOCHS, train_files[file_idx]);
            fprintf(stdout, "Epoch %d/%d - Training on %s\n", epoch + 1, EPOCHS, train_files[file_idx]);
            load_cifar10_batch(train_files[file_idx], h_batch_images, h_batch_labels, num_images);
            cudaCheck(cudaMemcpy(d_batch_images, h_batch_images, img_mem_size, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(d_batch_labels, h_batch_labels, label_mem_size, cudaMemcpyHostToDevice));

            int num_batches = (num_images + BATCH_SIZE - 1) / BATCH_SIZE;
            for (int batch = 0; batch < num_batches; batch++)
            {
                int current_batch_size = num_images - batch * BATCH_SIZE;
                if (current_batch_size > BATCH_SIZE)
                    current_batch_size = BATCH_SIZE;
                int image_offset = batch * BATCH_SIZE * IMG_CHANNELS * IMG_WIDTH * IMG_HEIGHT;
                float *d_batch_images_ptr = d_batch_images + image_offset;
                int label_offset = batch * BATCH_SIZE;
                int *d_batch_labels_ptr = d_batch_labels + label_offset;

                // ==========================================
                // FORWARD PASS
                // ==========================================

                int out_w = 28, out_h = 28;
                dim3 conv_block(TILE_WIDTH, TILE_WIDTH);
                dim3 conv_grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
                               (out_h + TILE_WIDTH - 1) / TILE_WIDTH,
                               current_batch_size * C1_FILTERS);
                size_t shared_mem_size = (TILE_WIDTH + C1_SIZE - 1) * (TILE_WIDTH + C1_SIZE - 1) * IMG_CHANNELS * sizeof(float);
                conv1_forward_kernel<<<conv_grid, conv_block, shared_mem_size>>>(
                    d_batch_images_ptr, d_w_c1, d_b_c1, d_conv1_out,
                    current_batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
                    out_w, out_h, C1_FILTERS, C1_SIZE);
                cudaCheck(cudaDeviceSynchronize());

                dim3 pool_block(POOL_SIZE, POOL_SIZE);
                dim3 pool_grid((14 + pool_block.x - 1) / pool_block.x,
                               (14 + pool_block.y - 1) / pool_block.y,
                               current_batch_size * C1_FILTERS);
                max_pool_forward_kernel<<<pool_grid, pool_block>>>(
                    d_conv1_out, d_p1_out,
                    current_batch_size, 28, 28, C1_FILTERS,
                    POOL_SIZE, POOL_STRIDE);
                cudaCheck(cudaDeviceSynchronize());

                dim3 conv2_block(TILE_WIDTH, TILE_WIDTH);
                dim3 conv2_grid((10 + TILE_WIDTH - 1) / TILE_WIDTH,
                                (10 + TILE_WIDTH - 1) / TILE_WIDTH,
                                current_batch_size * C2_FILTERS);
                conv2_forward_kernel<<<conv2_grid, conv2_block>>>(
                    d_p1_out, d_w_c2, d_b_c2, d_conv2_out,
                    current_batch_size, 14, 14, C1_FILTERS,
                    10, 10, C2_FILTERS, C2_SIZE);
                cudaCheck(cudaDeviceSynchronize());

                dim3 pool2_block(POOL_SIZE, POOL_SIZE);
                dim3 pool2_grid((5 + pool2_block.x - 1) / pool2_block.x,
                                (5 + pool2_block.y - 1) / pool2_block.y,
                                current_batch_size * C2_FILTERS);
                max_pool_forward_kernel<<<pool2_grid, pool2_block>>>(
                    d_conv2_out, d_p2_out,
                    current_batch_size, 10, 10, C2_FILTERS,
                    POOL_SIZE, POOL_STRIDE);
                cudaCheck(cudaDeviceSynchronize());

                dim3 fc1_block(128, 1);
                dim3 fc1_grid((FC1_OUT + fc1_block.x - 1) / fc1_block.x, current_batch_size);
                fc_forward_kernel<<<fc1_grid, fc1_block>>>(
                    d_p2_out, d_w_fc1, d_b_fc1, d_fc1_out,
                    current_batch_size, p2_out_size, FC1_OUT, true);
                cudaCheck(cudaDeviceSynchronize());

                dim3 fc2_block(128, 1);
                dim3 fc2_grid((FC2_OUT + fc2_block.x - 1) / fc2_block.x, current_batch_size);
                fc_forward_kernel<<<fc2_grid, fc2_block>>>(
                    d_fc1_out, d_w_fc2, d_b_fc2, d_fc2_out,
                    current_batch_size, FC1_OUT, FC2_OUT, true);
                cudaCheck(cudaDeviceSynchronize());

                dim3 fc3_block(128, 1);
                dim3 fc3_grid((NUM_CLASSES + fc3_block.x - 1) / fc3_block.x, current_batch_size);
                fc_forward_kernel<<<fc3_grid, fc3_block>>>(
                    d_fc2_out, d_w_fc3, d_b_fc3, d_fc3_out,
                    current_batch_size, FC2_OUT, NUM_CLASSES, false);
                cudaCheck(cudaDeviceSynchronize());

                dim3 loss_block(128);
                dim3 loss_grid((current_batch_size + loss_block.x - 1) / loss_block.x);
                softmax_cross_entropy_kernel<<<loss_grid, loss_block>>>(
                    d_fc3_out, d_batch_labels_ptr, d_probs, d_loss,
                    current_batch_size, NUM_CLASSES);
                cudaCheck(cudaDeviceSynchronize());

                // ==========================================
                // BACKWARD PASS
                // ==========================================

                dim3 err_block(128);
                dim3 err_grid((current_batch_size + err_block.x - 1) / err_block.x);
                output_error_kernel<<<err_grid, err_block>>>(
                    d_probs, d_batch_labels_ptr, d_d_fc3,
                    current_batch_size, NUM_CLASSES);
                cudaCheck(cudaDeviceSynchronize());

                dim3 back_fc3_grid((FC2_OUT + 127) / 128, current_batch_size);
                fc_backward_kernel<<<back_fc3_grid, 128>>>(d_d_fc3, d_w_fc3, d_d_fc2, d_fc2_out, current_batch_size, FC2_OUT, NUM_CLASSES, true);
                cudaCheck(cudaDeviceSynchronize());

                dim3 back_fc2_grid((FC1_OUT + 127) / 128, current_batch_size);
                fc_backward_kernel<<<back_fc2_grid, 128>>>(d_d_fc2, d_w_fc2, d_d_fc1, d_fc1_out, current_batch_size, FC1_OUT, FC2_OUT, true);
                cudaCheck(cudaDeviceSynchronize());

                dim3 back_fc1_grid((p2_out_size + 127) / 128, current_batch_size);
                fc_backward_kernel<<<back_fc1_grid, 128>>>(d_d_fc1, d_w_fc1, d_d_p2, d_p2_out, current_batch_size, p2_out_size, FC1_OUT, false);
                cudaCheck(cudaDeviceSynchronize());

                cudaCheck(cudaMemset(d_d_conv2, 0, current_batch_size * conv2_out_size * sizeof(float)));
                dim3 pool2_back_block(POOL_SIZE, POOL_SIZE);
                dim3 pool2_back_grid((5 + pool2_back_block.x - 1) / pool2_back_block.x,
                                     (5 + pool2_back_block.y - 1) / pool2_back_block.y, current_batch_size * C2_FILTERS);
                max_pool_backward_kernel<<<pool2_back_grid, pool2_back_block>>>(d_d_p2, d_conv2_out, d_d_conv2, current_batch_size, 10, 10, C2_FILTERS, POOL_SIZE, POOL_STRIDE, true);
                cudaCheck(cudaDeviceSynchronize());

                dim3 conv2_back_block(16, 16);
                dim3 conv2_back_grid((14 + 15) / 16, (14 + 15) / 16, current_batch_size * C1_FILTERS);
                conv_backward_kernel<<<conv2_back_grid, conv2_back_block>>>(d_d_conv2, d_w_c2, d_d_p1, current_batch_size, 14, 14, C1_FILTERS, 10, 10, C2_FILTERS, C2_SIZE);
                cudaCheck(cudaDeviceSynchronize());

                cudaCheck(cudaMemset(d_d_conv1, 0, current_batch_size * conv1_out_size * sizeof(float)));
                dim3 pool1_back_block(POOL_SIZE, POOL_SIZE);
                dim3 pool1_back_grid((14 + pool1_back_block.x - 1) / pool1_back_block.x,
                                     (14 + pool1_back_block.y - 1) / pool1_back_block.y, current_batch_size * C1_FILTERS);
                max_pool_backward_kernel<<<pool1_back_grid, pool1_back_block>>>(d_d_p1, d_conv1_out, d_d_conv1, current_batch_size, 28, 28, C1_FILTERS, POOL_SIZE, POOL_STRIDE, true);
                cudaCheck(cudaDeviceSynchronize());

                // ==========================================
                // GRADIENTS CALCULATION
                // ==========================================

                dim3 grad_fc3_block(128);
                dim3 grad_fc3_grid((fc3_w_size + grad_fc3_block.x - 1) / grad_fc3_block.x);
                fc_weight_grad_kernel<<<grad_fc3_grid, grad_fc3_block>>>(
                    d_fc2_out, d_d_fc3, d_d_w_fc3,
                    current_batch_size, FC2_OUT, NUM_CLASSES);
                cudaCheck(cudaDeviceSynchronize());

                dim3 bias_fc3_block(128);
                dim3 bias_fc3_grid((NUM_CLASSES + bias_fc3_block.x - 1) / bias_fc3_block.x);
                bias_grad_kernel<<<bias_fc3_grid, bias_fc3_block>>>(
                    d_d_fc3, d_d_b_fc3,
                    current_batch_size, NUM_CLASSES);
                cudaCheck(cudaDeviceSynchronize());

                dim3 grad_fc2_block(128);
                dim3 grad_fc2_grid((fc2_w_size + grad_fc2_block.x - 1) / grad_fc2_block.x);
                fc_weight_grad_kernel<<<grad_fc2_grid, grad_fc2_block>>>(
                    d_fc1_out, d_d_fc2, d_d_w_fc2,
                    current_batch_size, FC1_OUT, FC2_OUT);
                cudaCheck(cudaDeviceSynchronize());

                dim3 bias_fc2_block(128);
                dim3 bias_fc2_grid((FC2_OUT + bias_fc2_block.x - 1) / bias_fc2_block.x);
                bias_grad_kernel<<<bias_fc2_grid, bias_fc2_block>>>(
                    d_d_fc2, d_d_b_fc2,
                    current_batch_size, FC2_OUT);
                cudaCheck(cudaDeviceSynchronize());

                dim3 grad_fc1_block(128);
                dim3 grad_fc1_grid((fc1_w_size + grad_fc1_block.x - 1) / grad_fc1_block.x);
                fc_weight_grad_kernel<<<grad_fc1_grid, grad_fc1_block>>>(
                    d_p2_out, d_d_fc1, d_d_w_fc1,
                    current_batch_size, p2_out_size, FC1_OUT);
                cudaCheck(cudaDeviceSynchronize());

                dim3 bias_fc1_block(128);
                dim3 bias_fc1_grid((FC1_OUT + bias_fc1_block.x - 1) / bias_fc1_block.x);
                bias_grad_kernel<<<bias_fc1_grid, bias_fc1_block>>>(
                    d_d_fc1, d_d_b_fc1,
                    current_batch_size, FC1_OUT);
                cudaCheck(cudaDeviceSynchronize());

                // FIXED LAUNCH GRIDS FOR CONVOLUTION GRADIENTS
                dim3 grad_c2_block(C2_SIZE, C2_SIZE);      // 5x5
                dim3 grad_c2_grid(C1_FILTERS, C2_FILTERS); // 6, 16
                conv_weight_grad_kernel<<<grad_c2_grid, grad_c2_block>>>(
                    d_p1_out, d_d_conv2, d_d_w_c2,
                    current_batch_size, 14, 14, C1_FILTERS,
                    10, 10, C2_FILTERS, C2_SIZE);
                cudaCheck(cudaDeviceSynchronize());

                dim3 bias_c2_block(128);
                dim3 bias_c2_grid((C2_FILTERS + bias_c2_block.x - 1) / bias_c2_block.x);
                conv_bias_grad_kernel<<<bias_c2_grid, bias_c2_block>>>(
                    d_d_conv2, d_d_b_c2,
                    current_batch_size, 10, 10, C2_FILTERS);
                cudaCheck(cudaDeviceSynchronize());

                dim3 grad_c1_block(C1_SIZE, C1_SIZE);        // 5x5
                dim3 grad_c1_grid(IMG_CHANNELS, C1_FILTERS); // 3, 6
                conv_weight_grad_kernel<<<grad_c1_grid, grad_c1_block>>>(
                    d_batch_images_ptr, d_d_conv1, d_d_w_c1,
                    current_batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
                    28, 28, C1_FILTERS, C1_SIZE);
                cudaCheck(cudaDeviceSynchronize());

                dim3 bias_c1_block(128);
                dim3 bias_c1_grid((C1_FILTERS + bias_c1_block.x - 1) / bias_c1_block.x);
                conv_bias_grad_kernel<<<bias_c1_grid, bias_c1_block>>>(
                    d_d_conv1, d_d_b_c1,
                    current_batch_size, 28, 28, C1_FILTERS);
                cudaCheck(cudaDeviceSynchronize());

                // ==========================================
                // WEIGHT UPDATES - ADAM Optimizer
                // ==========================================
                
                // Increment ADAM iteration counter before update
                adam_t++;

                int update_block = 256;
                int update_grid = (fc3_w_size + update_block - 1) / update_block;
                adam_update_weights_kernel<<<update_grid, update_block>>>(
                    d_w_fc3, d_d_w_fc3, d_b_fc3, d_d_b_fc3,
                    m_w_fc3, v_w_fc3, m_b_fc3, v_b_fc3,
                    fc3_w_size, NUM_CLASSES, ADAM_LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, adam_t, current_batch_size);
                cudaCheck(cudaDeviceSynchronize());

                update_grid = (fc2_w_size + update_block - 1) / update_block;
                adam_update_weights_kernel<<<update_grid, update_block>>>(
                    d_w_fc2, d_d_w_fc2, d_b_fc2, d_d_b_fc2,
                    m_w_fc2, v_w_fc2, m_b_fc2, v_b_fc2,
                    fc2_w_size, FC2_OUT, ADAM_LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, adam_t, current_batch_size);
                cudaCheck(cudaDeviceSynchronize());

                update_grid = (fc1_w_size + update_block - 1) / update_block;
                adam_update_weights_kernel<<<update_grid, update_block>>>(
                    d_w_fc1, d_d_w_fc1, d_b_fc1, d_d_b_fc1,
                    m_w_fc1, v_w_fc1, m_b_fc1, v_b_fc1,
                    fc1_w_size, FC1_OUT, ADAM_LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, adam_t, current_batch_size);
                cudaCheck(cudaDeviceSynchronize());

                update_grid = (conv2_w_size + update_block - 1) / update_block;
                adam_update_weights_kernel<<<update_grid, update_block>>>(
                    d_w_c2, d_d_w_c2, d_b_c2, d_d_b_c2,
                    m_w_c2, v_w_c2, m_b_c2, v_b_c2,
                    conv2_w_size, C2_FILTERS, ADAM_LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, adam_t, current_batch_size);
                cudaCheck(cudaDeviceSynchronize());

                update_grid = (conv1_w_size + update_block - 1) / update_block;
                adam_update_weights_kernel<<<update_grid, update_block>>>(
                    d_w_c1, d_d_w_c1, d_b_c1, d_d_b_c1,
                    m_w_c1, v_w_c1, m_b_c1, v_b_c1,
                    conv1_w_size, C1_FILTERS, ADAM_LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, adam_t, current_batch_size);
                cudaCheck(cudaDeviceSynchronize());
            }
        }
        
        // ==========================================
        // EPOCH EVALUATION ON TEST SET
        // ==========================================
        cudaCheck(cudaMemcpy(d_batch_images, h_test_images, img_mem_size, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_batch_labels, h_test_labels, label_mem_size, cudaMemcpyHostToDevice));
        
        int epoch_correct = 0;
        int test_batches = (num_images + BATCH_SIZE - 1) / BATCH_SIZE;
        
        for (int batch = 0; batch < test_batches; batch++)
        {
            int current_batch_size = num_images - batch * BATCH_SIZE;
            if (current_batch_size > BATCH_SIZE)
                current_batch_size = BATCH_SIZE;
            int image_offset = batch * BATCH_SIZE * IMG_CHANNELS * IMG_WIDTH * IMG_HEIGHT;
            float *d_batch_images_ptr = d_batch_images + image_offset;
            int label_offset = batch * BATCH_SIZE;
            int *h_batch_labels_ptr = h_test_labels + label_offset;

            int out_w = 28, out_h = 28;
            dim3 conv_block(TILE_WIDTH, TILE_WIDTH);
            dim3 conv_grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
                           (out_h + TILE_WIDTH - 1) / TILE_WIDTH,
                           current_batch_size * C1_FILTERS);
            size_t shared_mem_size = (TILE_WIDTH + C1_SIZE - 1) * (TILE_WIDTH + C1_SIZE - 1) * IMG_CHANNELS * sizeof(float);
            conv1_forward_kernel<<<conv_grid, conv_block, shared_mem_size>>>(
                d_batch_images_ptr, d_w_c1, d_b_c1, d_conv1_out,
                current_batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
                out_w, out_h, C1_FILTERS, C1_SIZE);
            cudaCheck(cudaDeviceSynchronize());

            dim3 pool_block(POOL_SIZE, POOL_SIZE);
            dim3 pool_grid((14 + pool_block.x - 1) / pool_block.x, (14 + pool_block.y - 1) / pool_block.y, current_batch_size * C1_FILTERS);
            max_pool_forward_kernel<<<pool_grid, pool_block>>>(
                d_conv1_out, d_p1_out, current_batch_size, 28, 28, C1_FILTERS, POOL_SIZE, POOL_STRIDE);
            cudaCheck(cudaDeviceSynchronize());

            dim3 conv2_block(TILE_WIDTH, TILE_WIDTH);
            dim3 conv2_grid((10 + TILE_WIDTH - 1) / TILE_WIDTH, (10 + TILE_WIDTH - 1) / TILE_WIDTH, current_batch_size * C2_FILTERS);
            conv2_forward_kernel<<<conv2_grid, conv2_block>>>(
                d_p1_out, d_w_c2, d_b_c2, d_conv2_out, current_batch_size, 14, 14, C1_FILTERS, 10, 10, C2_FILTERS, C2_SIZE);
            cudaCheck(cudaDeviceSynchronize());

            dim3 pool2_block(POOL_SIZE, POOL_SIZE);
            dim3 pool2_grid((5 + pool2_block.x - 1) / pool2_block.x, (5 + pool2_block.y - 1) / pool2_block.y, current_batch_size * C2_FILTERS);
            max_pool_forward_kernel<<<pool2_grid, pool2_block>>>(
                d_conv2_out, d_p2_out, current_batch_size, 10, 10, C2_FILTERS, POOL_SIZE, POOL_STRIDE);
            cudaCheck(cudaDeviceSynchronize());

            dim3 fc1_block(128, 1);
            dim3 fc1_grid((FC1_OUT + fc1_block.x - 1) / fc1_block.x, current_batch_size);
            fc_forward_kernel<<<fc1_grid, fc1_block>>>(
                d_p2_out, d_w_fc1, d_b_fc1, d_fc1_out, current_batch_size, p2_out_size, FC1_OUT, true);
            cudaCheck(cudaDeviceSynchronize());

            dim3 fc2_block(128, 1);
            dim3 fc2_grid((FC2_OUT + fc2_block.x - 1) / fc2_block.x, current_batch_size);
            fc_forward_kernel<<<fc2_grid, fc2_block>>>(
                d_fc1_out, d_w_fc2, d_b_fc2, d_fc2_out, current_batch_size, FC1_OUT, FC2_OUT, true);
            cudaCheck(cudaDeviceSynchronize());

            dim3 fc3_block(128, 1);
            dim3 fc3_grid((NUM_CLASSES + fc3_block.x - 1) / fc3_block.x, current_batch_size);
            fc_forward_kernel<<<fc3_grid, fc3_block>>>(
                d_fc2_out, d_w_fc3, d_b_fc3, d_fc3_out, current_batch_size, FC2_OUT, NUM_CLASSES, false);
            cudaCheck(cudaDeviceSynchronize());

            dim3 loss_block(128);
            dim3 loss_grid((current_batch_size + loss_block.x - 1) / loss_block.x);
            softmax_cross_entropy_kernel<<<loss_grid, loss_block>>>(
                d_fc3_out, d_batch_labels + label_offset, d_probs, d_loss, current_batch_size, NUM_CLASSES);
            cudaCheck(cudaDeviceSynchronize());

            cudaCheck(cudaMemcpy(h_probs_eval, d_probs, current_batch_size * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < current_batch_size; i++)
            {
                int predicted = 0;
                float max_val = h_probs_eval[i * NUM_CLASSES];
                for (int c = 1; c < NUM_CLASSES; c++)
                {
                    float val = h_probs_eval[i * NUM_CLASSES + c];
                    if (val > max_val)
                    {
                        max_val = val;
                        predicted = c;
                    }
                }
                if (predicted == h_batch_labels_ptr[i])
                {
                    epoch_correct++;
                }
            }
        }
        
        float epoch_accuracy = (float)epoch_correct / (float)num_images * 100.0f;
        fprintf(output_file, "Epoch %d/%d - Test Accuracy: %.2f%% (%d / %d)\n", epoch + 1, EPOCHS, epoch_accuracy, epoch_correct, num_images);
        fprintf(stdout, "Epoch %d/%d - Test Accuracy: %.2f%% (%d / %d)\n", epoch + 1, EPOCHS, epoch_accuracy, epoch_correct, num_images);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float train_milliseconds = 0;
    cudaEventElapsedTime(&train_milliseconds, start, stop);

    fprintf(output_file, "==========================================\n");
    fprintf(output_file, "Training complete in %.2f ms\n", train_milliseconds);
    fprintf(output_file, "==========================================\n");
    fprintf(stdout, "==========================================\n");
    fprintf(stdout, "Training complete in %.2f ms\n", train_milliseconds);
    fprintf(stdout, "==========================================\n");

    fprintf(output_file, "Loading test batch and running inference...\n");
    fprintf(stdout, "Loading test batch and running inference...\n");
    load_cifar10_batch("test_batch.bin", h_test_images, h_test_labels, num_images);
    cudaCheck(cudaMemcpy(d_batch_images, h_test_images, img_mem_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_batch_labels, h_test_labels, label_mem_size, cudaMemcpyHostToDevice));

    int total_correct = 0;
    float *h_probs = (float *)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));
    int test_batches = (num_images + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int batch = 0; batch < test_batches; batch++)
    {
        int current_batch_size = num_images - batch * BATCH_SIZE;
        if (current_batch_size > BATCH_SIZE)
            current_batch_size = BATCH_SIZE;
        int image_offset = batch * BATCH_SIZE * IMG_CHANNELS * IMG_WIDTH * IMG_HEIGHT;
        float *d_batch_images_ptr = d_batch_images + image_offset;
        int label_offset = batch * BATCH_SIZE;
        int *h_batch_labels_ptr = h_test_labels + label_offset;

        int out_w = 28, out_h = 28;
        dim3 conv_block(TILE_WIDTH, TILE_WIDTH);
        dim3 conv_grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
                       (out_h + TILE_WIDTH - 1) / TILE_WIDTH,
                       current_batch_size * C1_FILTERS);
        size_t shared_mem_size = (TILE_WIDTH + C1_SIZE - 1) * (TILE_WIDTH + C1_SIZE - 1) * IMG_CHANNELS * sizeof(float);
        conv1_forward_kernel<<<conv_grid, conv_block, shared_mem_size>>>(
            d_batch_images_ptr, d_w_c1, d_b_c1, d_conv1_out,
            current_batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS,
            out_w, out_h, C1_FILTERS, C1_SIZE);

        dim3 pool_block(POOL_SIZE, POOL_SIZE);
        dim3 pool_grid((14 + pool_block.x - 1) / pool_block.x, (14 + pool_block.y - 1) / pool_block.y, current_batch_size * C1_FILTERS);
        max_pool_forward_kernel<<<pool_grid, pool_block>>>(
            d_conv1_out, d_p1_out, current_batch_size, 28, 28, C1_FILTERS, POOL_SIZE, POOL_STRIDE);

        dim3 conv2_block(TILE_WIDTH, TILE_WIDTH);
        dim3 conv2_grid((10 + TILE_WIDTH - 1) / TILE_WIDTH, (10 + TILE_WIDTH - 1) / TILE_WIDTH, current_batch_size * C2_FILTERS);
        conv2_forward_kernel<<<conv2_grid, conv2_block>>>(
            d_p1_out, d_w_c2, d_b_c2, d_conv2_out, current_batch_size, 14, 14, C1_FILTERS, 10, 10, C2_FILTERS, C2_SIZE);

        dim3 pool2_block(POOL_SIZE, POOL_SIZE);
        dim3 pool2_grid((5 + pool2_block.x - 1) / pool2_block.x, (5 + pool2_block.y - 1) / pool2_block.y, current_batch_size * C2_FILTERS);
        max_pool_forward_kernel<<<pool2_grid, pool2_block>>>(
            d_conv2_out, d_p2_out, current_batch_size, 10, 10, C2_FILTERS, POOL_SIZE, POOL_STRIDE);

        dim3 fc1_block(128, 1);
        dim3 fc1_grid((FC1_OUT + fc1_block.x - 1) / fc1_block.x, current_batch_size);
        fc_forward_kernel<<<fc1_grid, fc1_block>>>(
            d_p2_out, d_w_fc1, d_b_fc1, d_fc1_out, current_batch_size, p2_out_size, FC1_OUT, true);

        dim3 fc2_block(128, 1);
        dim3 fc2_grid((FC2_OUT + fc2_block.x - 1) / fc2_block.x, current_batch_size);
        fc_forward_kernel<<<fc2_grid, fc2_block>>>(
            d_fc1_out, d_w_fc2, d_b_fc2, d_fc2_out, current_batch_size, FC1_OUT, FC2_OUT, true);

        dim3 fc3_block(128, 1);
        dim3 fc3_grid((NUM_CLASSES + fc3_block.x - 1) / fc3_block.x, current_batch_size);
        fc_forward_kernel<<<fc3_grid, fc3_block>>>(
            d_fc2_out, d_w_fc3, d_b_fc3, d_fc3_out, current_batch_size, FC2_OUT, NUM_CLASSES, false);

        dim3 loss_block(128);
        dim3 loss_grid((current_batch_size + loss_block.x - 1) / loss_block.x);
        softmax_cross_entropy_kernel<<<loss_grid, loss_block>>>(
            d_fc3_out, d_batch_labels + label_offset, d_probs, d_loss, current_batch_size, NUM_CLASSES);

        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(h_probs, d_probs, current_batch_size * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < current_batch_size; i++)
        {
            int predicted = 0;
            float max_val = h_probs[i * NUM_CLASSES];
            for (int c = 1; c < NUM_CLASSES; c++)
            {
                float val = h_probs[i * NUM_CLASSES + c];
                if (val > max_val)
                {
                    max_val = val;
                    predicted = c;
                }
            }
            if (predicted == h_batch_labels_ptr[i])
            {
                total_correct++;
            }
        }
    }

    float accuracy = (float)total_correct / (float)num_images * 100.0f;
    fprintf(output_file, "Test Accuracy: %.2f%% (%d / %d)\n", accuracy, total_correct, num_images);
    fprintf(stdout, "Test Accuracy: %.2f%% (%d / %d)\n", accuracy, total_correct, num_images);

    // Cleanup Device Memory
    cudaFree(d_batch_images);
    cudaFree(d_batch_labels);

    cudaFree(d_w_c1);
    cudaFree(d_b_c1);
    cudaFree(d_w_c2);
    cudaFree(d_b_c2);
    cudaFree(d_w_fc1);
    cudaFree(d_b_fc1);
    cudaFree(d_w_fc2);
    cudaFree(d_b_fc2);
    cudaFree(d_w_fc3);
    cudaFree(d_b_fc3);

    cudaFree(d_conv1_out);
    cudaFree(d_p1_out);
    cudaFree(d_conv2_out);
    cudaFree(d_p2_out);
    cudaFree(d_fc1_out);
    cudaFree(d_fc2_out);
    cudaFree(d_fc3_out);

    cudaFree(d_probs);
    cudaFree(d_loss);

    cudaFree(d_d_fc3);
    cudaFree(d_d_fc2);
    cudaFree(d_d_fc1);
    cudaFree(d_d_p2);
    cudaFree(d_d_conv2);
    cudaFree(d_d_p1);
    cudaFree(d_d_conv1);

    cudaFree(d_d_w_c1);
    cudaFree(d_d_b_c1);
    cudaFree(d_d_w_c2);
    cudaFree(d_d_b_c2);
    cudaFree(d_d_w_fc1);
    cudaFree(d_d_b_fc1);
    cudaFree(d_d_w_fc2);
    cudaFree(d_d_b_fc2);
    cudaFree(d_d_w_fc3);
    cudaFree(d_d_b_fc3);

    // Free ADAM optimizer state buffers
    cudaFree(m_w_c1);
    cudaFree(v_w_c1);
    cudaFree(m_b_c1);
    cudaFree(v_b_c1);
    
    cudaFree(m_w_c2);
    cudaFree(v_w_c2);
    cudaFree(m_b_c2);
    cudaFree(v_b_c2);
    
    cudaFree(m_w_fc1);
    cudaFree(v_w_fc1);
    cudaFree(m_b_fc1);
    cudaFree(v_b_fc1);
    
    cudaFree(m_w_fc2);
    cudaFree(v_w_fc2);
    cudaFree(m_b_fc2);
    cudaFree(v_b_fc2);
    
    cudaFree(m_w_fc3);
    cudaFree(v_w_fc3);
    cudaFree(m_b_fc3);
    cudaFree(v_b_fc3);

    // Cleanup Host Memory
    free(h_batch_images);
    free(h_batch_labels);
    free(h_test_images);
    free(h_test_labels);

    free(h_w_c1);
    free(h_b_c1);
    free(h_w_c2);
    free(h_b_c2);
    free(h_w_fc1);
    free(h_b_fc1);
    free(h_w_fc2);
    free(h_b_fc2);
    free(h_w_fc3);
    free(h_b_fc3);

    free(h_probs);

    fclose(output_file);
    return 0;
}