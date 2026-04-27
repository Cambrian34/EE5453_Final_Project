#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "network_config.h"

/**
 * Export trained LeNet-5 weights to C header file for MSP430 inference
 * 
 * Usage: ./export_weights
 * 
 * Reads: lenet5_weights.bin (binary checkpoint from training)
 * Writes: lenet5_weights.h (C header with static weight arrays)
 * 
 * Header file can be directly included in MSP430 C projects
 */

#define MAX_ARRAY_DISPLAY_SIZE 100  // Max values per line in header file

// Function to write float value to file with proper formatting
void write_float(FILE *f, float value, int index, int total_elements) {
    if (index % 10 == 0) {
        fprintf(f, "\n    ");
    }
    if (index == total_elements - 1) {
        fprintf(f, "%.8ef", value);
    } else {
        fprintf(f, "%.8ef, ", value);
    }
}

int main() {
    printf("==========================================================\n");
    printf("LeNet-5 Weight Export Utility for MSP430 Inference\n");
    printf("==========================================================\n\n");

    // Open binary checkpoint file
    FILE *bin_file = fopen("lenet5_weights.bin", "rb");
    if (bin_file == NULL) {
        fprintf(stderr, "Error: Could not open lenet5_weights.bin\n");
        fprintf(stderr, "Make sure to run training first: ./lenet_train\n");
        return 1;
    }

    printf("[*] Reading checkpoint file: lenet5_weights.bin\n");

    // Read metadata header
    int magic, version;
    fread(&magic, sizeof(int), 1, bin_file);
    fread(&version, sizeof(int), 1, bin_file);

    if (magic != 0x4C455435) {
        fprintf(stderr, "Error: Invalid checkpoint file (bad magic number)\n");
        fclose(bin_file);
        return 1;
    }

    printf("[*] Checkpoint version: %d\n", version);

    // Read layer dimensions
    int c1_filters, c2_filters, fc1_size, fc2_size, num_classes;
    fread(&c1_filters, sizeof(int), 1, bin_file);
    fread(&c2_filters, sizeof(int), 1, bin_file);
    fread(&fc1_size, sizeof(int), 1, bin_file);
    fread(&fc2_size, sizeof(int), 1, bin_file);
    fread(&num_classes, sizeof(int), 1, bin_file);

    printf("[*] Network architecture:\n");
    printf("    - Conv1: %d filters, 5x5 kernel, 3 input channels\n", c1_filters);
    printf("    - Conv2: %d filters, 5x5 kernel, %d input channels\n", c2_filters, c1_filters);
    printf("    - FC1: %d neurons\n", fc1_size);
    printf("    - FC2: %d neurons\n", fc2_size);
    printf("    - Output: %d classes\n", num_classes);

    // Calculate sizes
    int conv1_w_size = c1_filters * 3 * 5 * 5;
    int conv2_w_size = c2_filters * c1_filters * 5 * 5;
    int p2_out_size = c2_filters * 5 * 5;
    int fc1_w_size = fc1_size * p2_out_size;
    int fc2_w_size = fc2_size * fc1_size;
    int fc3_w_size = num_classes * fc2_size;

    printf("[*] Weight parameter counts:\n");
    printf("    - Conv1 weights: %d\n", conv1_w_size);
    printf("    - Conv1 biases: %d\n", c1_filters);
    printf("    - Conv2 weights: %d\n", conv2_w_size);
    printf("    - Conv2 biases: %d\n", c2_filters);
    printf("    - FC1 weights: %d\n", fc1_w_size);
    printf("    - FC1 biases: %d\n", fc1_size);
    printf("    - FC2 weights: %d\n", fc2_w_size);
    printf("    - FC2 biases: %d\n", fc2_size);
    printf("    - FC3 weights: %d\n", fc3_w_size);
    printf("    - FC3 biases: %d\n", num_classes);

    int total_params = conv1_w_size + c1_filters + conv2_w_size + c2_filters +
                       fc1_w_size + fc1_size + fc2_w_size + fc2_size +
                       fc3_w_size + num_classes;
    printf("[*] Total parameters: %d (%.2f KB in float32)\n", total_params,
           total_params * 4.0 / 1024.0);

    // Allocate memory for all weights
    printf("[*] Allocating memory for weights...\n");
    float *w_conv1 = (float *)malloc(conv1_w_size * sizeof(float));
    float *b_conv1 = (float *)malloc(c1_filters * sizeof(float));
    float *w_conv2 = (float *)malloc(conv2_w_size * sizeof(float));
    float *b_conv2 = (float *)malloc(c2_filters * sizeof(float));
    float *w_fc1 = (float *)malloc(fc1_w_size * sizeof(float));
    float *b_fc1 = (float *)malloc(fc1_size * sizeof(float));
    float *w_fc2 = (float *)malloc(fc2_w_size * sizeof(float));
    float *b_fc2 = (float *)malloc(fc2_size * sizeof(float));
    float *w_fc3 = (float *)malloc(fc3_w_size * sizeof(float));
    float *b_fc3 = (float *)malloc(num_classes * sizeof(float));

    if (!w_conv1 || !b_conv1 || !w_conv2 || !b_conv2 || !w_fc1 || !b_fc1 ||
        !w_fc2 || !b_fc2 || !w_fc3 || !b_fc3) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(bin_file);
        return 1;
    }

    // Read weights from checkpoint
    printf("[*] Reading weights from checkpoint...\n");
    fread(w_conv1, sizeof(float), conv1_w_size, bin_file);
    fread(b_conv1, sizeof(float), c1_filters, bin_file);
    fread(w_conv2, sizeof(float), conv2_w_size, bin_file);
    fread(b_conv2, sizeof(float), c2_filters, bin_file);
    fread(w_fc1, sizeof(float), fc1_w_size, bin_file);
    fread(b_fc1, sizeof(float), fc1_size, bin_file);
    fread(w_fc2, sizeof(float), fc2_w_size, bin_file);
    fread(b_fc2, sizeof(float), fc2_size, bin_file);
    fread(w_fc3, sizeof(float), fc3_w_size, bin_file);
    fread(b_fc3, sizeof(float), num_classes, bin_file);
    fclose(bin_file);

    // Open output header file
    printf("[*] Generating header file: lenet5_weights.h\n");
    FILE *header_file = fopen("lenet5_weights.h", "w");
    if (header_file == NULL) {
        fprintf(stderr, "Error: Could not open lenet5_weights.h for writing\n");
        return 1;
    }

    // Write header file
    fprintf(header_file, "/**\n");
    fprintf(header_file, " * Auto-generated LeNet-5 Weights Header\n");
    fprintf(header_file, " * Generated for MSP430 Inference Engine\n");
    fprintf(header_file, " * \n");
    fprintf(header_file, " * Network Architecture:\n");
    fprintf(header_file, " *   - Input: 32x32 RGB images (CIFAR-10)\n");
    fprintf(header_file, " *   - Conv1: %d filters, kernel 5x5\n", c1_filters);
    fprintf(header_file, " *   - Pool1: 2x2 max pooling\n");
    fprintf(header_file, " *   - Conv2: %d filters, kernel 5x5\n", c2_filters);
    fprintf(header_file, " *   - Pool2: 2x2 max pooling\n");
    fprintf(header_file, " *   - FC1: %d neurons, ReLU\n", fc1_size);
    fprintf(header_file, " *   - FC2: %d neurons, ReLU\n", fc2_size);
    fprintf(header_file, " *   - Output: %d classes, Softmax\n", num_classes);
    fprintf(header_file, " *\n");
    fprintf(header_file, " * Total Parameters: %d\n", total_params);
    fprintf(header_file, " * Memory Size (float32): %.2f KB\n", total_params * 4.0 / 1024.0);
    fprintf(header_file, " */\n\n");

    fprintf(header_file, "#ifndef LENET5_WEIGHTS_H\n");
    fprintf(header_file, "#define LENET5_WEIGHTS_H\n\n");

    fprintf(header_file, "#include <stdint.h>\n\n");

    // Architecture constants
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// Architecture Constants\n");
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "#define LENET_C1_FILTERS %d\n", c1_filters);
    fprintf(header_file, "#define LENET_C2_FILTERS %d\n", c2_filters);
    fprintf(header_file, "#define LENET_FC1_SIZE %d\n", fc1_size);
    fprintf(header_file, "#define LENET_FC2_SIZE %d\n", fc2_size);
    fprintf(header_file, "#define LENET_NUM_CLASSES %d\n\n", num_classes);

    // Weight parameter counts
    fprintf(header_file, "#define LENET_CONV1_W_SIZE %d\n", conv1_w_size);
    fprintf(header_file, "#define LENET_CONV1_B_SIZE %d\n", c1_filters);
    fprintf(header_file, "#define LENET_CONV2_W_SIZE %d\n", conv2_w_size);
    fprintf(header_file, "#define LENET_CONV2_B_SIZE %d\n", c2_filters);
    fprintf(header_file, "#define LENET_FC1_W_SIZE %d\n", fc1_w_size);
    fprintf(header_file, "#define LENET_FC1_B_SIZE %d\n", fc1_size);
    fprintf(header_file, "#define LENET_FC2_W_SIZE %d\n", fc2_w_size);
    fprintf(header_file, "#define LENET_FC2_B_SIZE %d\n", fc2_size);
    fprintf(header_file, "#define LENET_FC3_W_SIZE %d\n", fc3_w_size);
    fprintf(header_file, "#define LENET_FC3_B_SIZE %d\n\n", num_classes);

    // Conv1 Layer
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// Conv1 Layer: %d filters, 5x5 kernel, 3 input channels\n", c1_filters);
    fprintf(header_file, "// Size: %d weights + %d biases = %d parameters\n",
            conv1_w_size, c1_filters, conv1_w_size + c1_filters);
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "const float w_conv1[LENET_CONV1_W_SIZE] = {");
    for (int i = 0; i < conv1_w_size; i++) {
        write_float(header_file, w_conv1[i], i, conv1_w_size);
    }
    fprintf(header_file, "\n};\n\n");

    fprintf(header_file, "const float b_conv1[LENET_CONV1_B_SIZE] = {");
    for (int i = 0; i < c1_filters; i++) {
        write_float(header_file, b_conv1[i], i, c1_filters);
    }
    fprintf(header_file, "\n};\n\n");

    // Conv2 Layer
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// Conv2 Layer: %d filters, 5x5 kernel, %d input channels\n",
            c2_filters, c1_filters);
    fprintf(header_file, "// Size: %d weights + %d biases = %d parameters\n",
            conv2_w_size, c2_filters, conv2_w_size + c2_filters);
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "const float w_conv2[LENET_CONV2_W_SIZE] = {");
    for (int i = 0; i < conv2_w_size; i++) {
        write_float(header_file, w_conv2[i], i, conv2_w_size);
    }
    fprintf(header_file, "\n};\n\n");

    fprintf(header_file, "const float b_conv2[LENET_CONV2_B_SIZE] = {");
    for (int i = 0; i < c2_filters; i++) {
        write_float(header_file, b_conv2[i], i, c2_filters);
    }
    fprintf(header_file, "\n};\n\n");

    // FC1 Layer
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// FC1 Layer: %d neurons\n", fc1_size);
    fprintf(header_file, "// Size: %d weights + %d biases = %d parameters\n",
            fc1_w_size, fc1_size, fc1_w_size + fc1_size);
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "const float w_fc1[LENET_FC1_W_SIZE] = {");
    for (int i = 0; i < fc1_w_size; i++) {
        write_float(header_file, w_fc1[i], i, fc1_w_size);
    }
    fprintf(header_file, "\n};\n\n");

    fprintf(header_file, "const float b_fc1[LENET_FC1_B_SIZE] = {");
    for (int i = 0; i < fc1_size; i++) {
        write_float(header_file, b_fc1[i], i, fc1_size);
    }
    fprintf(header_file, "\n};\n\n");

    // FC2 Layer
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// FC2 Layer: %d neurons\n", fc2_size);
    fprintf(header_file, "// Size: %d weights + %d biases = %d parameters\n",
            fc2_w_size, fc2_size, fc2_w_size + fc2_size);
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "const float w_fc2[LENET_FC2_W_SIZE] = {");
    for (int i = 0; i < fc2_w_size; i++) {
        write_float(header_file, w_fc2[i], i, fc2_w_size);
    }
    fprintf(header_file, "\n};\n\n");

    fprintf(header_file, "const float b_fc2[LENET_FC2_B_SIZE] = {");
    for (int i = 0; i < fc2_size; i++) {
        write_float(header_file, b_fc2[i], i, fc2_size);
    }
    fprintf(header_file, "\n};\n\n");

    // FC3 Layer (Output)
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// FC3 Layer (Output): %d classes\n", num_classes);
    fprintf(header_file, "// Size: %d weights + %d biases = %d parameters\n",
            fc3_w_size, num_classes, fc3_w_size + num_classes);
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "const float w_fc3[LENET_FC3_W_SIZE] = {");
    for (int i = 0; i < fc3_w_size; i++) {
        write_float(header_file, w_fc3[i], i, fc3_w_size);
    }
    fprintf(header_file, "\n};\n\n");

    fprintf(header_file, "const float b_fc3[LENET_FC3_B_SIZE] = {");
    for (int i = 0; i < num_classes; i++) {
        write_float(header_file, b_fc3[i], i, num_classes);
    }
    fprintf(header_file, "\n};\n\n");

    // Summary
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// Weight Summary\n");
    fprintf(header_file, "// ==========================================================\n");
    fprintf(header_file, "// Total parameters: %d\n", total_params);
    fprintf(header_file, "// Memory (float32): %.2f KB\n", total_params * 4.0 / 1024.0);
    fprintf(header_file, "//\n");
    fprintf(header_file, "// Layer breakdown:\n");
    fprintf(header_file, "//   Conv1: %d params (%.2f KB)\n",
            conv1_w_size + c1_filters, (conv1_w_size + c1_filters) * 4.0 / 1024.0);
    fprintf(header_file, "//   Conv2: %d params (%.2f KB)\n",
            conv2_w_size + c2_filters, (conv2_w_size + c2_filters) * 4.0 / 1024.0);
    fprintf(header_file, "//   FC1: %d params (%.2f KB)\n",
            fc1_w_size + fc1_size, (fc1_w_size + fc1_size) * 4.0 / 1024.0);
    fprintf(header_file, "//   FC2: %d params (%.2f KB)\n",
            fc2_w_size + fc2_size, (fc2_w_size + fc2_size) * 4.0 / 1024.0);
    fprintf(header_file, "//   FC3: %d params (%.2f KB)\n",
            fc3_w_size + num_classes, (fc3_w_size + num_classes) * 4.0 / 1024.0);
    fprintf(header_file, "// ==========================================================\n\n");

    fprintf(header_file, "#endif // LENET5_WEIGHTS_H\n");

    fclose(header_file);

    // Free allocated memory
    free(w_conv1);
    free(b_conv1);
    free(w_conv2);
    free(b_conv2);
    free(w_fc1);
    free(b_fc1);
    free(w_fc2);
    free(b_fc2);
    free(w_fc3);
    free(b_fc3);

    printf("\n[+] Success! Generated lenet5_weights.h\n");
    printf("\nFile Statistics:\n");
    printf("    - Archive size: %.2f KB (float32 weights)\n", total_params * 4.0 / 1024.0);
    printf("    - Total data points: %d\n", total_params);
    printf("\nUsage:\n");
    printf("    1. Include in MSP430 project: #include \"lenet5_weights.h\"\n");
    printf("    2. Access weights: w_conv1[], b_conv1[], w_conv2[], etc.\n");
    printf("    3. Use architecture constants: LENET_C1_FILTERS, LENET_NUM_CLASSES, etc.\n");
    printf("    4. Compile with: msp430-gcc -std=c99 -c your_inference.c\n");
    printf("\n==========================================================\n");

    return 0;
}
