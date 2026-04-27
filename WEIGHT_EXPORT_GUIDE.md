# LeNet-5 Weight Export Guide for MSP430 Inference

## Overview

This guide explains how to export trained LeNet-5 weights from the GPU training pipeline and convert them into a C header file suitable for embedding in MSP430 microcontroller firmware for inference.

## Workflow

```
Training (lenet_cifar10)
    ↓
Save Checkpoint (lenet5_weights.bin)
    ↓
Export Utility (export_weights)
    ↓
Header File (lenet5_weights.h)
    ↓
MSP430 Inference Project
```

## Step 1: Train the Network

Compile and run the training program as usual:

```bash
nvcc lenet_cifar10.cu lenet_kernels.cu data_utils.cu cuda_utils.cu -o lenet_train
./lenet_train
```

**Output:**
- `output.txt` - Training log
- `lenet5_weights.bin` - Binary checkpoint (~2.2 MB)

The training program automatically saves the checkpoint after the final epoch, during cleanup phase.

## Step 2: Export Weights to Header File

Compile the export utility:

```bash
nvcc export_weights.cu data_utils.cu -o export_weights
```

Run the export utility:

```bash
./export_weights
```

**Output:**
- `lenet5_weights.h` - C header file with all weights (~3 MB text file)

### Export Process Details

The export utility:
1. Reads the binary checkpoint (`lenet5_weights.bin`)
2. Extracts all weights and biases for 5 layers:
   - Conv1: 18 filters × 3 channels × 5×5 kernel = 2,718 weights + 18 biases
   - Conv2: 48 filters × 18 channels × 5×5 kernel = 21,648 weights + 48 biases
   - FC1: 1,200 inputs → 360 neurons = 432,360 weights + 360 biases
   - FC2: 360 inputs → 252 neurons = 91,212 weights + 252 biases
   - FC3: 252 inputs → 10 classes = 2,530 weights + 10 biases
3. Generates C header file with:
   - Architecture constants (filter counts, layer sizes)
   - Static const float arrays for each weight/bias matrix
   - Extensive comments documenting the network structure
   - Memory usage summary

**Total Parameters:** ~550,000 weights (2.2 MB in float32 format)

## Step 3: Using Weights in MSP430 Project

### Header File Contents

The generated `lenet5_weights.h` provides:

```c
// Architecture constants
#define LENET_C1_FILTERS 18
#define LENET_C2_FILTERS 48
#define LENET_FC1_SIZE 360
#define LENET_FC2_SIZE 252
#define LENET_NUM_CLASSES 10

// Size constants
#define LENET_CONV1_W_SIZE 2718    // weights
#define LENET_CONV1_B_SIZE 18      // biases
// ... etc for all layers

// Weight arrays
const float w_conv1[LENET_CONV1_W_SIZE];
const float b_conv1[LENET_CONV1_B_SIZE];
const float w_conv2[LENET_CONV2_W_SIZE];
const float b_conv2[LENET_CONV2_B_SIZE];
const float w_fc1[LENET_FC1_W_SIZE];
const float b_fc1[LENET_FC1_B_SIZE];
const float w_fc2[LENET_FC2_W_SIZE];
const float b_fc2[LENET_FC2_B_SIZE];
const float w_fc3[LENET_FC3_W_SIZE];
const float b_fc3[LENET_FC3_B_SIZE];
```

### Including in MSP430 Code

1. **Copy the header file** to your MSP430 project directory

2. **Include in your inference code:**

```c
#include "lenet5_weights.h"

// Access weights anywhere in your code
void run_inference(float *input_image, float *output_confidences) {
    // Conv1 forward pass
    for (int f = 0; f < LENET_C1_FILTERS; f++) {
        // Use w_conv1[] and b_conv1[]
        float bias = b_conv1[f];
        // ... kernel computation ...
    }
    
    // FC1 forward pass
    for (int n = 0; n < LENET_FC1_SIZE; n++) {
        float sum = b_fc1[n];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * w_fc1[n * input_size + i];
        }
        // ... activation ...
    }
}
```

### Memory Considerations for MSP430

- **Total weight memory:** 2.2 MB (float32)
- **Flash storage:** Required for weights to persist
- **RAM overhead:** Implementation depends on inference batch size
- **Options if 2.2 MB is too large:**
  1. **Quantize weights** to int16 (reduces to 1.1 MB)
  2. **Quantize weights** to int8 (reduces to 550 KB)
  3. **Store on external flash** (SD card, serial flash)
  4. **Stream weights** into RAM in chunks during inference

### Compiling for MSP430

```bash
# Standard MSP430 GCC compilation
msp430-gcc -mmcu=msp430fr5969 -O3 -c inference_engine.c -o inference_engine.o

# Link with other objects
msp430-gcc -mmcu=msp430fr5969 inference_engine.o main.o -o firmware.elf

# Convert to hex for programming
msp430-objcopy -O ihex firmware.elf firmware.hex
```

## File Format Details

### Binary Checkpoint Format (`lenet5_weights.bin`)

```
[Header]
- Magic: 0x4C455435 (4 bytes) = "LET5"
- Version: 1 (4 bytes)

[Metadata]
- C1 filters (4 bytes)
- C2 filters (4 bytes)
- FC1 size (4 bytes)
- FC2 size (4 bytes)
- Num classes (4 bytes)

[Weights]
- w_conv1[conv1_w_size] (float32)
- b_conv1[c1_filters] (float32)
- w_conv2[conv2_w_size] (float32)
- b_conv2[c2_filters] (float32)
- w_fc1[fc1_w_size] (float32)
- b_fc1[fc1_size] (float32)
- w_fc2[fc2_w_size] (float32)
- b_fc2[fc2_size] (float32)
- w_fc3[fc3_w_size] (float32)
- b_fc3[num_classes] (float32)
```

### Header File Format (`lenet5_weights.h`)

```
[Comments]
- Network architecture description
- Parameter counts and memory usage

[Architecture Constants]
#define LENET_* values

[Size Constants]
#define LENET_*_SIZE values

[Weight Arrays]
const float w_layer[size] = { ... }
const float b_layer[size] = { ... }
(repeated for each layer)

[Summary Comments]
- Total parameters
- Memory breakdown by layer
```

## Troubleshooting

### Export fails with "Could not open lenet5_weights.bin"

**Cause:** Training program didn't create checkpoint file
**Solution:** 
1. Ensure training completed successfully
2. Check output includes "Saving trained weights to checkpoint file..."
3. Verify `lenet5_weights.bin` exists and is ~2.2 MB

### Generated header file is very large

**Expected behavior:** Header file is ~3-4 MB of text containing 550K float values. This is normal.

**If too large for MSP430:**
1. Consider quantization (int16 or int8)
2. Use external storage for weights
3. Implement dynamic weight loading

### Weights seem wrong/confidence is too low

**Verification steps:**
1. Check training accuracy in `output.txt` (should be >70%)
2. Verify epoch count (should be 30)
3. Test with reference Python implementation comparing inference results
4. Check weight ranges in header file (should vary, not all zeros or ones)

## Example MSP430 Inference Function

```c
#include "lenet5_weights.h"

// Simple forward pass skeleton
int infer_digit(float *image_input) {
    float conv1_out[18 * 28 * 28];  // 18 filters × 28×28
    float pool1_out[18 * 14 * 14];  // After 2×2 max pool
    float conv2_out[48 * 10 * 10];  // 48 filters × 10×10
    float pool2_out[48 * 5 * 5];    // After 2×2 max pool
    float fc1_out[FC1_SIZE];
    float fc2_out[FC2_SIZE];
    float logits[NUM_CLASSES];
    
    // Conv1 forward pass (reference pseudo-code)
    // for each output pixel in 28×28:
    //   for each filter:
    //     sum = bias
    //     for each 5×5 kernel:
    //       sum += input[...] * w_conv1[...]
    //     conv1_out[...] = ReLU(sum)
    
    // Pool1 max pooling...
    // Conv2 forward pass...
    // Pool2 max pooling...
    // FC1 dense layer with ReLU...
    // FC2 dense layer with ReLU...
    // FC3 output layer with softmax...
    
    // Find class with max confidence
    int predicted_class = 0;
    float max_confidence = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] > max_confidence) {
            max_confidence = logits[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}
```

## Files Generated

| File | Size | Purpose |
|------|------|---------|
| `lenet5_weights.bin` | ~2.2 MB | Binary checkpoint (intermediate, can delete after export) |
| `lenet5_weights.h` | ~3-4 MB | C header file for MSP430 (keep for projects) |

## Performance Notes

- **Float32 Inference:** Full precision, ~70-80% CIFAR-10 accuracy
- **MSP430F5969:** ~200 MHz, sufficient for real-time inference on small batches
- **Inference latency:** Depends on implementation (expect 100ms-1s per image)
- **Power consumption:** Varies by inference size, typically 10-100 mW active

## Next Steps

1. Train network: `./lenet_train`
2. Export weights: `./export_weights`
3. Copy `lenet5_weights.h` to MSP430 project
4. Implement inference engine for MSP430 (forward pass kernels)
5. Test with CIFAR-10 images
6. Deploy on MSP430 hardware



