#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

// ============================================================================
// Hyperparameters & Network Architecture Definitions
// ============================================================================
#define BATCH_SIZE 64
#define EPOCHS 30
#define LEARNING_RATE 0.01f

// CIFAR-10 specs
#define IMG_CHANNELS 3
#define IMG_WIDTH 32
#define IMG_HEIGHT 32
#define NUM_CLASSES 10
#define NUM_TRAIN_BATCHES 5

// Shared memory tile sizes for optimization
#define TILE_WIDTH 16

// LeNet-5 Architecture specs for CIFAR-10
#define C1_FILTERS 6
#define C1_SIZE 5
#define C1_OUT_W 28
#define C1_OUT_H 28

#define POOL_SIZE 2
#define POOL_STRIDE 2
#define P1_OUT_W 14
#define P1_OUT_H 14

#define C2_FILTERS 16
#define C2_SIZE 5
#define C2_OUT_W 10
#define C2_OUT_H 10

#define P2_OUT_W 5
#define P2_OUT_H 5

#define FC1_SIZE 120
#define FC2_SIZE 84
#define FC1_OUT 120
#define FC2_OUT 84

#endif // NETWORK_CONFIG_H
