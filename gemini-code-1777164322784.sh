nvcc lenet_cifar10.cu lenet_kernels.cu data_utils.cu -o lenet_train -O3
3. Execute the binary:
```bash
./lenet_train

### Explaining the Bonus Optimization
In project report, make sure to highlight the `conv1_forward_kernel`.
By utilizing `__shared__ float s_input[]` and cooperative thread loading,
this kernel pulls an entire tile of the image into the GPU's ultra-fast L1 
shared memory cache. Threads then compute the dot product using the shared
memory rather than fetching pixels repeatedly from slower global memory. 
This directly satisfies the `(+10) Bonus` requirement for "applying GPU
parallelism and further optimizing the training process through improved 
kernel design."

*Note: For the absolute completion of the architecture loop, you will string
together the remaining `fc_forward` and `pool_forward` kernels 
following the exact data shape transitions defined in your PyTorch
notebook reference.*