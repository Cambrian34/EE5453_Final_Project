# LeNet-5 CIFAR-10 GPU Training and Weight Export Makefile

# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O2
NVCCFLAGS = -std=c++11 -O2 -arch=sm_50

# Source files
TRAIN_SOURCES = lenet_cifar10.cu lenet_kernels.cu data_utils.cu
EXPORT_SOURCES = export_weights.cu data_utils.cu

# Output executables
TRAIN_EXEC = lenet_train
EXPORT_EXEC = export_weights

# Generated files
CHECKPOINT = lenet5_weights.bin
HEADER = lenet5_weights.h

# Phony targets
.PHONY: all train export clean help weights

# Default target
all: $(TRAIN_EXEC) $(EXPORT_EXEC)

# Training executable
$(TRAIN_EXEC): $(TRAIN_SOURCES) *.h
	@echo "[*] Compiling training program..."
	$(NVCC) $(NVCCFLAGS) -o $@ $(TRAIN_SOURCES)
	@echo "[+] Successfully built: $@"

# Export utility executable
$(EXPORT_EXEC): $(EXPORT_SOURCES) *.h
	@echo "[*] Compiling export utility..."
	$(NVCC) $(NVCCFLAGS) -o $@ $(EXPORT_SOURCES)
	@echo "[+] Successfully built: $@"

# Full workflow: train and export
weights: $(TRAIN_EXEC) $(EXPORT_EXEC)
	@echo ""
	@echo "=========================================================="
	@echo "LeNet-5 Training and Weight Export Workflow"
	@echo "=========================================================="
	@echo ""
	@echo "[STEP 1] Running training..."
	@echo "Command: ./$(TRAIN_EXEC)"
	@echo ""
	./$(TRAIN_EXEC)
	@if [ -f $(CHECKPOINT) ]; then \
		echo ""; \
		echo "[STEP 2] Exporting weights to header file..."; \
		echo "Command: ./$(EXPORT_EXEC)"; \
		echo ""; \
		./$(EXPORT_EXEC); \
		if [ -f $(HEADER) ]; then \
			echo ""; \
			echo "[+] SUCCESS! Generated $(HEADER)"; \
			echo "    Ready for MSP430 embedding"; \
		else \
			echo "[!] ERROR: Failed to generate $(HEADER)"; \
		fi \
	else \
		echo "[!] ERROR: Training did not create $(CHECKPOINT)"; \
	fi

# Train only
train: $(TRAIN_EXEC)
	@echo "Running training (outputs to output.txt)..."
	./$(TRAIN_EXEC)

# Export only (requires existing checkpoint)
export: $(EXPORT_EXEC)
	@if [ -f $(CHECKPOINT) ]; then \
		echo "Exporting weights from checkpoint..."; \
		./$(EXPORT_EXEC); \
	else \
		echo "ERROR: $(CHECKPOINT) not found!"; \
		echo "Run 'make train' first to generate the checkpoint."; \
		exit 1; \
	fi

# Clean build artifacts and generated files
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TRAIN_EXEC) $(EXPORT_EXEC)
	@echo "Clean: removed executables"

# Clean everything including generated weights
distclean: clean
	@echo "Removing generated weight files..."
	rm -f $(CHECKPOINT) $(HEADER)
	@echo "Full clean complete"

# Display help
help:
	@echo ""
	@echo "LeNet-5 CIFAR-10 GPU Training Makefile"
	@echo "======================================="
	@echo ""
	@echo "Targets:"
	@echo "  make all        - Build training and export executables"
	@echo "  make train      - Build and run training program"
	@echo "  make export     - Build and run export utility (requires checkpoint)"
	@echo "  make weights    - Full workflow: train → export weights"
	@echo "  make clean      - Remove executables"
	@echo "  make distclean  - Remove executables and generated files"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Workflow:"
	@echo "  1. make weights           # Train and export in one command"
	@echo "  2. cp lenet5_weights.h MSP430_project/"
	@echo "  3. Implement inference engine for MSP430"
	@echo ""
	@echo "Generated Files:"
	@echo "  lenet_train          - Training executable"
	@echo "  export_weights       - Weight export utility"
	@echo "  lenet5_weights.bin   - Binary checkpoint (~2.2 MB)"
	@echo "  lenet5_weights.h     - C header file (~3-4 MB)"
	@echo "  output.txt           - Training log"
	@echo ""

.DEFAULT_GOAL := help
