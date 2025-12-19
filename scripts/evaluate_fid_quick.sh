#!/bin/bash
# Quick FID evaluation script

echo "=========================================="
echo "Diffusion Model FID Evaluation"
echo "=========================================="

# Default parameters
NUM_SAMPLES=100
BATCH_SIZE=16
DATA_DIR="dataset/OpenFWI/Velocity_Data/"
MODELS_DIR="pretrained_models"
OUTPUT_DIR="fid_evaluation"

# Run evaluation
python scripts/evaluate_diffusion_fid.py \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --data_dir $DATA_DIR \
    --models_dir $MODELS_DIR \
    --output_dir $OUTPUT_DIR \
    --device cuda

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
