#!/bin/bash

# For debugging/trouble shooting, can load this
# salloc --time=01:00:00 --account=def-kmoran --cpus-per-task=4 --mem=32G --gres=gpu:h100:2 
# bash run_ADCP_salloc.sh
# -----------------------------
# Load your modules and activate conda
# -----------------------------
module load python/3.11
source /home/slonimer/torch_env25/bin/activate

# Go to working directory
cd /lustre10/scratch/slonimer/

# -----------------------------
# Define variants and batch sizes
# -----------------------------
VARIANTS=("resnet18" "resnet34" "resnet50")
BATCH_SIZES=(16 32 64)
RUN_ID=0
NUM_EPOCH=1

# -----------------------------
# Loop through variants and batch sizes
# -----------------------------
for VARIANT in "${VARIANTS[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        ((RUN_ID+=1))
        LOG_FILE="/lustre10/scratch/slonimer/logs/run_${RUN_ID}_${VARIANT}_bs${BATCH}.log"

        echo "Running ${VARIANT} with batch size ${BATCH} (run ID ${RUN_ID})"
        echo "Logging to ${LOG_FILE}"

        # Run Python training script with run_id
        python run_ADCP_training.py \
            --resnet $VARIANT \
            --batch_size $BATCH \
            --epochs $NUM_EPOCH \
            --run_id $RUN_ID \
            > $LOG_FILE 2>&1
    done
done