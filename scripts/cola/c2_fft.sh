#!/bin/bash

# Define the directory where you want to store output and error files
log_dir="/home/jpmunoz/AutoML/Phuong/fedtext/logs/c2"
# Create the directory if it doesn't exist
mkdir -p $log_dir

echo "Loading modules"
#source /export/work/yusx/miniconda3/bin/activate /export/work/yusx/miniconda3/envs/clip_test1/
source /home/jpmunoz/AutoML/Phuong/fedtext_env/bin/activate
#cd /export/work/yusx/phuong/FLoRA/system
cd /home/jpmunoz/AutoML/Phuong/fedtext/system
echo "$PWD"
#echo "Started batch job at $(date)"

# Learning rates to grid search over
learning_rates=(1e-3 5e-3 1e-4 5e-4 1e-5 5e-5 1e-6 5e-6 1e-7 5e-7 1e-8 5e-8)

# Iterate over learning rates, rank values, alpha values, and dropout values
for lr in "${learning_rates[@]}"; do
    # Dynamically generate job name and log file names based on algorithm and learning rate
    job_name="c2_dir10_fft_lr${lr}_dropout${dropout}"
    output_file="${log_dir}/${job_name}.out"
    error_file="${log_dir}/${job_name}.err"

    echo "$PWD"
    echo "Running with algo=flora, learning_rate=${lr}, lora_rank=${rank}, lora_alpha=${alpha}, and lora_dropout=${dropout}"

    # Run the job directly
    (
        nvidia-smi -L > $output_file && \
        nvidia-smi --query-gpu=compute_cap --format=csv >> $output_file && \
        echo 'GPU details saved to $output_file' && \
        time CUDA_VISIBLE_DEVICES=1 python main.py \
                    -data cola \
                    -algo fedfft \
                    -gr 50 \
                    -did 1 \
                    -nc 10 \
                    -lbs 128 \
                    -lr ${lr} \
                    -sd 42 \
                    -pfl
    ) > $output_file 2> $error_file 

    echo "Started job $job_name with algo=fedfft, learning_rate=${lr} at $(date)"
done

echo "Finish starting all jobs at $(date)"
