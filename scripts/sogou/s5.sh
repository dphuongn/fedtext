#!/bin/bash

# Define the directory where you want to store output and error files
log_dir="/export/work/yusx/phuong/FLoRA/logs/s5"

# Create the directory if it doesn't exist
mkdir -p $log_dir

echo "Loading modules"
source /export/work/yusx/miniconda3/bin/activate /export/work/yusx/miniconda3/envs/clip_test1/
cd /export/work/yusx/phuong/FLoRA/system
echo "$PWD"
echo "Started batch job at $(date)"

# Learning rates to grid search over
learning_rates=(1e-3 5e-3 1e-4 5e-4 1e-5 1e-6 5e-6 1e-7 5e-7 1e-8)
rank_values=(1 2 4 8 16 32)
alpha_values=(1 4 8 16 32 64)
dropout_values=(0 0.05 0.1)

# Iterate over learning rates, rank values, alpha values, and dropout values
for lr in "${learning_rates[@]}"; do
    for rank in "${rank_values[@]}"; do
        for alpha in "${alpha_values[@]}"; do
            for dropout in "${dropout_values[@]}"; do
                # Dynamically generate job name and log file names based on algorithm and learning rate
                job_name="s5_dir_flora_lr${lr}_rank${rank}_alpha${alpha}_dropout${dropout}"
                output_file="${log_dir}/${job_name}.out"
                error_file="${log_dir}/${job_name}.err"

                echo "$PWD"
                echo "Running with algo=flora, learning_rate=${lr}, lora_rank=${rank}, lora_alpha=${alpha}, and lora_dropout=${dropout}"

                # Run the job directly
                (
                    nvidia-smi -L > $output_file && \
                    nvidia-smi --query-gpu=compute_cap --format=csv >> $output_file && \
                    echo 'GPU details saved to $output_file' && \
                    time python main.py \
                                -data sogou_news \
                                -algo flora \
                                -gr 100 \
                                -did 0 \
                                -nc 10 \
                                -lbs 128 \
                                -lr ${lr} \
                                -sd 42 \
                                --lora_rank ${rank} \
                                --lora_alpha ${alpha} \
                                --lora_dropout ${dropout} \
                                --lora_query \
                                --lora_key \
                                --lora_value \
                                --lora_projection \
                                --lora_mlp \
                                --lora_head \
                                -pfl
                ) > $output_file 2> $error_file &

                echo "Started job $job_name with algo=flora, learning_rate=${lr}, lora_rank=${rank}, lora_alpha=${alpha}, and lora_dropout=${dropout} at $(date)"
            done
        done
    done
done

echo "Finish starting all jobs at $(date)"
