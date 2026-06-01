#!/bin/bash
#SBATCH --job-name="judge-calib"
#SBATCH --partition=accel-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=atirador1@eafit.edu.co

# Calibracion del LLM-as-Judge sobre sets controlados.
#
# Lanzar desde el root del repo:
#   cd /home/ugr-atirador1/LLM-Distillation-RAG
#   sbatch 04_distillation/judge_calibration/submit_calibrate_judge_apolo.sh
#
# Output esperado en:
#   ./outputs/judge_calibration/{results.jsonl, report.json, calibrate_judge.log}

echo "================================================="
echo "Job $SLURM_JOB_ID on host $(hostname)"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Inicio: $(date)"
echo "================================================="

module load mods_alphafold/python-3.10.2-gcc-9.3.0-j6w76qf
module load cuda/11.3.0_Intel_oneAPI-2022_update-1
source /home/ugr-atirador1/LLM-Distillation-RAG/venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

python 04_distillation/judge_calibration/calibrate_judge.py \
    --judge_name meta-llama/Llama-2-7b-chat-hf \
    --output_dir ./outputs/judge_calibration \
    --versions v1 v2

EXIT_CODE=$?

echo "============================================"
echo "Fin:       $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
