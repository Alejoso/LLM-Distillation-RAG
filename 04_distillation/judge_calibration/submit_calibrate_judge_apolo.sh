#!/bin/bash
#SBATCH --job-name="judge-calib"
#SBATCH --partition=accel-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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

# Comparamos 4 variantes sobre 4 sets (calibration, traps, traps_extended, holdout):
#   v2   = prompt rubric+few-shot (baseline, el mejor prompt hasta ahora)
#   v4   = prompt reference-first afinado (off-topic / absurdo / lang-mix)
#   v2g  = v2 + guardrail determinista pre-LLM (answer_guardrails.py)
#   v4g  = v4 + guardrail determinista pre-LLM
# El guardrail resuelve deterministamente la clase degenerada (vacio, eco,
# circular, abstencion) que causaba el artefacto "student > teacher"; el prompt
# solo juzga la clase semantica. v2/v2g y v4/v4g comparten la generacion LLM en
# los items que la guarda deja pasar (una sola llamada por prompt por item).
# NOTA: v3 se descarta (empeoro a v2: wMAE 1.48 vs 0.98, false_5 88 vs 52).
python 04_distillation/judge_calibration/calibrate_judge.py \
    --judge_name meta-llama/Llama-2-7b-chat-hf \
    --output_dir ./outputs/judge_calibration \
    --variants v2 v4 v2g v4g

EXIT_CODE=$?

echo "============================================"
echo "Fin:       $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
