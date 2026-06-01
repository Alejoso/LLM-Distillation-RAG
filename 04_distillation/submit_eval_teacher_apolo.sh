#!/bin/bash
#SBATCH --job-name="eval-teacher"
#SBATCH --partition=accel-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=atirador1@eafit.edu.co

# Evalua el teacher (con y sin RAG) sobre el mismo subset de 1500 items que
# usaron base_student/no_rag/with_rag. Reusa el teacher_output ya generado en
# fase 02, asi que NO vuelve a correr inferencia del teacher: solo corre al juez.
#
# Argumentos opcionales:
#   $1  experimentos (separados por coma): no_rag,with_rag (default)
#   $2  judge_prompt_version: v1 (default, comparable con eval previa) | v2
#
# Ejemplo:
#   sbatch 04_distillation/submit_eval_teacher_apolo.sh
#   sbatch 04_distillation/submit_eval_teacher_apolo.sh no_rag v1
#   sbatch 04_distillation/submit_eval_teacher_apolo.sh no_rag,with_rag v2

EXPERIMENTS="${1:-no_rag,with_rag}"
JUDGE_VERSION="${2:-v1}"

IFS=',' read -ra EXP_ARR <<< "$EXPERIMENTS"

echo "================================================="
echo "Eval teacher job $SLURM_JOB_ID on host $(hostname)"
echo "Experiments:   ${EXP_ARR[*]}"
echo "Judge prompt:  $JUDGE_VERSION"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Inicio:        $(date)"
echo "================================================="

module load mods_alphafold/python-3.10.2-gcc-9.3.0-j6w76qf
module load cuda/11.3.0_Intel_oneAPI-2022_update-1
source /home/ugr-atirador1/LLM-Distillation-RAG/venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

python 04_distillation/eval_teacher.py \
    --output_dir ./outputs \
    --experiments "${EXP_ARR[@]}" \
    --judge_prompt_version "$JUDGE_VERSION"

EXIT_CODE=$?

echo "============================================"
echo "Fin:       $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
