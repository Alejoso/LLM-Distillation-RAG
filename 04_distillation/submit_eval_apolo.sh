#!/bin/bash
#SBATCH --job-name="distill-eval"
#SBATCH --partition=accel-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=atirador1@eafit.edu.co

# Evaluacion de un solo modelo por job. Lanzar los 3 en paralelo:
#   sbatch --job-name=eval-base    submit_eval_apolo.sh base_student
#   sbatch --job-name=eval-norag   submit_eval_apolo.sh no_rag
#   sbatch --job-name=eval-withrag submit_eval_apolo.sh with_rag
#
# Argumentos:
#   $1  model_key      (base_student | no_rag | with_rag)   [requerido]
#   $2  subset_size    int                                  [default: 1500]

MODEL_KEY="${1:?ERROR: falta model_key. Usa: sbatch submit_eval_apolo.sh <model_key> [subset_size]}"
SUBSET_SIZE="${2:-1500}"

case "$MODEL_KEY" in
    base_student|no_rag|with_rag) ;;
    *)
        echo "ERROR: model_key invalido: '$MODEL_KEY'. Debe ser base_student | no_rag | with_rag." >&2
        exit 1
        ;;
esac

echo "================================================="
echo "Eval job $SLURM_JOB_ID on host $(hostname)"
echo "Model key:    $MODEL_KEY"
echo "Subset size:  $SUBSET_SIZE"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Inicio:       $(date)"
echo "================================================="
echo

module load mods_alphafold/python-3.10.2-gcc-9.3.0-j6w76qf
module load cuda/11.3.0_Intel_oneAPI-2022_update-1
source /home/ugr-atirador1/LLM-Distillation-RAG/venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

python pipeline_destilacion_apolo.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --only_eval \
    --only_models "$MODEL_KEY" \
    --eval_subset_size "$SUBSET_SIZE"

EXIT_CODE=$?

echo "============================================"
echo "Fin:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
