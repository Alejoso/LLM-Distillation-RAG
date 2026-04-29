#!/bin/bash
#SBATCH --job-name="distill-compare"
#SBATCH --partition=accel-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:3
#SBATCH --time=1-24:00:00
#SBATCH --output=logs/%x_%j.out      # Standard output log (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err       # Standard error log
#SBATCH --mail-user=atirador1@eafit.edu.co         # Your email address from user input



echo "================================================="
echo "Starting job $SLURM_JOB_ID on host $(hostname)"
echo "Job name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of nodes: $SLURM_NNODES"
echo "Total number of tasks: $SLURM_NTASKS"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Inicio:       $(date)"
echo "================================================="
echo

# Cargar modulos (ajustar segun Apolo)
module load mods_alphafold/python-3.10.2-gcc-9.3.0-j6w76qf
module load cuda/11.3.0_Intel_oneAPI-2022_update-1
source /home/ugr-atirador1/LLM-Distillation-RAG/venv/bin/activate
# cd /home/ugr-atirador1/LLM-Distillation-RAG/Scripts/ProccessDataTrainning

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# Ejecutar pipeline completo (2 destilaciones + evaluacion)
# Si ChromaDB no existe, automaticamente solo corre sin RAG
# Usar --reset solo la primera vez para limpiar checkpoints viejos.
# En corridas posteriores (o si se reanuda tras una caida), quitar --reset.
python pipeline_destilacion_apolo.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --reset

EXIT_CODE=$?

echo "============================================"
echo "Fin:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
