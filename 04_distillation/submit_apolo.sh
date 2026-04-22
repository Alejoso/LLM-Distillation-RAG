#!/bin/bash
#SBATCH --job-name=distill-compare
#SBATCH --partition=longjobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=END,FAIL

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodo:         $SLURM_NODELIST"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Inicio:       $(date)"
echo "============================================"

# Cargar modulos (ajustar segun Apolo)
module load python/3.10
module load cuda/11.8

# Entorno virtual
VENV_DIR="$HOME/envs/distillation"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements_apolo.txt
else
    source "$VENV_DIR/bin/activate"
fi

WORK_DIR="$(dirname "$(readlink -f "$0")")"
cd "$WORK_DIR"

echo "Directorio:   $WORK_DIR"
echo "Python:       $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# Ejecutar pipeline completo (2 destilaciones + evaluacion)
# Si ChromaDB no existe, automaticamente solo corre sin RAG
python pipeline_destilacion_apolo.py \
    --data_dir ./data \
    --output_dir ./outputs

EXIT_CODE=$?

echo "============================================"
echo "Fin:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
