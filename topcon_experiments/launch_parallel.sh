#!/bin/bash
# Stop the sequential run and start 4 concurrent per-metric processes.
#
# Edit CONDA_SH / CONDA_ENV below to match your installation before running.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
cd "$HERE" || exit 1

CONDA_SH="${CONDA_SH:-$HOME/anaconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-autogluon}"

pkill -f "_run_exp2_rerun.py"
sleep 3
echo "remaining sequential procs: $(pgrep -cf '_run_exp2_rerun.py')"

# shellcheck disable=SC1090
source "$CONDA_SH" "$CONDA_ENV"

# Pin BLAS/OpenMP threads so the NeuralNetTorch / NeuralNetFastAI members of the
# AutoGluon ensembles use a fixed thread count -> stable reduction order.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

for m in iv_Voc iv_Jsc iv_FF iv_Eff; do
  setsid nohup python -u _run_exp2_one_metric.py "$m" \
      > "$LOG_DIR/exp2_rerun_${m}.log" 2>&1 < /dev/null &
  echo "launched $m pid=$!"
  sleep 4
done

echo "--- launched at $(date) ---"
