###
### CPU Training Script for PSSP-Transformer
###
### Usage:
###   ./train_cpu.sh
###   ./train_cpu.sh --profile  ### enable profiler
###

ARGS=""
if [[ "$1" == "--profile" ]]; then
  ARGS="$ARGS -profile"
  echo "### start autograd profiler"
  shift
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
TOTAL_CORES=`expr $CORES \* $SOCKETS`

export $KMP_SETTING
echo -e "\n### using $KMP_SETTING"

export OMP_NUM_THREADS=$TOTAL_CORES
echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"

python main.py $ARGS
