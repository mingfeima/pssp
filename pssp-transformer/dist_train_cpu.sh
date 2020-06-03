###
### Distributed CPU Training Script for PSSP-Transformer
###
### Usage (by default, the script will treats 2 sockets as 2 ranks
###  ./dis_train_cpu.sh
###

### You need to update the IP according to your local machine

IP="10.239.60.17:7689"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
LAST_CORE0=`expr $CORES - 1`
LAST_CORE1=`expr $CORES \* $SOCKETS - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
PREFIX_S0="numactl -C0-19 -m0"

export $KMP_SETTING
echo -e "\n### using $KMP_SETTING"

export OMP_NUM_THREADS=$CORES
echo -e "### using OMP_NUM_THREADS=$CORES"

PREFIX_S0="numactl -C0-$LAST_CORE0 -m0"
PREFIX_S1="numactl -C$CORES-$LAST_CORE1 -m1"
echo -e "### using s0 prefix: $PREFIX_S0"
echo -e "### using s1 prefix: $PREFIX_S1"

$PREFIX_S0 python -u main.py -world_size=2 -rank=0 -dist_backend=gloo -dist_url="tcp://$IP" &
$PREFIX_S1 python -u main.py -world_size=2 -rank=1 -dist_backend=gloo -dist_url="tcp://$IP"
