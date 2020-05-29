## Benchmarking CPU Training for PSSP-Transformer
The page indicates peformance BKMs for training PSSP-Transformer on CPU.

### Peformance Optimization
`LayerNorm` backward path on public PyTorch runs sequentially at the moment. I have a patch [#35750](https://github.com/pytorch/pytorch/pull/35750) addressing this issue. For details, please refer to this pull request.

### CPU Training BKMs
Most of PyTorch operators are _stateless_ which means the output tensor is re-created for every execution. And allocating (malloc) large memory buffers may pose extensive performance overhead which is also known as _clear_page_. To minimize the clear_page overhead, one can apply customized memroy allocator, e.g. [jemalloc](https://github.com/jemalloc/jemalloc/wiki/Getting-Started).

From skylake on, Xeon CPU has **NUMA** (non-uniform memory access) setting to regulate memory access. However, jemalloc is not NUMA aware which means it is ineffective in case a thread has memory access across different NUMA nodes.

So for CPU training, it is recommended to use distributed training, e.g. use 2 ranks on a dual socket CPU. Also:
1. Use distributed training, bind `RANK0` to `NUMA0` and so on.
1. Apply jemalloc.
2. Set correct environment variables including `OMP_NUM_THREADS`, `KMP_AFFINITY`, etc. 

Refer to `dist_train_cpu.sh` for details.

### Performance Result
Test machine: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz, 2 sockets, 20 cores each socket.

I. **build your PyTorch** - you can either cherry-pick the patch from [#35750](https://github.com/pytorch/pytorch/pull/35750) to your local branch or use the branch i prepared for this project [pssp](https://github.com/mingfeima/pytorch/tree/pssp), the build process is identical to official PyTorch as indicated [here](https://github.com/pytorch/pytorch#install-dependencies).

II. **config jemalloc** - you need to build jemalloc as follows:
```bash
  a) download from release: https://github.com/jemalloc/jemalloc/releases
  b) tar -jxvf jemalloc-5.2.0.tar.bz2
  c) ./configure; make
  d) cd /home/mingfeim/packages/jemalloc-5.2.0/bin; chmod 777 jemalloc-config
```

III. **reference cpu training** - train with single node
You need to prepare the dataset first
```bash
  a) download from http://www.princeton.edu/~jzthree/datasets/ICML2014/
    cullpdb+profile_6133_filtered.npy.gz
    cb513+profile_split1.npy.gz
  b) move to directory ${pssp_root_dir}/pssp-data/
  c) cd ${pssp_root_dir}/pssp-transformer; python preprocess.py
```
Then you are free to go as follows:
```bash
  ./train_cpu.sh
  ./train_cpu.sh --profile ### this will launch PyTorch autograd profiler to identify perf hotspots
```
On my test machine, each epoch takes `235.6` sec.
```bash
[001/100] train_loss  1.491 test_loss:  1.068 train_acc 0.525 test_acc 0.685 time 235.6 sec
```

IV. **distributed cpu training** - train with `torch.distributed`
For jemalloc config, environment settings, please refer to `dist_train_cpu.sh`. By default the script will lauch 2 ranks for a dual socket CPU.
```bash
./dist_train_cpu.sh
```
On my test machine, each epoch takes `137.9` sec.
```bash
[001/100] train_loss  1.911 test_loss:  1.166 train_acc 0.384 test_acc 0.667  time 137.9 sec
```
The distributed CPU training scenario is similar to multi-gpu training, you need to tune hyper-parameter accordingly.
