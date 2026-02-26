# README: Heat Backend for `predict` Benchmark

This repository introduces a computational backend for the `predict` visibility benchmark using the **[Heat](https://github.com/helmholtz-analytics/heat) framework**. 


The Heat backend is an experimental implementation designed to:
* **Evaluate Scalability**: Test the performance and scaling of a purely data-parallel (MPI-based) approach compared to Dask's task-graph scheduling.
* **Enable GPU Portability**: Leverage PyTorch-backed tensors to easily offload core interferometry kernels to GPUs with minimal code changes.
* **Optimize High-Performance I/O**: Utilize MPI-parallelized reads for Zarr datasets across large HPC clusters.

This is Work In Progress.

---

## Key Changes

The Heat backend shifts from a task-based, lazy-evaluation model to a data-parallel, eager-execution strategy designed for high-performance computing (HPC) environments.

### 1. Backend and device selection
* Users can toggle between the original `dask` implementation and the new `heat` implementation via the `--backend` command-line argument.
* A new `--device` argument allows the Heat backend to target either `cpu` or `gpu`. When `gpu` is selected, core visibility calculations are offloaded to hardware accelerators using PyTorch-backed tensors.
* When the Heat backend is active, the application bypasses all `LocalCluster` or `Client` setup, running instead as a standard MPI application.

### 2. Implementation of JIT-Compiled kernels
* A new file, `heat_kernels.py`, has been added. This file contains the Heat/PyTorch implementations of the core prediction algorithm.
* The Heat backend uses JIT-compiled PyTorch kernels (`_local_predict_kernel`) to execute the double-batching logic directly on the compute device.
* The spectral model calculation has been ported from a CPU-bound NumPy implementation to a fully vectorized `torch_spectra` kernel. This calculates complex spectral models directly on the target device, eliminating host-to-device transfer bottlenecks.

### 3. Distributed Orchestration and I/O
* The Heat backend replaces Dask’s dynamic task scheduling with a bulk-synchronous MPI approach where Heat handles distributed communication.
* To prevent I/O bottlenecks, the Heat backend uses parallel wildcard loading (`MAIN_*/UVW`). This allows every MPI rank to read its own data partition from the Zarr store simultaneously.


### 4. Memory management
* In Dask mode, the `--chunks` argument defines the granularity of the lazy task graph.
* In Heat mode, these same values are re-purposed as **batch sizes** (`source_batch_size` and `chan_batch_size`) to prevent out-of-memory (OOM) errors both on GPUs and CPUs, by controlling the memory footprint of internal loops.


## Usage

Unlike the Dask version, the Heat backend is executed using `mpirun` or a cluster-specific job launcher (e.g., `srun`). A sample sbatch script is provided in `run_heat_benchmark_example.sh`. You need to adapt the module loading and environment setup to your specific HPC system.

### 1. Data Prerequisites

This backend is designed to work with the Zarr dataset structure created by `dask-ms`, where the main data is partitioned into sub-directories (e.g., `MAIN_0`, `MAIN_1`, ...).

The parallel I/O in the Heat backend loads the UVW data from all these partitions in parallel on the available devices (loading a zarr group with `variable="MAIN_*/UVW"`).

### 2. Running the Benchmark

The `run_heat_benchmark.sh` script provides an example of how to run the `predict` benchmark with the Heat backend on an HPC system (tested on JUWELS Booster).

Execute the application using `sbatch <script>`. The arguments are the same as the Dask version, but you must add `--backend heat` and (optionally) `--device gpu`. 

### 3. Argument Interpretation

* `--backend heat`: Activates the Heat backend.
* `--device gpu`: Instructs Heat to use the GPUs for all computations.
* `--chunks`: For the Heat backend, these values are **re-purposed as batch sizes** to control memory usage during the double-batching strategy (vary these values to fit the memory capacity of your GPU):
    * `chunks["source"]`: Sets the `source_batch_size`.
    * `chunks["chan"]`: Sets the `chan_batch_size`.

---

## Implementation Details

### Parallel I/O

The Heat backend performs a scalable, parallel read of the input data. It uses Heat's wildcard loading capability to find all `MAIN_*` partitions and load the `UVW` array from each one:

```python
ht_uvw = ht.load(args.store, variable="MAIN_*/UVW", split=0)
```

This allows all MPI ranks to participate in I/O, avoiding a "rank 0 reads" bottleneck. Sky model and metadata are replicated on all ranks (`split=None`).

### heat_wsclean_predict Kernel

The core prediction kernel, `heat_wsclean_predict`, is designed to avoid materializing the massive `(row, source, chan)` intermediate tensor.

It uses a double-batching strategy by looping over both the `source` and `chan` dimensions in small chunks. This keeps the peak memory footprint low, allowing the computation to fit on the device.

