# README: Heat Backend for `predict` Benchmark

This branch introduces a new computational backend for the `predict` visibility benchmark, utilizing the **[Heat](https://github.com/helmholtz-analytics/heat) framework**.

The primary goal of this backend is to replace the Dask-based computation with a pure MPI-based implementation, enabling efficient execution on large-scale HPC systems and offloading the core computation to **GPUs**.

---

## Key Changes

1.  **New Backend Selection:**
    * `app.py` introduces two new command-line arguments:
        * `--backend`: Allows choosing between `dask` (default) and `heat`.
        * `--device`: Allows selecting `cpu` (default) or `gpu` when using the Heat backend.

2.  **Dask-Free Execution:**
    * When `--backend heat` is selected, `app.py` **bypasses all Dask and Dask.distributed setup**.
    * No `LocalCluster` or `Client` is created. The script runs as a standard MPI application, and Heat handles the distributed communication.

3.  **Heat-Native Compute Kernel:**
    * A new file, `heat_kernels.py`, has been added. This file contains the Heat/PyTorch implementations of the core prediction algorithm.
    * `prediction.py` contains a new `elif backend == "heat":` block that orchestrates the Heat-based workflow.

---

## Usage

Unlike the Dask version, the Heat backend is executed using `mpirun` or a cluster-specific job launcher (e.g., `srun`).

### 1. Data Prerequisites

This backend is designed to work with the Zarr dataset structure created by `dask-ms`, where the main data is partitioned into sub-directories (e.g., `MAIN_0`, `MAIN_1`, ...).

The parallel I/O in the Heat backend **requires this structure** to load the data.

### 2. Running the Benchmark

Execute the application using `mpirun`. The arguments are the same as the Dask version, but you must add `--backend heat` and (optionally) `--device gpu`.

```bash
# Example running on 4 MPI ranks with GPU acceleration
mpirun -n 4 python -m predict.app \
    /path/to/partitioned.zarr \
    --output-store /path/to/output.zarr \
    --backend heat \
    --device gpu \
    --dimensions "{chan: 4096, source: 1000}" \
    --chunks "{row: 10000, chan: 64, source: 100}"
 ```

### 3. Argument Interpretation

* `--backend heat`: Activates the Heat backend.
* `--device gpu`: Instructs Heat to use the GPU for all computations.
* `--chunks`: For the Heat backend, these values are **re-purposed as batch sizes** to solve the OOM bottleneck.
    * `chunks["source"]`: Sets the `source_batch_size`.
    * `chunks["chan"]`: Sets the `chan_batch_size`.

---

## Implementation Details

### Parallel I/O

The Heat backend performs a scalable, parallel read of the input data. It uses Heat's wildcard loading capability to find all `MAIN_*` partitions and load the `UVW` array from each one:

```python
# Loads all UVW arrays from MAIN_*/UVW and concatenates them
# along the 'row' axis (split=0)
ht_uvw = ht.load(args.store, variable="MAIN_*/UVW", split=0)
```

This allows all MPI ranks to participate in I/O, avoiding a "rank 0 reads" bottleneck. Sky model and metadata are replicated on all ranks (`split=None`).

### Solving Out-of-Memory (OOM) with Double-Batching

The core prediction kernel, `heat_wsclean_predict`, is designed to avoid materializing the massive `(row, source, chan)` intermediate tensor.

It uses a **double-batching strategy** by looping over both the `source` and `chan` dimensions in small chunks. This keeps the peak memory footprint low, allowing the computation to fit on the device.

### Hybrid CPU/GPU Kernel

The core `heat_wsclean_predict` kernel operates on local PyTorch tensors. However, the spectral model calculation (`np_spectra`) is still performed on the **CPU using NumPy** inside the loop. The result is then copied back to the GPU for the main computation.

```python
# This calculation runs on CPU
spectrum_local = torch.tensor(np_spectra(flux.larray.cpu().numpy(), ...))
# Result is moved back to the compute device
spectrum_local = spectrum_local.to(phase_local.device)
```

This is a known performance bottleneck, and a future optimization would be to port `np_spectra` to a pure Torch implementation.