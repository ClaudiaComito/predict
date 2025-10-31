import heat as ht
import torch
import torch.jit
from africanus.constants import two_pi_over_c

import logging
import time

import perun


def heat_radec_to_lm(radec, phase_dir):
    """
    Convert RA/Dec to l/m coordinates using Heat.
    Assumes radec and phase_dir are NumPy arrays replicated on all ranks.
    """
    # Note: phase_dir is a NumPy array, radec is a Heat DNDarray
    l = ht.sin(radec[:, 0] - phase_dir[0]) * ht.cos(radec[:, 1])
    m = (ht.sin(radec[:, 1]) * ht.cos(phase_dir[1]) -
         ht.cos(radec[:, 1]) * ht.sin(phase_dir[1]) *
         ht.cos(radec[:, 0] - phase_dir[0]))
    return ht.stack([l, m], axis=1)


@torch.jit.script  # <-- JIT-compiled spectra function
def torch_spectra(I: torch.Tensor, 
                  coeffs: torch.Tensor, 
                  log_poly: torch.Tensor, 
                  ref_freq: torch.Tensor, 
                  frequency: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of africanus.model.wsclean.spec_model.spectra.
    
    Calculates the spectral model using either ordinary or logarithmic polynomials
    in a vectorized manner.
    """
    
    n_source = I.shape[0]
    n_chan = frequency.shape[0]

    # Validate shapes
    if coeffs.shape[0] != n_source or ref_freq.shape[0] != n_source:
        # NOTE: Using print instead of raise for JIT-compatibility
        if coeffs.shape[0] != n_source or ref_freq.shape[0] != n_source:
            print("First dimension of I, coeffs, and ref_freq must match (source,)")
            # This will likely fail later, but JIT requires valid paths
        
    ncoeffs = coeffs.shape[1]

    # Reshape inputs for broadcasting

    # (source, 1)
    I_r = I.unsqueeze(1)
    ref_freq_r = ref_freq.unsqueeze(1)
    
    # (1, chan)
    freq_r = frequency.unsqueeze(0)
    
    # (source, 1, comp)
    coeffs_r = coeffs.unsqueeze(1)
    
    # (1, 1, comp) - these are the powers (c + 1)
    coeffs_idx = torch.arange(1, ncoeffs + 1, 
                              device=coeffs.device, 
                              dtype=coeffs.dtype).reshape(1, 1, ncoeffs)
    
    # (source, 1) - Boolean mask
    # NOTE: JIT-script requires we handle the bool case.
    # We will assume log_poly is ALWAYS a tensor, as created in prediction.py
    # if isinstance(log_poly, bool):
    #     log_poly_mask = torch.full((n_source, 1), log_poly, 
    #                                dtype=torch.bool, device=I.device)
    # else:
    #     if log_poly.shape[0] != n_source:
    #         print("log_poly shape must match I")
    log_poly_mask = log_poly.unsqueeze(1)


    # Calculate frequency ratio

    # (source, chan) via broadcasting (1, chan) / (source, 1)
    # Clamp to avoid division by zero or log(0)
    nu_over_rf = (freq_r / ref_freq_r).clamp(min=1e-30)

    
    # Calculate both models (vectorized)

    # Ordinary Model: I + sum(coeffs * ((nu/rf) - 1)**(c+1))
    # (source, chan)
    ord_term_base = nu_over_rf - 1.0
    # (source, chan, 1) ** (1, 1, comp) -> (source, chan, comp)
    ord_powered_terms = ord_term_base.unsqueeze(2) ** coeffs_idx
    # (source, 1, comp) * (source, chan, comp) -> (source, chan, comp)
    ord_full_terms = coeffs_r * ord_powered_terms
    # (source, chan)
    ord_poly_sum = ord_full_terms.sum(dim=2)
    # (source, 1) + (source, chan) -> (source, chan)
    ord_model = I_r + ord_poly_sum

    # Logarithmic Model: I * exp(sum(coeffs * (log(nu/rf))**(c+1)))
    # (source, chan)
    log_term_base = torch.log(nu_over_rf)
    # (source, chan, 1) ** (1, 1, comp) -> (source, chan, comp)
    log_powered_terms = log_term_base.unsqueeze(2) ** coeffs_idx
    # (source, 1, comp) * (source, chan, comp) -> (source, chan, comp)
    log_full_terms = coeffs_r * log_powered_terms
    # (source, chan)
    log_poly_sum = log_full_terms.sum(dim=2)
    # (source, 1) * exp((source, chan)) -> (source, chan)
    log_model = I_r * torch.exp(log_poly_sum)

    # Combine models using the boolean mask
    spectral_model = torch.where(log_poly_mask, log_model, ord_model)
    
    return spectral_model

@torch.jit.script  
def _local_predict_kernel(phase_local: torch.Tensor,
                          frequency_local: torch.Tensor,
                          flux_local: torch.Tensor,
                          spi_local: torch.Tensor,
                          log_poly_local: torch.Tensor,
                          ref_freq_local: torch.Tensor,
                          source_batch_size: int,
                          chan_batch_size: int,
                          two_pi_over_c: float) -> torch.Tensor:
    """
    JIT-compiled local prediction kernel.
    Runs on a single worker's local torch tensors.
    """

    #  Calculate Spectrum
    # Call our other JIT-compiled function
    spectrum_local = torch_spectra(flux_local, spi_local, log_poly_local,
                                   ref_freq_local, frequency_local)

    # Prepare for loops
    n_local_rows = phase_local.shape[0]
    n_sources = phase_local.shape[1]
    n_chan = frequency_local.shape[0]

    # Initialize the final local visibility tensor to zeros
    vis_local = torch.zeros((n_local_rows, n_chan), dtype=torch.complex64, device=phase_local.device)

    # Run batched loops

    # Loop over the source dimension in batches
    for s_start in range(0, n_sources, source_batch_size):
        s_end = min(s_start + source_batch_size, n_sources)

        # Slice the source-dependent arrays
        phase_batch = phase_local[:, s_start:s_end]
        spectrum_batch = spectrum_local[s_start:s_end, :]

        #  Nested loop over the channel dimension 
        for c_start in range(0, n_chan, chan_batch_size):
            c_end = min(c_start + chan_batch_size, n_chan)

            # Slice frequency and spectrum
            freq_chan_batch = frequency_local[c_start:c_end]
            spectrum_chan_batch = spectrum_batch[:, c_start:c_end]

            #  Perform calculations for this double-batch 
            
            # (row, s_batch, 1) * (c_batch) -> (row, s_batch, c_batch)
            phasor_arg_chan_batch = two_pi_over_c * (phase_batch.unsqueeze(2) * freq_chan_batch)

            # Compute complex phasor using Euler's formula (more JIT-friendly)
            phasor_chan_batch = torch.exp(phasor_arg_chan_batch * 1j)

            # Multiply by spectrum batch
            vis_contrib_chan_batch = phasor_chan_batch * spectrum_chan_batch

            # Sum-reduce over the source batch dimension
            vis_local[:, c_start:c_end] += torch.sum(vis_contrib_chan_batch, dim=1)

    # Reshape to expected (row, chan, corr=1) output and return
    return vis_local.unsqueeze(2)

@perun.monitor() #this fails i.e. the function runs but no monitoring output
def heat_wsclean_predict(uvw, lm, source_type, flux, spi, log_poly, ref_freq, gauss_shape, frequency, source_batch_size, chan_batch_size):
    """
    Vectorized wsclean_predict implementation using Heat.
    This function orchestrates the distributed computation and calls
    the JIT-compiled _local_predict_kernel on each worker.
    """
    
    #   Distributed (Heat) Calculations 
    logging.info("Starting Heat distributed phase/n calculations.")
    
    # Calculate n component (source,)
    n = ht.sqrt(1.0 - ht.sum(lm**2, axis=1)) - 1.0

    # Calculate phase term (row, source) using outer products
    start = time.perf_counter()
    phase = (uvw[:, 0:2] @ lm.T +
             uvw[:, 2:3] @ n.reshape(1, -1))
    end = time.perf_counter()
    logging.info(f"Calculated distributed phase in {end - start} seconds.")

    del n, lm, uvw  # Free memory in the orchestrator

    # Call Local (Torch JIT) Kernel 
    logging.info("Calling JIT-compiled local kernel on each worker.")
    
    # Convert batch sizes to standard ints for JIT-compatibility
    s_batch = int(source_batch_size)
    c_batch = int(chan_batch_size)

    # Call the JIT-compiled function with the local .larray tensors

    start = time.perf_counter()
    vis_local_tensor = _local_predict_kernel(
        phase.larray,
        frequency.larray,
        flux.larray,
        spi.larray,
        log_poly.larray,
        ref_freq.larray,
        s_batch,
        c_batch,
        two_pi_over_c
    )
    end = time.perf_counter()
    logging.info(f"JIT local kernel computation complete in {end - start} seconds.")

    # Wrap Result in Heat DNDarray
    logging.info("Computation complete. Wrapping result in ht.array.")
    
    # ht.array synchronizes all ranks and wraps the local tensor
    # in a distributed DNDarray.
    return ht.array(vis_local_tensor, is_split=0)