import heat as ht
import torch
from africanus.constants import two_pi_over_c
from africanus.model.wsclean.spec_model import spectra as np_spectra

import logging

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

def heat_wsclean_predict(uvw, lm, source_type, flux, spi, log_poly, ref_freq, gauss_shape, frequency, source_batch_size, chan_batch_size):
    """
    Vectorized wsclean_predict implementation using Heat.
    """
    # Calculate n component (source,)
    n = ht.sqrt(1.0 - ht.sum(lm**2, axis=1)) - 1.0

    # Calculate phase term (row, source) using outer products (via matmul)
    # uvw is distributed at split=0, lm and n are replicated (split=None)
    phase = (uvw[:, 0:2] @ lm.T +
             uvw[:, 2:3] @ n.reshape(1, -1))

    del n, lm, uvw  # Free memory

    # Form the complex phasor (row, source, chan) using another outer product
    # phase is (row, source), frequency is (chan,)

    # test: all local operations to torch tensors, test memory use
    phase_local = phase.larray
    frequency_local = frequency.larray

    #TODO: port africanus.model.wsclean.spec_model.spectra to torch
    spectrum_local = torch.tensor(np_spectra(flux.larray.cpu().numpy(), spi.larray.cpu().numpy(), log_poly.larray.cpu().numpy(),
                                                ref_freq.larray.cpu().numpy(), frequency_local.cpu().numpy()))

    # push spectrum_local to the same device as phase_local
    spectrum_local = spectrum_local.to(phase_local.device)
    # Get local shapes
    n_local_rows = phase_local.shape[0]
    n_sources = phase_local.shape[1]
    n_chan = frequency_local.shape[0]

    # Initialize the final local visibility tensor to zeros
    vis_local = torch.zeros((n_local_rows, n_chan), dtype=torch.complex64, device=phase_local.device)

    # Loop over the source dimension in batches
    for s_start in range(0, n_sources, source_batch_size):
        s_end = min(s_start + source_batch_size, n_sources)

        # Slice the source-dependent arrays for the current source batch
        phase_batch = phase_local[:, s_start:s_end]
        spectrum_batch = spectrum_local[s_start:s_end, :]
        logging.info(f"Processing sources {s_start} to {s_end}.")

        # --- Nested loop over the channel dimension ---
        for c_start in range(0, n_chan, chan_batch_size):
            c_end = min(c_start + chan_batch_size, n_chan)

            # Slice frequency and spectrum for the current channel batch
            freq_chan_batch = frequency_local[c_start:c_end]
            spectrum_chan_batch = spectrum_batch[:, c_start:c_end]

            # --- Perform calculations for this smaller double-batch ---
            # Shape: (n_local_rows, source_batch_size, chan_batch_size)
            phasor_arg_chan_batch = two_pi_over_c * (phase_batch.unsqueeze(2) * freq_chan_batch)

            # Compute complex phasor for the batch
            phasor_chan_batch = torch.cos(phasor_arg_chan_batch) + torch.sin(phasor_arg_chan_batch) * 1j
            del phasor_arg_chan_batch

            # Multiply by spectrum batch
            vis_contrib_chan_batch = phasor_chan_batch * spectrum_chan_batch
            del phasor_chan_batch, spectrum_chan_batch

            # Sum-reduce over the source batch dimension and accumulate into the correct channel slice
            vis_local[:, c_start:c_end] += torch.sum(vis_contrib_chan_batch, dim=1)
            del vis_contrib_chan_batch
        
        logging.info(f"Completed all channels for sources {s_start} to {s_end}.")

    # Reshape to expected (row, chan, corr=1) output and wrap in DNDarray
    # NB: ranks get synchronized here
    return ht.array(vis_local.unsqueeze(2), is_split=0)

