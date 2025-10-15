import heat as ht
from africanus.constants import two_pi_over_c
from africanus.model.wsclean.spec_model import spectra as np_spectra

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

def heat_wsclean_predict(uvw, lm, source_type, flux, spi, log_poly, ref_freq, gauss_shape, frequency):
    """
    Vectorized wsclean_predict implementation using Heat.
    """
    # 1. Calculate spectrum on each rank (source, chan).
    # The `spectra` function is a NumPy function, so we run it on the local tensor data.
    spectrum_np = np_spectra(flux.larray.numpy(), spi.larray.numpy(), log_poly.larray.numpy(),
                             ref_freq.larray.numpy(), frequency.larray.numpy())
    spectrum = ht.array(spectrum_np, split=None, comm=uvw.comm)

    # 2. Calculate n component (source,)
    n = ht.sqrt(1.0 - ht.sum(lm**2, axis=1)) - 1.0

    # 3. Calculate phase term (row, source) using outer products (via matmul)
    # uvw is distributed at split=0, lm and n are replicated (split=None)
    phase = (uvw[:, 0:1] @ lm[:, 0:1].T +
             uvw[:, 1:2] @ lm[:, 1:2].T +
             uvw[:, 2:3] @ n.reshape(1, -1))

    # 4. Form the complex phasor (row, source, chan) using another outer product
    # phase is (row, source), frequency is (chan,)
    phasor_arg = two_pi_over_c * (phase.expand_dims(2) * frequency)
    
    re = ht.cos(phasor_arg)
    im = ht.sin(phasor_arg)
    phasor = re + im * 1j

    # 5. Multiply by spectrum, broadcasted along the row dimension
    # phasor is (row, source, chan), spectrum is (source, chan)
    vis_contrib = phasor * spectrum

    # 6. Sum-reduce over the source axis to get final visibilities
    vis = ht.sum(vis_contrib, axis=1)

    # Reshape to expected (row, chan, corr=1) output
    return vis.expand_dims(2)