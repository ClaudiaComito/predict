from argparse import Namespace
from collections import namedtuple

# import dask.array as da
import numpy as np
from numpy.random import default_rng

WSCleanModel = namedtuple(
    "WSCleanModel",
    ["source_type", "radec", "flux", "spi", "ref_freq", "log_poly", "gauss_shape"],
)


def generate_sky_model(args: Namespace) -> WSCleanModel:
    sdims = args.dimensions["source"]
    # chunks not used in this implementation
    # Initialise a random number generator
    seed = 42
    rng = default_rng(seed)

    # MeerKAT centre frequency
    ref_freq = np.full(sdims, (0.856e9 + 2 * 0.856e9) / 2, dtype=np.float32)
    # Use integer codes for masking: 0 for POINT, 1 for GAUSSIAN
    source_type = np.zeros(sdims, dtype=np.int32)
    flux = rng.random(sdims, dtype=np.float32) * 1e-4
    spi = rng.random((sdims, 2), dtype=np.float32) * 1e-3


    # six degrees around zero
    radec = rng.random((sdims, 2), dtype=np.float32)
    radec = np.deg2rad(6.0 * (radec - 0.5))
    log_si = np.full(sdims, False)
    gauss_shape = np.zeros((sdims, 3), dtype=np.float32)

    return WSCleanModel(
        source_type,
        radec,
        flux,
        spi,
        ref_freq,
        log_si,
        gauss_shape,
    )
