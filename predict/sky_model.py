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
    rng = default_rng()

    # MeerKAT centre frequency
    ref_freq = np.full(sdims, (0.856e9 + 2 * 0.856e9) / 2)
    # Use integer codes for masking: 0 for POINT, 1 for GAUSSIAN
    source_type = np.zeros(sdims, dtype=np.int8)
    flux = rng.random(sdims) * 1e-4
    spi = rng.random((sdims, 2)) * 1e-3


    # six degrees around zero
    radec = rng.random((sdims, 2))
    radec = np.deg2rad(6.0 * (radec - 0.5))
    log_si = np.full(sdims, False)
    gauss_shape = np.zeros((sdims, 3))

    return WSCleanModel(
        source_type,
        radec,
        flux,
        spi,
        ref_freq,
        log_si,
        gauss_shape,
    )
