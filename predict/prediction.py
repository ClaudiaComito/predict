import argparse
from contextlib import ExitStack
import math
import warnings

import dask
import dask.array as da
from dask.distributed import get_client
from dask.graph_manipulation import clone
from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import wsclean_predict

from daskms import xds_from_storage_ms, xds_from_storage_table
from daskms.experimental.zarr import xds_to_zarr
from daskms.fsspec_store import DaskMSStore
from daskms.optimisation import inlined_array

from predict.annotations import annotate_datasets, dim_propagator
import logging
from predict.sky_model import WSCleanModel


def expand_vis(vis, corrs):
    if corrs == 1:
        return vis
    elif corrs == 2:
        return da.concatenate([vis, vis], axis=2).rechunk({2: corrs})
    elif corrs == 4:
        zeros = da.zeros_like(vis)
        return da.concatenate([vis, zeros, zeros, vis], axis=2).rechunk({2: corrs})
    else:
        raise ValueError(f"MS Correlations {corrs} not in (1, 2, 4)")


def predict_vis(args: argparse.Namespace, sky_model: WSCleanModel, backend: str):
    if backend == "dask":
        client = get_client()
        nchan = args.dimensions["chan"]
        chan_chunks = args.chunks["chan"]
        schunks = args.chunks["source"]

        store = DaskMSStore(args.store)
        kw = {"group_cols": "__row__"} if store.type() == "casa" else {}

        ddid_ds = xds_from_storage_table(f"{args.store}::DATA_DESCRIPTION", **kw)
        pol_ds = xds_from_storage_table(f"{args.store}::POLARIZATION", **kw)
        spw_ds = xds_from_storage_table(f"{args.store}::SPECTRAL_WINDOW", **kw)
        field_ds = xds_from_storage_table(f"{args.store}::FIELD", **kw)

        (ddid_ds,) = dask.compute(ddid_ds)
        (pol_ds,) = dask.compute(pol_ds)
        (spw_ds,) = dask.compute(spw_ds)
        (field_ds,) = dask.compute(field_ds)

        datasets = xds_from_storage_ms(
            args.store,
            columns=["UVW"],
            group_cols=["FIELD_ID", "DATA_DESC_ID"],
            chunks={k: args.chunks[k] for k in ("row",)},
        )

        with ExitStack() as stack:
            if args.plugin == "pinned":
                stack.enter_context(dask.config.set(array_plugins=[dim_propagator("row")]))
            elif args.plugin in {"none", "autorestrictor"}:
                pass
            else:
                raise ValueError(f"Unhandled {args.plugin} case")


            datasets = annotate_datasets(datasets)
            out_datasets = []

            for ds in datasets:
                field = field_ds[ds.attrs["FIELD_ID"]]
                ddid = ddid_ds[ds.attrs["DATA_DESC_ID"]]
                # spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.values[0]]
                pol = pol_ds[ddid.POLARIZATION_ID.values[0]]

                with dask.annotate(dims=("chan",)):
                    frequency = da.linspace(0.856e9, 2 * 0.856e9, nchan, chunks=chan_chunks)

                radec = sky_model.radec
                source_type = sky_model.source_type
                flux = sky_model.flux
                spi = sky_model.spi
                log_poly = sky_model.log_poly
                ref_freq = sky_model.ref_freq
                gauss_shape = sky_model.gauss_shape
                # Ingest numpy sky model arrays into dask, chunking along source dim
                radec = da.from_array(sky_model.radec, chunks=(schunks, 2))
                source_type = da.from_array(sky_model.source_type, chunks=schunks)
                flux = da.from_array(sky_model.flux, chunks=schunks)
                spi = da.from_array(sky_model.spi, chunks=schunks)
                log_poly = da.from_array(sky_model.log_poly, chunks=schunks)
                ref_freq = da.from_array(sky_model.ref_freq, chunks=schunks)
                gauss_shape = da.from_array(sky_model.gauss_shape, chunks=schunks)

                lm = radec_to_lm(radec, field.PHASE_DIR.values[0][0])

                with warnings.catch_warnings():
                    # Ignore dask chunk warnings emitted when going from 1D
                    # inputs to a 2D space of chunks
                    warnings.simplefilter("ignore", category=da.PerformanceWarning)

                    vis = wsclean_predict(
                        ds.UVW.data,
                        lm,
                        source_type,
                        flux,
                        spi,
                        log_poly,
                        ref_freq,
                        gauss_shape,
                        frequency,
                    )

                    if args.expand_vis:
                        vis = expand_vis(vis, pol.NUM_CORR.values[0])

                # Assign visibilities to MODEL_DATA array on the dataset
                ods = ds.assign(**{args.output_column: (("row", "chan", "corr"), vis)})
                out_datasets.append(ods)

            # Write to table
            write = xds_to_zarr(out_datasets, args.output_store, columns=[args.output_column])
            write = annotate_datasets(write)


        for i, ds in enumerate(write):
            out_array = getattr(ds, args.output_column).data
            annotations = out_array.dask.layers[out_array.name].annotations
            assert annotations["dims"] == ("row", "chan", "corr")
            assert annotations["dataset_id"] == i, (annotations["dataset_id"], i)

        dask.compute(write, sync=True, optimize_graph=args.optimize_graph)
    
    elif backend == "heat":
        logging.info("Heat backend selected.")
        # TODO: Implement Heat-based visibility prediction
        raise NotImplementedError("The Heat backend is not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend}")
