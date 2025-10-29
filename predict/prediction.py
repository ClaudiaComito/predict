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

import logging

logging.basicConfig(level=logging.INFO)

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
                # convert integer source type to "POINT" or "GAUSSIAN"
                source_type_int = da.from_array(sky_model.source_type, chunks=schunks)
                source_type = da.where(source_type_int == 1, "GAUSS", "POINT").astype("<U5")
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

        import heat as ht
        from predict.heat_kernels import heat_radec_to_lm, heat_wsclean_predict

        logging.info(f"Heat backend selected on device {args.device}.")
        ht.devices.use_device(args.device)

        # Ingest sky model and replicate on all processes (split=None)
        ht_source_type = ht.array(sky_model.source_type, split=None)
        ht_radec = ht.array(sky_model.radec, split=None)
        ht_flux = ht.array(sky_model.flux, split=None)
        ht_spi = ht.array(sky_model.spi, split=None)
        ht_ref_freq = ht.array(sky_model.ref_freq, split=None)
        ht_log_poly = ht.array(sky_model.log_poly, split=None)
        ht_gauss_shape = ht.array(sky_model.gauss_shape, split=None)

        # Read all UVW data partitions in parallel using wildcard path

        # The variable pattern "MAIN_*/UVW" instructs ht.load to find all directories
        # matching MAIN_*, load the UVW array from each, and concatenate them
        # along the specified split axis (0, the row axis).
        ht_uvw = ht.load(args.store, variable="MAIN_*/UVW", split=0, dtype=ht.float32)

        # In this particular example, we know we have a single FIELD and DATA_DESCRIPTION for all MAIN_* partitions.
        field_ds = xds_from_storage_table(f"{args.store}::FIELD")[0].compute()
        ht_phase_dir = ht.array(field_ds.PHASE_DIR.values[0][0], dtype=ht.float32, split=None)

        # Create frequency array
        nchan = args.dimensions["chan"]
        ht_frequency = ht.linspace(0.856e9, 2 * 0.856e9, nchan, dtype=ht.float32, split=None)

        # Get batch sizes from the chunks argument
        source_batch_size = args.chunks["source"]
        chan_batch_size = args.chunks["chan"]

        # Convert radec to lm coordinates
        ht_lm = heat_radec_to_lm(ht_radec, ht_phase_dir)

        # Call the Heat prediction kernel
        ht_vis = heat_wsclean_predict(ht_uvw, ht_lm, ht_source_type, ht_flux, ht_spi, 
                                        ht_log_poly, ht_ref_freq, ht_gauss_shape, ht_frequency,
                                        source_batch_size=source_batch_size,
                                        chan_batch_size=chan_batch_size)

        # Save the resulting distributed tensor to a Zarr store
        logging.info("Computation complete. Writing output to %s", args.output_store)
        ht_vis.save(args.output_store, overwrite=True)
        logging.info("Output successfully written.")

    else:
        raise ValueError(f"Unknown backend: {backend}")
    

