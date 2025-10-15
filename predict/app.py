import argparse
from contextlib import ExitStack
from datetime import datetime
import logging
from pprint import pformat
import sys
from typing import Dict, Iterable
from unittest import mock

import dask
import dask.config
from dask.distributed import Client, LocalCluster
from dask.distributed.client import performance_report

from daskms.fsspec_store import DaskMSStore

from predict.pinned_plugin import install_pinned_plugin
from predict.autorestrictor import install_autorestrictor_plugin
from predict.prediction import predict_vis
from predict.sky_model import generate_sky_model
from predict.utils import _fuse_annotations


class Application:
    # Default number of channels and sources to simulate
    DEFAULT_CHANS = 4096
    DEFAULT_SOURCES = 1000

    # Default chunking for the predict dimensions
    DEFAULT_ROW_CHUNKS = 10000
    DEFAULT_CHAN_CHUNKS = 64
    DEFAULT_SOURCE_CHUNKS = 100

    # Default dimension argument
    DEFAULT_DIMENSIONS = f"{{chan: {DEFAULT_CHANS}," f"source: {DEFAULT_SOURCES}}}"

    # Default chunk argument
    DEFAULT_CHUNKS = (
        f"{{"
        f"row: {DEFAULT_ROW_CHUNKS},"
        f"chan: {DEFAULT_CHAN_CHUNKS},"
        f"source: {DEFAULT_SOURCE_CHUNKS}"
        f"}}"
    )

    # Scheduler options pertinent to the plugins we test
    PLUGIN_OPTIONS_SCHEDULER_OPTIONS = {
        "distributed.scheduler.worker-saturation": "inf",
        "distributed.scheduler.work-stealing": False,
    }

    def __init__(self, args: Iterable[str]):
        logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
        self.args = self.parse_args(args)

    @staticmethod
    def parse_dim_dict(dims: str) -> Dict[str, int]:
        msg = f"'{dims}' does not conform to {{d1: s1, d2: s2, ..., dn: sn}}"
        err = argparse.ArgumentTypeError(msg)
        dims = dims.strip()

        if not dims[0] == "{" and dims[-1] == "}":
            raise err

        result = {}

        for dim in [d.strip() for d in dims[1:-1].split(",")]:
            bits = [b.strip() for b in dim.split(":")]

            if len(bits) != 2:
                raise err

            try:
                result[bits[0]] = int(bits[1])
            except ValueError:
                raise err

        return result

    @staticmethod
    def make_report_url(report_prefix: str) -> str:
        return datetime.now().strftime(f"{report_prefix}-%Y%m%d-%H%M%S.html")

    @staticmethod
    def parse_args(args: Iterable[str]) -> argparse.Namespace:
        p = argparse.ArgumentParser()
        p.add_argument("store", help="Measurement Set store")
        p.add_argument("--output-store", required=True, type=DaskMSStore)
        p.add_argument("--output-column", default="MODEL_DATA")
        p.add_argument("--address", help="distributed scheduler address")
        p.add_argument(
            "--workers", help="number of distributed workers", type=int, default=8
        )
        p.add_argument(
            "--dimensions",
            default=Application.DEFAULT_DIMENSIONS,
            type=Application.parse_dim_dict,
        )

        p.add_argument(
            "--chunks",
            default=Application.DEFAULT_CHUNKS,
            type=Application.parse_dim_dict,
        )

        p.add_argument("--report-prefix", default="report",
                       type=Application.make_report_url)

        p.add_argument("--expand-vis", action="store_true", default=False)
        p.add_argument("--optimize-graph", action="store_true", default=False)
        p.add_argument("--plugin", choices=["none", "pinned", "autorestrictor"],
                       default="none")
        p.add_argument("--backend", choices=["dask", "heat"],
                       default="dask", help="Computational backend to use.")

        args = p.parse_args(args)

        if args.output_store.exists():
            logging.warning("Removing existing output store %s", args.output_store)
            args.output_store.rm(recursive=True)
        args.dimensions.setdefault("chan", Application.DEFAULT_CHANS)
        args.dimensions.setdefault("source", Application.DEFAULT_SOURCES)

        args.chunks.setdefault("row", Application.DEFAULT_ROW_CHUNKS)
        args.chunks.setdefault("chan", Application.DEFAULT_CHAN_CHUNKS)
        args.chunks.setdefault("source", Application.DEFAULT_SOURCE_CHUNKS)

        logging.info("Command Line Arguments")
        logging.info("----------------------")

        for k, v in vars(args).items():
            logging.info("%s: %s", k, v)

        logging.info("----------------------")

        return args

    @staticmethod
    def get_client(args: argparse.Namespace, stack: ExitStack) -> Client:
        if not args.address:
            cluster = stack.enter_context(
                LocalCluster(
                    n_workers=args.workers, processes=True, threads_per_worker=1
                )
            )
            address = cluster.scheduler_address
            logging.info("Created LocalCluster with Scheduler at %s", address)

        else:
            address = args.address
            logging.info("Connected to Distributed Scheduler at %s", address)

        return stack.enter_context(Client(address))

    def run(self):
        with ExitStack() as stack:
            if self.args.backend == "dask":
                logging.info("dask configuration")
                logging.info(pformat(dask.config.config))
                stack.enter_context(
                    mock.patch("dask.blockwise._fuse_annotations", _fuse_annotations)
                )
                stack.enter_context(
                    dask.config.set({
                        "distributed.scheduler.dashboard.tasks.task-stream-length": sys.maxsize})
                )

                client = self.get_client(self.args, stack)

                if self.args.plugin == "autorestrictor":
                    stack.enter_context(dask.config.set(self.PLUGIN_OPTIONS_SCHEDULER_OPTIONS))
                    client.amm.stop()  # Disable active memory manager
                    client.run_on_scheduler(install_autorestrictor_plugin)
                elif self.args.plugin == "pinned":
                    stack.enter_context(dask.config.set(self.PLUGIN_OPTIONS_SCHEDULER_OPTIONS))
                    client.amm.stop()  # Disable active memory manager
                    client.run_on_scheduler(install_pinned_plugin)
                elif self.args.plugin == "none":
                    pass
                else:
                    raise ValueError(f"Unhandled plugin {self.arg.plugin}")

                logging.info("Waiting for %d workers to be ready", self.args.workers)
                client.wait_for_workers(self.args.workers)

            logging.info(
                "Generating sky model of %s sources", self.args.dimensions["source"]
            )
            model = generate_sky_model(self.args)

            logging.info("Predicting Visibilities")
            logging.info("Storing report at %s", self.args.report_prefix)
            stack.enter_context(performance_report(self.args.report_prefix))
            predict_vis(self.args, model, backend=self.args.backend)
            logging.info("Done")


def main():
    Application(sys.argv[1:]).run()


if __name__ == "__main__":
    main()
