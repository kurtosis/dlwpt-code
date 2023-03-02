import argparse
import sys
from util.logconf import logging

from torch.utils.data import DataLoader

from part2_redo.dsets import LunaDataset, get_ct_sample_size
from part2_redo.my_utils import enumerate_with_estimate

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--batch-size",
            help="Batch size for training.",
            default=1024,
            type=int,
        )
        parser.add_argument(
            "--num-workers",
            help="Number of workers for data loading",
            default=8,
            type=int,
        )
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info(f"Starting {type(self).__name__} {self.cli_args}")
        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str="seriesuid",
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )
        batch_iter = enumerate_with_estimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            pass


if __name__ == "__main__":
    LunaPrepCacheApp().main()
