import datetime
import time
from util.logconf import logging

log = logging.getLogger(__name__)


def enumerate_with_estimate(
    iter, desc_str, start_ndx=0, print_ndx=4, backoff=None, iter_len=None,
):
    if iter_len is None:
        iter_len = len(iter)
    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning(f"{desc_str}-----/{iter_len}, starting")
    start_ts = time.time()

    for current_ndx, item in enumerate(iter):
        yield current_ndx, item
        if current_ndx == print_ndx:
            duration_sec = (
                (time.time() - start_ts)
                / (current_ndx - start_ndx + 1)
                * (iter_len - start_ndx)
            )
            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)
            log.info(
                f"{desc_str} {current_ndx}/{iter_len}, done at {str(done_dt).rsplit('.',1)[0]}, {str(done_td).rsplit('.',1)[0]}"
            )
            print_ndx *= backoff
        if current_ndx + 1 == start_ndx:
            start_ts = time.time()
    log.warning(
        f"{desc_str}-----/{iter_len}, done at {str(datetime.datetime.now()).rsplit('.',1)[0]}"
    )
