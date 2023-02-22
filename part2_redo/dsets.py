import copy
import functools
import glob
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from util.disk import getCache
from util.logconf import logging
from util.util import XyzTuple, xyz2irc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
raw_cache = getCache("part2_dsets")


@functools.lru_cache(1)
def get_candidate_df(require_on_disk=True):
    """
    Construct a dataframe with metadata for all candidates on disk.
    """
    candidates = pd.read_csv("data/part2/luna/candidates.csv")
    if require_on_disk:
        mhd_list = glob.glob("data-unversioned/part2/luna/subset*/*.mhd")
        on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
        candidates = candidates[candidates.seriesuid.isin(on_disk_set)]
    candidates["index_c"] = candidates.index
    annotations = pd.read_csv("data/part2/luna/annotations.csv")
    annotations["index_a"] = annotations.index
    candidates["is_nodule_bool"] = candidates["class"] == 1

    candidates = pd.merge(
        candidates, annotations, on="seriesuid", how="left", suffixes=("", "_a")
    )

    candidates["dist_sq"] = (
        (candidates.coordX - candidates.coordX_a) ** 2
        + (candidates.coordY - candidates.coordY_a) ** 2
        + (candidates.coordZ - candidates.coordZ_a) ** 2
    )

    candidates["dist_rank"] = candidates.groupby(by="index_c")["dist_sq"].rank()
    candidates = candidates[(candidates.dist_rank == 1) | (candidates.dist_rank.isna())]
    candidates["dist"] = np.sqrt(candidates["dist_sq"])
    candidates["dist_rel"] = candidates["dist"] / candidates.diameter_mm
    candidates.loc[candidates.dist_rel > 0.2985, "diameter_mm"] = 0
    candidates.sort_values(
        ["is_nodule_bool", "diameter_mm"], ascending=False, inplace=True
    )
    candidates.reset_index(inplace=True, drop=True)
    return candidates[
        ["is_nodule_bool", "diameter_mm", "seriesuid", "coordX", "coordY", "coordZ"]
    ]


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(f"data-unversioned/part2/luna/subset*/{series_uid}.mhd")[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vx_size_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.hu_a.shape[axis], repr(
                [
                    self.series_uid,
                    center_xyz,
                    self.origin_xyz,
                    self.vx_size_xyz,
                    center_irc,
                    axis,
                ]
            )

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(series_uid, center_xyz, width_irc):
    ct = get_ct(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc


def get_ct_augmented_candidate(
    augmentation_dict, series_uid, center_xyz, width_irc, use_cache=True
):
    if use_cache:
        ct_chunk, center_irc = get_ct_raw_candidate(series_uid, center_xyz, width_irc)
    else:
        ct = get_ct(series_uid)
        ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)
    transform_t = torch.eye(4)
    for i in range(3):
        if "flip" in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        if "offset" in augmentation_dict:
            offset_float = augmentation_dict["offset"]
            random_float = random.random() * 2 - 1
            transform_t[i, 3] = offset_float * random_float
        if "scale" in augmentation_dict:
            scale_float = augmentation_dict["scale"]
            random_float = random.random() * 2 - 1
            transform_t[i, i] *= 1.0 + scale_float + random_float
    if "rotate" in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        rotation_t = torch.tensor(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        transform_t @= rotation_t
    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32), ct_t.size(), align_corners=False
    )
    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode="border",
        align_corners=False,
    ).to("cpu")
    if "noise" in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict["noise"]
        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        is_val_set_bool=None,
        ratio_int=0,
        augmentation_dict=None,
        series_uid=None,
        sortby_str="random",
    ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        self.candidates = copy.copy(get_candidate_df())
        self.use_cache = True

        if series_uid:
            self.candidates = self.candidates[self.candidates.seriesuid == series_uid]
        elif is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidates = self.candidates[::val_stride]
            assert not self.candidates.empty
        elif val_stride > 0:
            self.candidates = self.candidates[self.candidates.index % val_stride != 0]
            assert not self.candidates.empty
        self.candidates.reset_index(inplace=True, drop=True)

        if sortby_str == "random":
            self.candidates = self.candidates.sample(frac=1).reset_index(drop=True)
        elif sortby_str == "label_and_size":
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.neg_df = copy.copy(
            self.candidates[self.candidates.is_nodule_bool == False]
        )
        self.neg_df.reset_index(drop=True, inplace=True)
        self.pos_df = copy.copy(self.candidates[self.candidates.is_nodule_bool == True])
        self.pos_df.reset_index(drop=True, inplace=True)

        log.info(
            f"{self!r}: {len(self.candidates)} {'validation' if is_val_set_bool else 'training'} samples"
        )

    def shuffle_samples(self):
        if self.ratio_int:
            self.neg_df = self.neg_df.sample(frac=1).reset_index(drop=True)
            self.pos_df = self.pos_df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        if self.ratio_int:
            return 20000
        else:
            return len(self.candidates)

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)
            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_df)
                candidate_series = self.neg_df.iloc[neg_ndx]
            else:
                pos_ndx %= len(self.pos_df)
                candidate_series = self.pos_df.iloc[pos_ndx]
        else:
            candidate_series = self.candidates.iloc[ndx]

        width_irc = (32, 48, 48)

        candidate_a, center_irc = get_ct_raw_candidate(
            # candidate_series.seriesuid, candidate_series.center_xyz, width_irc,
            candidate_series.seriesuid,
            (candidate_series.coordX, candidate_series.coordY, candidate_series.coordZ),
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not candidate_series.is_nodule_bool, candidate_series.is_nodule_bool],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidate_series.seriesuid,
            torch.tensor(center_irc),
        )


clim = (-1000.0, 300)


def show_candidate(series_uid, batch_ndx=None, **kwargs):
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    # pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.is_nodule_bool]
    pos_list = ds.candidates[ds.candidates.is_nodule_bool]
    if batch_ndx is None:
        if not pos_list.empty:
            batch_ndx = pos_list.index[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = Ct(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()

    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title("index {}".format(int(center_irc[0])), fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[int(center_irc[0])], clim=clim, cmap="gray")

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title("row {}".format(int(center_irc[1])), fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:, int(center_irc[1])], clim=clim, cmap="gray")
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title("col {}".format(int(center_irc[2])), fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:, :, int(center_irc[2])], clim=clim, cmap="gray")
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title("index {}".format(int(center_irc[0])), fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(ct_a[ct_a.shape[0] // 2], clim=clim, cmap="gray")

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title("row {}".format(int(center_irc[1])), fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(ct_a[:, ct_a.shape[1] // 2], clim=clim, cmap="gray")
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title("col {}".format(int(center_irc[2])), fontsize=30)
    for label in subplot.get_xticklabels() + subplot.get_yticklabels():
        label.set_fontsize(20)
    plt.imshow(ct_a[:, :, ct_a.shape[2] // 2], clim=clim, cmap="gray")
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title("slice {}".format(index), fontsize=30)
            for label in subplot.get_xticklabels() + subplot.get_yticklabels():
                label.set_fontsize(20)
            plt.imshow(ct_a[index], clim=clim, cmap="gray")

    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)
