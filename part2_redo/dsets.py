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
raw_cache = getCache("part2_ch13_dsets")


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
    annotations = pd.read_csv("data/part2/luna/annotations_with_malignancy.csv")
    # annotations["index_a"] = annotations.index
    annotations["is_nodule_bool"] = True
    annotations["has_annotation_bool"] = True
    candidates["is_nodule_bool"] = candidates["class"] == 1
    candidates_not_nodule = candidates[candidates["is_nodule_bool"] == False].copy()
    candidates_not_nodule["mal_bool"] = False
    candidates_not_nodule["has_annotation_bool"] = False
    candidates_not_nodule["diameter_mm"] = 0.0
    candidates = pd.concat([annotations, candidates_not_nodule])
    candidates = candidates[
        [
            "is_nodule_bool",
            "has_annotation_bool",
            "mal_bool",
            "diameter_mm",
            "seriesuid",
            "coordX",
            "coordY",
            "coordZ",
        ]
    ]
    return candidates.sort_values(candidates.columns.to_list(), ascending=False)
    # candidates = pd.merge(
    #     candidates, annotations, on="seriesuid", how="left", suffixes=("", "_a")
    # )
    #
    # candidates["dist_sq"] = (
    #         (candidates.coordX - candidates.coordX_a) ** 2
    #         + (candidates.coordY - candidates.coordY_a) ** 2
    #         + (candidates.coordZ - candidates.coordZ_a) ** 2
    # )
    #
    # candidates["dist_rank"] = candidates.groupby(by="index_c")["dist_sq"].rank()
    # candidates = candidates[(candidates.dist_rank == 1) | (candidates.dist_rank.isna())]
    # candidates["dist"] = np.sqrt(candidates["dist_sq"])
    # candidates["dist_rel"] = candidates["dist"] / candidates.diameter_mm
    # candidates.loc[candidates.dist_rel > 0.2985, "diameter_mm"] = 0
    # candidates.sort_values(
    #     ["is_nodule_bool", "diameter_mm"], ascending=False, inplace=True
    # )
    # candidates.reset_index(inplace=True, drop=True)
    # return candidates[
    #     ["is_nodule_bool", "diameter_mm", "seriesuid", "coordX", "coordY", "coordZ"]
    # ]


class Ct:
    def __init__(self, seriesuid):
        mhd_path = glob.glob(f"data-unversioned/part2/luna/subset*/{seriesuid}.mhd")[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.seriesuid = seriesuid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        self.candidates = get_candidate_df()
        self.candidates = self.candidates[self.candidates.seriesuid == seriesuid]
        self.pos_df = self.candidates[self.candidates.is_nodule_bool == True]
        self.pos_mask = self.build_annotation_mask()
        self.pos_indexes = self.pos_mask.sum(axis=(1, 2)).nonzero()[0].tolist()

    def build_annotation_mask(self, threshold_hu=-700):
        bounding_box_a = np.zeros_like(self.hu_a, dtype=np.bool)
        for _, candidate in self.pos_df.iterrows():
            center_irc = xyz2irc(
                (candidate.coordX, candidate.coordY, candidate.coordZ),
                self.origin_xyz,
                self.vx_size_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)
            index_radius = 2
            try:
                while (
                    self.hu_a[ci + index_radius, cr, cc] > threshold_hu
                    and self.hu_a[ci - index_radius, cr, cc] > threshold_hu
                ):
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while (
                    self.hu_a[ci, cr + row_radius, cc] > threshold_hu
                    and self.hu_a[ci, cr - row_radius, cc] > threshold_hu
                ):
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while (
                    self.hu_a[ci, cr, cc + col_radius] > threshold_hu
                    and self.hu_a[ci, cr, cc - col_radius] > threshold_hu
                ):
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            bounding_box_a[
                ci - index_radius : ci + index_radius + 1,
                cr - row_radius : cr + row_radius + 1,
                cc - col_radius : cc + col_radius + 1,
            ] = True
        mask_a = bounding_box_a & (self.hu_a > threshold_hu)
        return mask_a

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
                    self.seriesuid,
                    center_xyz,
                    self.origin_xyz,
                    self.vx_size_xyz,
                    center_irc,
                    axis,
                ]
            )

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.seriesuid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.seriesuid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.pos_mask[tuple(slice_list)]
        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct(seriesuid):
    return Ct(seriesuid)


@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(seriesuid, center_xyz, width_irc):
    ct = get_ct(seriesuid)
    ct_chunk, pos_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def get_ct_sample_size(seriesuid):
    ct = Ct(seriesuid)
    return int(ct.hu_a.shape[0]), ct.pos_indexes


def get_ct_augmented_candidate(
    augmentation_dict, seriesuid, center_xyz, width_irc, use_cache=True
):
    if use_cache:
        ct_chunk, pos_chunk, center_irc = get_ct_raw_candidate(
            seriesuid, center_xyz, width_irc
        )
    else:
        ct = get_ct(seriesuid)
        ct_chunk, pos_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
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
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False,
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
        seriesuid=None,
        sortby_str="random",
    ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        self.candidates = copy.copy(get_candidate_df())
        self.use_cache = True

        if seriesuid:
            self.candidates = self.candidates[self.candidates.seriesuid == seriesuid]
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
        elif sortby_str == "seriesuid":
            self.candidates.sort_values(
                ["seriesuid", "coordX", "coordY", "coordZ"], inplace=True
            )
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
                cand = self.neg_df.iloc[neg_ndx]
            else:
                pos_ndx %= len(self.pos_df)
                cand = self.pos_df.iloc[pos_ndx]
        else:
            cand = self.candidates.iloc[ndx]

        width_irc = (32, 48, 48)

        center_xyz = (
            cand.coordX,
            cand.coordY,
            cand.coordZ,
        )

        if self.augmentation_dict:
            candidate_t, center_irc = get_ct_augmented_candidate(
                self.augmentation_dict,
                cand.seriesuid,
                center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, pos_a, center_irc = get_ct_raw_candidate(
                cand.seriesuid,
                center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = get_ct(cand.seriesuid)
            candidate_a, pos_a, center_irc = ct.get_raw_candidate(
                center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor(
            [not cand.is_nodule_bool, cand.is_nodule_bool],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            cand.seriesuid,
            torch.tensor(center_irc),
        )


class PrepcacheLunaDataset(Dataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.candidates = copy.copy(get_candidate_df())
        # self.pos_df = self.candidates[self.candidates.is_nodule_bool == True]
        self.seen_set = ()
        self.candidates.sort_values("seriesuid", inplace=True)
        self.candidates.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, ndx):
        cand = self.candidates.iloc[ndx]
        center_xyz = (
            cand.coordX,
            cand.coordY,
            cand.coordZ,
        )
        get_ct_raw_candidate(
            cand.seriesuid,
            center_xyz,
            (7, 96, 96),
        )
        if cand.seriesuid not in self.seen_set:
            self.seen_set.add(cand.seriesuid)
            get_ct_sample_size(cand.seriesuid)
        return 0, 1


class Luna2dSegmentationDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        is_val_set_bool=None,
        seriesuid=None,
        full_ct_bool=False,
        context_slices_count=3,
    ):
        self.full_ct_bool = full_ct_bool
        self.context_slices_count = context_slices_count
        self.candidates = get_candidate_df()
        if seriesuid:
            self.series_list = [seriesuid]
        else:
            self.series_list = self.candidates["seriesuid"].unique().tolist()

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]

        self.sample_list = []
        for seriesuid in self.series_list:
            index_count, pos_indexes = get_ct_sample_size(seriesuid)
            if self.full_ct_bool:
                self.sample_list += [
                    (seriesuid, slice_ndx) for slice_ndx in range(index_count)
                ]
            else:
                self.sample_list += [
                    (seriesuid, slice_ndx) for slice_ndx in pos_indexes
                ]

        series_set = set(self.series_list)
        self.candidates = self.candidates[self.candidates.seriesuid.isin(series_set)]
        self.pos_df = self.candidates[self.candidates["is_nodule_bool"] == True]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        seriesuid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_full_slice(seriesuid, slice_ndx)

    def getitem_full_slice(self, seriesuid, slice_ndx):
        ct = get_ct(seriesuid)
        ct_t = torch.zeros((self.context_slices_count * 2 + 1, 512, 512))
        start_ndx = slice_ndx - self.context_slices_count
        end_ndx = slice_ndx + self.context_slices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))
        ct_t.clamp_(-1000, 1000)
        pos_t = torch.from_numpy(ct.pos_mask[slice_ndx]).unsqueeze(0)
        return ct_t, pos_t, ct.seriesuid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 20000

    def shuffle_samples(self):
        self.candidates = self.candidates.sample(frac=1).reset_index(drop=True)
        self.pos_df = self.pos_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, ndx):
        cand = self.pos_df.iloc[ndx % len(self.pos_df)]
        return self.getitem_training_crop(cand)

    def getitem_training_crop(self, cand):
        center_xyz = (
            cand.coordX,
            cand.coordY,
            cand.coordZ,
        )
        ct_a, pos_a, center_irc = get_ct_raw_candidate(
            cand.seriesuid,
            center_xyz,
            (7, 96, 96),
        )
        pos_a = pos_a[3:4]
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(
            ct_a[:, row_offset : row_offset + 64, col_offset : col_offset + 64]
        ).to(torch.float32)
        pos_t = torch.from_numpy(
            pos_a[:, row_offset : row_offset + 64, col_offset : col_offset + 64]
        ).to(torch.bool)
        # ).to(torch.long)
        slice_ndx = center_irc.index
        return ct_t, pos_t, cand.seriesuid, slice_ndx


clim = (-1000.0, 300)


def show_candidate(seriesuid, batch_ndx=None, **kwargs):
    ds = LunaDataset(seriesuid=seriesuid, **kwargs)
    # pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.is_nodule_bool]
    pos_list = ds.candidates[ds.candidates.is_nodule_bool]
    if batch_ndx is None:
        if not pos_list.empty:
            batch_ndx = pos_list.index[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = Ct(seriesuid)
    ct_t, pos_t, seriesuid, center_irc = ds[batch_ndx]
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

    print(seriesuid, batch_ndx, bool(pos_t[0]), pos_list)
