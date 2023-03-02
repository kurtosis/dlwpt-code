import argparse
import datetime
import glob
import hashlib
import os
import logging
import shutil

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_erosion,
    center_of_mass,
    label,
    measurements,
    morphology,
)

# import scipy.ndimage.morphology as morphology
# import scipy.ndimage.measurements as measurements
import sys

from torch import nn
import torch
import torch.cuda
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from part2_redo.model import UNetWrapper
from part2_redo.dsets import (
    get_candidate_df,
    get_ct,
    LunaDataset,
    Luna2dSegmentationDataset,
)
from part2_redo.my_utils import enumerate_with_estimate
import part2_redo.model
from util.util import irc2xyz

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

log = logging.getLogger(__name__)


def match_and_score(detections, truth, threshold=0.5):
    true_nodules = truth[truth.is_nodule_bool == True].reset_index()
    truth_diams = true_nodules["diameter_mm"]
    truth_xyz = np.array(true_nodules[["coordX", "coordY", "coordZ"]])
    detected_xyz = np.array(detections[["coordX", "coordY", "coordZ"]])
    detections["d_class"] = 3
    detections.loc[detections["prob_mal"] < threshold, "d_class"] = 2
    detections.loc[detections["prob_nodule"] < threshold, "d_class"] = 1
    confusion = np.zeros((3, 4), dtype=int)
    if len(detections) == 0:
        confusion[1, 0] += true_nodules[true_nodules.mal_bool == False].shape[0]
        confusion[2, 0] += true_nodules[true_nodules.mal_bool == True].shape[0]
    elif len(truth) == 0:
        for _, det in detections.iterrows():
            confusion[0, det["d_class"]] += 1
    else:
        normalized_dists = (
            np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1)
            / truth_diams[:, None]
        )
        matches = normalized_dists < 0.7
        unmatched_detections = np.ones(len(detections), dtype=bool)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=int)
        for i_tn, i_detection in zip(*matches.nonzero()):
            matched_true_nodules[i_tn] = max(
                matched_true_nodules[i_tn], detections.loc[i_detection, "d_class"]
            )
            unmatched_detections[i_detection] = False
        for ud, dc in zip(unmatched_detections, detections["d_class"]):
            if ud:
                confusion[0, dc] += 1
        for tn, dc in zip(true_nodules.iterrows(), matched_true_nodules):
            confusion[2 if tn[1]["mal_bool"] else 1, dc] += 1
    return confusion


def print_confusion(label_str, confusions, do_mal):
    row_labels = ["non-nodules", "benign", "malignant"]
    if do_mal:
        col_labels = [
            "",
            "complete miss",
            "filtered out",
            "pred. benign",
            "pred. mal",
        ]
    else:
        col_labels = [
            "",
            "complete miss",
            "filtered out",
            "pred. nodule",
        ]
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    cell_width = 16
    f = "{:>" + str(cell_width) + "}"
    print(label_str)
    print(" | ".join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ""
        print(" | ".join([f.format(i) for i in r]))


class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--use-mps",
            help="Use MPS to run on MacBook GPU. (Note: Conv3D does not work w/ MPS)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use for training",
            default=4,
            type=int,
        )
        parser.add_argument(
            "--num-workers",
            help="Number of worker processes for background data loading",
            default=4,
            type=int,
        )
        parser.add_argument(
            "--run-validation",
            help="Run over validation rather than a single CT.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--include-train",
            help="Include data that was in the training set. (default: validation data only)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--segmentation-path",
            help="Path to the saved segmentation model",
            nargs="?",
            default="data/part2/models/seg_2020-01-26_19.45.12_w4d3c1-bal_1_nodupe-label_pos-d1_fn8-adam.best.state",
        )
        parser.add_argument(
            "--cls-model",
            help="What to model class name to use for the classifier.",
            action="store",
            default="LunaModel",
        )
        parser.add_argument(
            "--classification-path",
            help="Path to the saved classification model",
            nargs="?",
            default="data/part2/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state",
        )
        parser.add_argument(
            "--mal-model",
            help="What to model class name to use for the malignancy classifier.",
            action="store",
            default="LunaModel",
            # default='ModifiedLunaModel',
        )
        parser.add_argument(
            "--mal-path",
            help="Path to the saved malignancy classification model",
            nargs="?",
            default=None,
        )
        parser.add_argument(
            "--tb-prefix",
            default="redo_ch14",
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )
        parser.add_argument(
            "seriesuid",
            nargs="?",
            default=None,
            # default="1.3.6.1.4.1.14519.5.2.1.6279.6001.592821488053137951302246128864",
            help="Series UID to use.",
        )
        self.cli_args = parser.parse_args(sys_argv)
        self.use_cuda = torch.cuda.is_available()
        self.use_mps = self.cli_args.use_mps
        self.device = torch.device(
            "cuda" if self.use_cuda else "mps" if self.use_mps else "cpu"
        )
        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.init_model_path("seg")
        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.init_model_path("cls")
        self.seg_model, self.cls_model, self.mal_model = self.init_models()

    def init_model_path(self, type_str):
        local_path = os.path.join(
            "data-unversioned",
            "part2",
            "models",
            "p2ch13",
            type_str + "_*_*.best.state",
        )
        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                "data", "part2", "models", type_str + "*_*.*.state"
            )
        else:
            pretrained_path = None
        file_list.sort()
        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise

    def init_models(self):
        log.debug(self.cli_args.segmentation_path)
        seg_dict = torch.load(self.cli_args.segmentation_path, map_location=self.device)
        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode="upconv",
        )
        seg_model.load_state_dict(seg_dict["model_state"])
        seg_model.eval()

        log.debug(self.cli_args.classification_path)
        cls_dict = torch.load(
            self.cli_args.classification_path, map_location=self.device
        )
        model_cls = getattr(part2_redo.model, self.cli_args.cls_model)
        cls_model = model_cls()
        cls_model.load_state_dict(cls_dict["model_state"])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)
            seg_model.to(self.device)
            cls_model.to(self.device)
        elif self.use_mps:
            seg_model.to(self.device)
            cls_model.to(self.device)

        if self.cli_args.mal_path:
            model_cls = getattr(part2_redo.model, self.cli_args.mal_model)
            mal_model = model_cls()
            mal_dict = torch.load(self.cli_args.mal_path, map_location=self.device)
            mal_model.load_state_dict(mal_dict["model_state"])
            mal_model.eval()
            if self.use_cuda or self.use_mps:
                mal_model.to(self.device)
        else:
            mal_model = None
        return seg_model, cls_model, mal_model

    def init_segmentation_dl(self, seriesuid):
        seg_ds = Luna2dSegmentationDataset(
            context_slices_count=3,
            seriesuid=seriesuid,
            full_ct_bool=True,
        )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size
            * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return seg_dl

    def init_classification_dl(self, candidates):
        cls_ds = LunaDataset(sortby_str="seriesuid", candidates=candidates)
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size
            * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return cls_dl

    def segment_ct(self, ct, seriesuid):
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            seg_dl = self.init_segmentation_dl(seriesuid)
            for input_t, _, _, slice_ndx_list in seg_dl:
                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g)
                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy()
            mask_a = output_a > 0.5
            mask_a = binary_erosion(mask_a, iterations=1)
        return mask_a

    def group_segmentation_output(self, seriesuid, ct, clean_a):
        candidate_label_a, candidate_count = label(clean_a)
        center_irc_list = center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidate_label_a,
            index=np.arange(1, candidate_count + 1),
        )
        centers_xyz = [
            irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vx_size_xyz,
                ct.direction_a,
            )
            for center_irc in center_irc_list
        ]
        candidates = pd.DataFrame(
            {
                "is_nodule_bool": False,
                "has_annotation_bool": False,
                "mal_bool": False,
                "diameter_mm": 0.0,
                "seriesuid": seriesuid,
                "coordX": [xyz[0] for xyz in centers_xyz],
                "coordY": [xyz[1] for xyz in centers_xyz],
                "coordZ": [xyz[2] for xyz in centers_xyz],
            }
        )
        return candidates

    def classify_candidates(self, ct, candidates):
        cls_dl = self.init_classification_dl(candidates)
        classifications = []
        for _, batch_tup in enumerate(cls_dl):
            input_t, _, _, center_list = batch_tup
            input_g = input_t.to(self.device)
            with torch.no_grad():
                _, prob_nodule_g = self.cls_model(input_g)
                if self.mal_model is not None:
                    _, prob_mal_g = self.mal_model(input_g)
                else:
                    prob_mal_g = torch.zeros_like(prob_nodule_g)
            zip_iter = zip(
                center_list, prob_nodule_g[:, 1].tolist(), prob_mal_g[:, 1].tolist()
            )
            for center_irc, prob_nodule, prob_mal in zip_iter:
                coordX, coordY, coordZ = irc2xyz(
                    center_irc,
                    direction_a=ct.direction_a,
                    origin_xyz=ct.origin_xyz,
                    vxSize_xyz=ct.vx_size_xyz,
                )
                cls_tup = (
                    prob_nodule,
                    prob_mal,
                    coordX,
                    coordY,
                    coordZ,
                    center_irc,
                )
                classifications.append(cls_tup)
        return pd.DataFrame(
            classifications,
            columns=[
                "prob_nodule",
                "prob_mal",
                "coordX",
                "coordY",
                "coordZ",
                "center_irc",
            ],
        )

    def main(
        self,
    ):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        val_ds = LunaDataset(
            val_stride=10,
            is_val_set_bool=True,
        )
        val_set = set(val_ds.candidates["seriesuid"])
        df_all_candidates = get_candidate_df()
        all_seriesuids = set(df_all_candidates["seriesuid"])

        if self.cli_args.seriesuid:
            series_set = set(self.cli_args.seriesuid.split(","))
        else:
            series_set = all_seriesuids

        if self.cli_args.include_train:
            train_list = sorted(series_set - val_set)
        else:
            train_list = []

        val_list = sorted(series_set & val_set)
        series_iter = enumerate_with_estimate(
            val_list + train_list,
            "Series",
        )
        all_confusion = np.zeros((3, 4), dtype=int)

        for _, seriesuid in series_iter:
            ct = get_ct(seriesuid)
            mask_a = self.segment_ct(ct, seriesuid)
            candidates = self.group_segmentation_output(seriesuid, ct, mask_a)
            classifications = self.classify_candidates(ct, candidates)
            if not self.cli_args.run_validation:
                print(f"Found nodule candidates in {seriesuid}:")
                for _, (
                    prob,
                    prob_mal,
                    coordX,
                    coordY,
                    coordZ,
                    center_irc,
                ) in classifications.iterrows():
                    if prob > 0.5:
                        s = f"nodule prob {prob: .3f}, "
                        if self.mal_model:
                            s += f"malignancy prob {prob_mal: .3f}, "
                        s += f"center xyz {coordX} {coordY} {coordZ}"
                        print(s)
            if seriesuid in all_seriesuids:
                one_confusion = match_and_score(
                    classifications,
                    df_all_candidates[df_all_candidates.seriesuid == seriesuid],
                )
                all_confusion += one_confusion
                print_confusion(seriesuid, one_confusion, self.mal_model is not None)
            print_confusion("Total", all_confusion, self.mal_model is not None)


if __name__ == "__main__":
    NoduleAnalysisApp().main()
