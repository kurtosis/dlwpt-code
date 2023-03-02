import argparse
import datetime
import hashlib
import os
# import logging
import shutil

from matplotlib import pyplot
import numpy as np
import sys

from torch import nn
import torch
import torch.cuda
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import part2_redo.dsets
# import p2ch14.dsets
import part2_redo.model
# import p2ch14.model

# from part2_redo.dsets import (
#     LunaDataset,
#     Luna2dSegmentationDataset,
#     TrainingLuna2dSegmentationDataset,
#     get_ct,
# )
from part2_redo.my_utils import enumerate_with_estimate
from util.util import enumerateWithEstimate
from util.logconf import logging
# from part2_redo.model import LunaModel, UNetWrapper, SegmentationAugmentation
# from p2ch14.model import LunaModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# METRICS_LABEL_NDX = 0
# METRICS_PRED_NDX = 1
# METRICS_LOSS_NDX = 2
# METRICS_SIZE = 3

# ch 14 metrics
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PRED_P_NDX = 2
METRICS_LOSS_CH14_NDX = 3
METRICS_SIZE = 4

# METRICS_LOSS_SEG_NDX = 1
# METRICS_TP_NDX = 7
# METRICS_FN_NDX = 8
# METRICS_FP_NDX = 9
# METRICS_SIZE_SEG = 10


# class LunaTrainingApp:
#     def __init__(self, sys_argv=None):
#         if sys_argv is None:
#             sys_argv = sys.argv[1:]
#         parser = argparse.ArgumentParser()
#         parser.add_argument(
#             "--use-mps",
#             help="Use MPS to run on MacBook GPU. (Note: Conv3D does not work w/ MPS)",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--num_workers",
#             help="Number of workers for background data loading",
#             default=8,
#             type=int,
#         )
#         parser.add_argument(
#             "--batch-size",
#             help="Batch size to use for training",
#             default=32,
#             type=int,
#         )
#         parser.add_argument(
#             "--epochs",
#             help="Number of epochs to train for",
#             default=1,
#             type=int,
#         )
#         parser.add_argument(
#             "--balanced",
#             help="Balance the training data to 50/50 pos/neg.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augmented",
#             help="Augment the training data",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-flip",
#             help="Augment the training data with random flip",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-offset",
#             help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-scale",
#             help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-rotate",
#             help="Augment the training data by randomly rotating the data around the head-foot axis.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-noise",
#             help="Augment the training data by randomly adding noise to the data.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--tb-prefix",
#             default="part2_redo",
#             help="Data prefix to use for Tensorboard run.",
#         )
#         parser.add_argument(
#             "comment",
#             help="Comment suffix for Tensorboard run.",
#             nargs="?",
#             default="dlwpt",
#         )
#         self.cli_args = parser.parse_args(sys_argv)
#         self.use_cuda = torch.cuda.is_available()
#         self.use_mps = self.cli_args.use_mps
#         self.device = torch.device(
#             "cuda" if self.use_cuda else "mps" if self.use_mps else "cpu"
#         )
#         self.model = self.init_model()
#         self.optimizer = self.init_optimizer()
#         self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
#         self.trn_writer = None
#         self.val_writer = None
#         self.total_training_samples_count = 0
#         self.augmentation_dict = {}
#         if self.cli_args.augmented or self.cli_args.augment_flip:
#             self.augmentation_dict["flip"] = True
#         if self.cli_args.augmented or self.cli_args.augment_offset:
#             self.augmentation_dict["offset"] = 0.1
#         if self.cli_args.augmented or self.cli_args.augment_scale:
#             self.augmentation_dict["scale"] = 0.2
#         if self.cli_args.augmented or self.cli_args.augment_rotate:
#             self.augmentation_dict["rotate"] = True
#         if self.cli_args.augmented or self.cli_args.augment_noise:
#             self.augmentation_dict["noise"] = 25.0
#
#     def init_model(self):
#         model = LunaModel()
#         if self.use_cuda:
#             log.info(f"Using CUDA; {torch.cuda.device_count()} devices")
#             if torch.cuda.device_count() > 1:
#                 model = nn.DataParallel(model)
#             model = model.to(self.device)
#         elif self.use_mps:
#             log.info(f"Using MPS")
#             model = model.to(self.device)
#         return model
#
#     def init_optimizer(self):
#         return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
#
#     def init_dl(self, val=False):
#         if val:
#             ds = LunaDataset(
#                 val_stride=10,
#                 is_val_set_bool=val,
#             )
#         else:
#             ds = LunaDataset(
#                 val_stride=10,
#                 is_val_set_bool=val,
#                 ratio_int=int(self.cli_args.balanced),
#                 augmentation_dict=self.augmentation_dict,
#             )
#         batch_size = self.cli_args.batch_size
#         if self.use_cuda:
#             batch_size *= torch.cuda.device_count()
#         dl = DataLoader(
#             ds,
#             batch_size=batch_size,
#             num_workers=self.cli_args.num_workers,
#             pin_memory=self.use_cuda,
#             # pin_memory=self.use_cuda | self.use_mps,
#         )
#         return dl
#
#     def init_tensorboard_writers(self):
#         if self.trn_writer is None:
#             log_dir = os.path.join("runs", self.cli_args.tb_prefix, self.time_str)
#             self.trn_writer = SummaryWriter(
#                 log_dir=log_dir + "-trn_cls-" + self.cli_args.comment
#             )
#             self.val_writer = SummaryWriter(
#                 log_dir=log_dir + "-val_cls-" + self.cli_args.comment
#             )
#
#     def do_training(self, epoch_ndx, train_dl):
#         trn_metrics_g = torch.zeros(
#             METRICS_SIZE,
#             len(train_dl.dataset),
#             device=self.device,
#         )
#         self.model.train()
#         train_dl.dataset.shuffle_samples()
#         batch_iter = enumerate_with_estimate(
#             train_dl,
#             f"E{epoch_ndx} training",
#             start_ndx=train_dl.num_workers,
#         )
#         for batch_ndx, batch_tup in batch_iter:
#             self.optimizer.zero_grad()
#             loss_var = self.compute_batch_loss(
#                 batch_ndx,
#                 batch_tup,
#                 train_dl.batch_size,
#                 trn_metrics_g,
#             )
#             loss_var.backward()
#             self.optimizer.step()
#         self.total_training_samples_count += len(train_dl.dataset)
#         return trn_metrics_g.to("cpu")
#
#     def do_validation(self, epoch_ndx, val_dl):
#         with torch.no_grad():
#             self.model.eval()
#             val_metrics_g = torch.zeros(
#                 METRICS_SIZE,
#                 len(val_dl.dataset),
#                 device=self.device,
#             )
#             batch_iter = enumerate_with_estimate(
#                 val_dl,
#                 f"E{epoch_ndx} Validation",
#                 start_ndx=val_dl.num_workers,
#             )
#             for batch_ndx, batch_tup in batch_iter:
#                 self.compute_batch_loss(
#                     batch_ndx,
#                     batch_tup,
#                     val_dl.batch_size,
#                     val_metrics_g,
#                 )
#         return val_metrics_g.to("cpu")
#
#     def main(self):
#         log.info(f"Starting {type(self).__name__}, {self.cli_args}")
#         train_dl = self.init_dl()
#         val_dl = self.init_dl(val=True)
#         for epoch_ndx in range(1, self.cli_args.epochs + 1):
#             log.info(
#                 f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)} trn, {len(val_dl)} "
#                 f"val batches of size {self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}"
#             )
#             trn_metrics_t = self.do_training(epoch_ndx, train_dl)
#             self.log_metrics(epoch_ndx, "trn", trn_metrics_t)
#             val_metrics_t = self.do_validation(epoch_ndx, val_dl)
#             self.log_metrics(epoch_ndx, "val", val_metrics_t)
#         if hasattr(self, "trn_writer"):
#             self.trn_writer.close()
#             self.val_writer.close()
#
#     def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
#         input_t, label_t, series_list, _center_list = batch_tup
#         input_g = input_t.to(self.device, non_blocking=True)
#         label_g = label_t.to(self.device, non_blocking=True)
#         logits_g, probability_g = self.model(input_g)
#         loss_func = nn.CrossEntropyLoss(reduction="none")
#         loss_g = loss_func(logits_g, label_g[:, 1])
#         start_ndx = batch_ndx * batch_size
#         end_ndx = start_ndx + label_t.size(0)
#         metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
#         metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
#         metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()
#         return loss_g.mean()
#
#     def log_metrics(self, epoch_ndx, mode_str, metrics_t, threshold=0.5):
#         self.init_tensorboard_writers()
#         log.info(f"E{epoch_ndx} {type(self).__name__}")
#         neg_label_mask = metrics_t[METRICS_LABEL_NDX] <= threshold
#         neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= threshold
#         pos_label_mask = ~neg_label_mask
#         pos_pred_mask = ~neg_pred_mask
#         neg_count = int(neg_label_mask.sum())
#         pos_count = int(pos_label_mask.sum())
#         neg_correct = int((neg_label_mask & neg_pred_mask).sum())
#         pos_correct = int((pos_label_mask & pos_pred_mask).sum())
#         pos_incorrect = neg_count - neg_correct
#         neg_incorrect = pos_count - pos_correct
#         metrics_dict = {
#             "loss/all": metrics_t[METRICS_LOSS_NDX].mean(),
#             "loss/neg": metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean(),
#             "loss/pos": metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean(),
#             "correct/all": (
#                 (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
#             ),
#             "correct/neg": neg_correct / np.float32(neg_count) * 100,
#             "correct/pos": pos_correct / np.float32(pos_count) * 100,
#             "pr/precision": pos_correct / np.float32(pos_correct + pos_incorrect),
#             "pr/recall": pos_correct / np.float32(pos_correct + neg_incorrect),
#         }
#         metrics_dict["pr/f1_score"] = (
#             2
#             * metrics_dict["pr/precision"]
#             * metrics_dict["pr/recall"]
#             / (metrics_dict["pr/precision"] + metrics_dict["pr/recall"])
#         )
#
#         log.info(
#             f"E{epoch_ndx} {mode_str} {metrics_dict['loss/all']:.4f} loss, {metrics_dict['correct/all']:-5.1f}% correct"
#         )
#         log.info(
#             f"E{epoch_ndx} {mode_str} {metrics_dict['pr/precision']:.4f} precision, {metrics_dict['pr/recall']:.4f} recall, {metrics_dict['pr/f1_score']:.4f} f1 score"
#         )
#         log.info(
#             f"E{epoch_ndx} {mode_str}_neg {metrics_dict['loss/neg']:.4f} loss, {metrics_dict['correct/neg']:-5.1f}% correct ({neg_correct} of {neg_count}"
#         )
#         log.info(
#             f"E{epoch_ndx} {mode_str}_pos {metrics_dict['loss/pos']:.4f} loss, {metrics_dict['correct/pos']:-5.1f}% correct ({pos_correct} of {pos_count}"
#         )
#
#         writer = getattr(self, mode_str + "_writer")
#         for k, v in metrics_dict.items():
#             writer.add_scalar(k, v, self.total_training_samples_count)
#
#         writer.add_pr_curve(
#             "pr",
#             metrics_t[METRICS_LABEL_NDX],
#             metrics_t[METRICS_PRED_NDX],
#             self.total_training_samples_count,
#         )
#
#         bins = [x / 50.0 for x in range(51)]
#         # neg_hist_mask = neg_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
#         # pos_hist_mask = pos_label_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)
#         neg_hist_mask = neg_label_mask
#         pos_hist_mask = pos_label_mask
#
#         if neg_hist_mask.any():
#             writer.add_histogram(
#                 "is_neg",
#                 metrics_t[METRICS_PRED_NDX, neg_hist_mask],
#                 self.total_training_samples_count,
#                 bins=bins,
#             )
#         if pos_hist_mask.any():
#             writer.add_histogram(
#                 "is_pos",
#                 metrics_t[METRICS_PRED_NDX, pos_hist_mask],
#                 self.total_training_samples_count,
#                 bins=bins,
#             )


class ClassificationTrainingApp:
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
            "--num_workers",
            help="Number of workers for background data loading",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use for training",
            default=24,
            type=int,
        )
        parser.add_argument(
            "--epochs",
            help="Number of epochs to train for",
            default=1,
            type=int,
        )
        parser.add_argument('--dataset',
            help="What dataset to feed the model.",
            action='store',
            default='LunaDataset',
        )
        parser.add_argument('--model',
            help="What model class name to use.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument(
            "--tb-prefix",
            default="redo_ch14",
            help="Data prefix to use for Tensorboard run.",
        )
        parser.add_argument(
            "comment",
            help="Comment suffix for Tensorboard run.",
            nargs="?",
            default="redo",
        )
        parser.add_argument(
            "--finetune",
            help="Start finetuning from this model.",
            default="",
        )
        parser.add_argument(
            "--finetune-depth",
            help="Number of blocks (counted from the head) to include in finetuning",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--malignant",
            help="Train the model to classify nodules as benign or malignant.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augmented",
            help="Augment the training data",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "--augment-flip",
            help="Augment the training data with random flip",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment-offset",
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment-scale",
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment-rotate",
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment-noise",
            help="Augment the training data by randomly adding noise to the data.",
            action="store_true",
            default=False,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.total_training_samples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict["flip"] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict["offset"] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict["scale"] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict["rotate"] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict["noise"] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.use_mps = self.cli_args.use_mps
        self.device = torch.device(
            "cuda" if self.use_cuda else "mps" if self.use_mps else "cpu"
        )

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model_cls = getattr(part2_redo.model, self.cli_args.model)
        # model_cls = getattr(p2ch14.model, self.cli_args.model)
        model = model_cls()

        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location="cpu")
            model_blocks = [
                n
                for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth :]
            log.info(
                f"Finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}"
            )
            model.load_state_dict(
                {
                    k: v
                    for k, v in d["model_state"].items()
                    if k.split(".")[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in model.named_parameters():
                if n.split(".")[0] not in finetune_blocks:
                    p.requires_grad_(False)
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        elif self.use_mps:
            log.info(f"Using MPS")
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        lr = 0.003 if self.cli_args.finetune else 0.001
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def init_dl(self, val=False):
        ds_cls = getattr(part2_redo.dsets, self.cli_args.dataset)
        # ds_cls = getattr(p2ch14.dsets, self.cli_args.dataset)
        if val:
            ds = ds_cls(
                val_stride=10,
                is_val_set_bool=val,
                # isValSet_bool=False,
            )
        else:
            ds = ds_cls(
                val_stride=10,
                is_val_set_bool=val,
                # isValSet_bool=False,
                ratio_int=1,
            )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            # pin_memory=self.use_cuda | self.use_mps,
        )
        return dl

    def init_tensorboard_writers(self):
        if self.trn_writer is None:
            log_dir = os.path.join("runs", self.cli_args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + "-trn_cls-" + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir + "-val_cls-" + self.cli_args.comment
            )

    def do_training(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffle_samples()
        # train_dl.dataset.shuffleSamples()
        trn_metrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )
        batch_iter = enumerate_with_estimate(
        # batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} training",
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.compute_batch_loss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trn_metrics_g,
            )
            loss_var.backward()
            self.optimizer.step()
        self.total_training_samples_count += len(train_dl.dataset)
        return trn_metrics_g.to("cpu")

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )
            batch_iter = enumerate_with_estimate(
            # batch_iter = enumerateWithEstimate(
                val_dl,
                f"E{epoch_ndx} Validation",
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics_g,
                    augment=False,
                )
        return val_metrics_g.to("cpu")

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        train_dl = self.init_dl()
        val_dl = self.init_dl(val=True)
        best_score = 0.0
        validation_cadence = 5 if not self.cli_args.finetune else 1
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info(
                f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)} trn, {len(val_dl)} "
                f"val batches of size {self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}"
            )
            trn_metrics_t = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, "trn", trn_metrics_t)
            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                val_metrics_t = self.do_validation(epoch_ndx, val_dl)
                score = self.log_metrics(epoch_ndx, "val", val_metrics_t)
                best_score = max(score, best_score)
                self.save_model("cls", epoch_ndx, score == best_score)

        if hasattr(self, "trn_writer"):
            self.trn_writer.close()
            self.val_writer.close()

    def compute_batch_loss(
        self, batch_ndx, batch_tup, batch_size, metrics_g, augment=True
    ):
        input_t, label_t, index_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)

        if augment:
            input_g = part2_redo.model.augment3d(input_g)
            # input_g = p2ch14.model.augment3d(input_g)

        logits_g, probability_g = self.model(input_g)
        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1], reduction="none")
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        _, pred_label_g = torch.max(probability_g, dim=1, keepdim=False, out=None)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = pred_label_g
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_CH14_NDX, start_ndx:end_ndx] = loss_g
        return loss_g.mean()

    def log_metrics(self, epoch_ndx, mode_str, metrics_t, threshold=0.5):
        self.init_tensorboard_writers()
        log.info(f"E{epoch_ndx} {type(self).__name__}")
        if self.cli_args.dataset == "MalignantLunaDataset":
            pos = "mal"
            neg = "ben"
        else:
            pos = "pos"
            neg = "neg"

        neg_label_mask = metrics_t[METRICS_LABEL_NDX] == 0
        neg_pred_mask = metrics_t[METRICS_PRED_NDX] == 0
        pos_label_mask = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask
        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())
        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        pos_correct = int((pos_label_mask & pos_pred_mask).sum())
        tp_count = pos_correct
        fp_count = neg_count - neg_correct
        fn_count = pos_count - pos_correct
        metrics_dict = {
            "loss/all": metrics_t[METRICS_LOSS_CH14_NDX].mean(),
            "loss/neg": metrics_t[METRICS_LOSS_CH14_NDX, neg_label_mask].mean(),
            "loss/pos": metrics_t[METRICS_LOSS_CH14_NDX, pos_label_mask].mean(),
            "correct/all": (
                (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
            ),
            "correct/neg": neg_correct / np.float32(neg_count) * 100,
            "correct/pos": pos_correct / np.float32(pos_count) * 100,
            "pr/precision": tp_count / np.float32(tp_count + fp_count),
            "pr/recall": tp_count / np.float32(tp_count + fn_count),
        }
        metrics_dict["pr/f1_score"] = (
            2
            * metrics_dict["pr/precision"]
            * metrics_dict["pr/recall"]
            / (metrics_dict["pr/precision"] + metrics_dict["pr/recall"])
        )

        threshold = torch.linspace(1, 0, 100)
        tpr = (
            metrics_t[None, METRICS_PRED_P_NDX, pos_label_mask] >= threshold[:, None]
        ).sum(1).float() / pos_count
        fpr = (
            metrics_t[None, METRICS_PRED_P_NDX, neg_label_mask] >= threshold[:, None]
        ).sum(1).float() / neg_count
        fp_diff = fpr[1:] - fpr[:-1]
        tp_avg = (tpr[1:] + tpr[:-1]) / 2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict["auc"] = auc

        log.info(
            f"E{epoch_ndx} {mode_str} {metrics_dict['loss/all']:.4f} loss, {metrics_dict['correct/all']:-5.1f}% correct"
        )
        log.info(
            f"E{epoch_ndx} {mode_str} {metrics_dict['pr/precision']:.4f} precision, {metrics_dict['pr/recall']:.4f} recall, {metrics_dict['pr/f1_score']:.4f} f1 score"
        )
        log.info(
            f"E{epoch_ndx} {mode_str}_neg {metrics_dict['loss/neg']:.4f} loss, {metrics_dict['correct/neg']:-5.1f}% correct ({neg_correct} of {neg_count}"
        )
        log.info(
            f"E{epoch_ndx} {mode_str}_pos {metrics_dict['loss/pos']:.4f} loss, {metrics_dict['correct/pos']:-5.1f}% correct ({pos_correct} of {pos_count}"
        )

        writer = getattr(self, mode_str + "_writer")
        for k, v in metrics_dict.items():
            k = k.replace("pos", pos)
            k = k.replace("neg", neg)
            writer.add_scalar(k, v, self.total_training_samples_count)

        fig = pyplot.figure()
        pyplot.plot(fpr, tpr)
        writer.add_figure("roc", fig, self.total_training_samples_count)
        writer.add_scalar("auc", auc, self.total_training_samples_count)

        bins = np.linspace(0, 1)
        writer.add_histogram(
            f"label_{neg}",
            metrics_t[METRICS_PRED_P_NDX, neg_label_mask],
            self.total_training_samples_count,
            bins=bins,
        )
        writer.add_histogram(
            f"label_{pos}",
            metrics_t[METRICS_PRED_P_NDX, pos_label_mask],
            self.total_training_samples_count,
            bins=bins,
        )

        if not self.cli_args.malignant:
            score = metrics_dict["pr/f1_score"]
        else:
            score = metrics_dict["auc"]

        return score

    def save_model(self, type_str, epoch_ndx, is_best=False):
        file_path = os.path.join(
            "data-unversioned",
            "part2",
            "models",
            self.cli_args.tb_prefix,
            f"{type_str}_{self.time_str}_{self.cli_args.comment}_{self.total_training_samples_count}",
        )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        state = {
            "model_state": model.state_dict(),
            "model_name": type(model).__name__,
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "epoch": epoch_ndx,
            "total_training_samples_count": self.total_training_samples_count,
        }
        torch.save(state, file_path)

        log.debug(f"Saved model params to {file_path}")
        if is_best:
            best_path = os.path.join(
                "data-unversioned",
                "part2",
                "models",
                self.cli_args.tb_prefix,
                f"{type_str}_{self.time_str}_{self.cli_args.comment}.best.state",
            )
            shutil.copyfile(file_path, best_path)
            log.debug(f"Saved model params to {best_path}")
        with open(file_path, "rb") as f:
            log.info(f"SHA!: {hashlib.sha1(f.read()).hexdigest()}")


# class SegmentationTrainingApp:
#     def __init__(self, sys_argv=None):
#         if sys_argv is None:
#             sys_argv = sys.argv[1:]
#         parser = argparse.ArgumentParser()
#         parser.add_argument(
#             "--use-mps",
#             help="Use MPS to run on MacBook GPU. (Note: Conv3D does not work w/ MPS)",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--batch-size",
#             help="Batch size to use for training",
#             default=16,
#             type=int,
#         )
#         parser.add_argument(
#             "--num-workers",
#             help="Number of worker processes for background data loading",
#             default=8,
#             type=int,
#         )
#         parser.add_argument(
#             "--epochs",
#             help="Number of epochs to train for",
#             default=10,
#             type=int,
#         )
#         parser.add_argument(
#             "--augmented",
#             help="Augment the training data.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-flip",
#             help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-offset",
#             help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-scale",
#             help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-rotate",
#             help="Augment the training data by randomly rotating the data around the head-foot axis.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--augment-noise",
#             help="Augment the training data by randomly adding noise to the data.",
#             action="store_true",
#             default=False,
#         )
#         parser.add_argument(
#             "--tb-prefix",
#             default="ch13_redo",
#             help="Data prefix to use for Tensorboard run. Defaults to chapter.",
#         )
#         parser.add_argument(
#             "comment",
#             help="Comment suffix for Tensorboard run.",
#             nargs="?",
#             default="none",
#         )
#         self.cli_args = parser.parse_args(sys_argv)
#         self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
#         self.total_training_samples_count = 0
#         self.trn_writer = None
#         self.val_writer = None
#
#         self.augmentation_dict = {}
#         if self.cli_args.augmented or self.cli_args.augment_flip:
#             self.augmentation_dict["flip"] = True
#         if self.cli_args.augmented or self.cli_args.augment_offset:
#             self.augmentation_dict["offset"] = 0.1
#         if self.cli_args.augmented or self.cli_args.augment_scale:
#             self.augmentation_dict["scale"] = 0.2
#         if self.cli_args.augmented or self.cli_args.augment_rotate:
#             self.augmentation_dict["rotate"] = True
#         if self.cli_args.augmented or self.cli_args.augment_noise:
#             self.augmentation_dict["noise"] = 25.0
#
#         self.use_cuda = torch.cuda.is_available()
#         self.use_mps = self.cli_args.use_mps
#         self.device = torch.device(
#             "cuda" if self.use_cuda else "mps" if self.use_mps else "cpu"
#         )
#         log.info(f"SegmentationTrainingApp created on device {self.device}")
#         self.segmentation_model, self.augmentation_model = self.init_model()
#         self.optimizer = self.init_optimizer()
#
#     def init_model(self):
#         segmentation_model = UNetWrapper(
#             in_channels=7,
#             n_classes=1,
#             depth=3,
#             wf=4,
#             padding=True,
#             batch_norm=True,
#             up_mode="upconv",
#         )
#         augmentation_model = SegmentationAugmentation(**self.augmentation_dict)
#         segmentation_model = segmentation_model.to(self.device)
#         augmentation_model = augmentation_model.to(self.device)
#         return segmentation_model, augmentation_model
#
#     def init_optimizer(self):
#         return Adam(self.segmentation_model.parameters())
#
#     def init_dl(self, val=False):
#         if val:
#             ds = Luna2dSegmentationDataset(
#                 val_stride=10,
#                 is_val_set_bool=val,
#                 context_slices_count=3,
#             )
#         else:
#             ds = TrainingLuna2dSegmentationDataset(
#                 val_stride=10,
#                 is_val_set_bool=val,
#                 context_slices_count=3,
#             )
#         batch_size = self.cli_args.batch_size
#         if self.use_cuda:
#             batch_size *= torch.cuda.device_count()
#         dl = DataLoader(
#             ds,
#             batch_size=batch_size,
#             num_workers=self.cli_args.num_workers,
#             pin_memory=self.use_cuda,
#             # pin_memory=self.use_cuda | self.use_mps,
#         )
#         return dl
#
#     def init_tensorboard_writers(self):
#         if self.trn_writer is None:
#             log_dir = os.path.join("runs", self.cli_args.tb_prefix, self.time_str)
#             self.trn_writer = SummaryWriter(
#                 log_dir=log_dir + "-trn_seg-" + self.cli_args.comment
#             )
#             self.val_writer = SummaryWriter(
#                 log_dir=log_dir + "-val_seg-" + self.cli_args.comment
#             )
#
#     def dice_loss(self, prediction_g, label_g, epsilon=1):
#         dice_label_g = label_g.sum(dim=[1, 2, 3])
#         dice_prediction_g = prediction_g.sum(dim=[1, 2, 3])
#         dice_correct_g = (prediction_g * label_g).sum(dim=[1, 2, 3])
#         dice_ratio_g = (2 * dice_correct_g + epsilon) / (
#             dice_prediction_g + dice_label_g + epsilon
#         )
#         return 1 - dice_ratio_g
#
#     def compute_batch_loss(
#         self, batch_ndx, batch_tup, batch_size, metrics_g, class_threshold=0.5
#     ):
#         input_t, label_t, series_list, _slice_ndx_list = batch_tup
#         input_g = input_t.to(self.device, non_blocking=True)
#         label_g = label_t.to(self.device, non_blocking=True)
#         if self.segmentation_model.training and self.augmentation_dict:
#             input_g, label_g = self.augmentation_model(input_g, label_g)
#             input_g = input_g.to(self.device, non_blocking=True)
#             label_g = label_g.to(self.device, non_blocking=True)
#         prediction_g = self.segmentation_model(input_g)
#         dice_loss_g = self.dice_loss(prediction_g, label_g)
#         fn_loss_g = self.dice_loss(prediction_g * label_g, label_g)
#
#         start_ndx = batch_ndx * batch_size
#         end_ndx = start_ndx + input_t.size(0)
#         with torch.no_grad():
#             prediction_bool_g = (prediction_g[:, 0:1] > class_threshold).to(
#                 torch.float32
#             )
#             tp = (prediction_bool_g * label_g).sum(dim=[1, 2, 3])
#             fn = ((1 - prediction_bool_g) * label_g).sum(dim=[1, 2, 3])
#             fp = (prediction_bool_g * (~label_g)).sum(dim=[1, 2, 3])
#             # fp = (prediction_bool_g * (1 - label_g)).sum(dim=[1, 2, 3])
#             metrics_g[METRICS_LOSS_SEG_NDX, start_ndx:end_ndx] = dice_loss_g
#             metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
#             metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
#             metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp
#         return dice_loss_g.mean() + fn_loss_g.mean() * 8
#
#     def log_images(self, epoch_ndx, mode_str, dl):
#         self.segmentation_model.eval()
#         images = sorted(dl.dataset.series_list)[:12]
#         for series_ndx, seriesuid in enumerate(images):
#             ct = get_ct(seriesuid)
#             for slice_ndx in range(6):
#                 ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5
#                 sample_tup = dl.dataset.getitem_full_slice(seriesuid, ct_ndx)
#                 ct_t, label_t, seriesuid, ct_ndx = sample_tup
#                 input_g = ct_t.to(self.device).unsqueeze(0)
#                 label_g = label_t.to(self.device).unsqueeze(0)
#                 prediction_g = self.segmentation_model(input_g)[0]
#                 prediction_a = prediction_g.to("cpu").detach().numpy()[0] > 0.5
#                 label_a = label_g.cpu().numpy()[0][0] > 0.5
#                 ct_t[:-1, :, :] /= 2000
#                 ct_t[:-1, :, :] += 0.5
#                 ct_slice_a = ct_t[dl.dataset.context_slices_count].numpy()
#                 image_a = np.zeros((512, 512, 3), dtype=np.float32)
#                 image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))
#                 image_a[:, :, 0] += prediction_a & (1 - label_a)
#                 image_a[:, :, 0] += (1 - prediction_a) & label_a
#                 image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5
#                 image_a[:, :, 1] += prediction_a & label_a
#                 image_a *= 0.5
#                 image_a.clip(0, 1, image_a)
#                 writer = getattr(self, f"{mode_str}_writer")
#                 writer.add_image(
#                     f"{mode_str}/{series_ndx}_prediction_{slice_ndx}",
#                     image_a,
#                     self.total_training_samples_count,
#                     dataformats="HWC",
#                 )
#
#                 if epoch_ndx == 1:
#                     image_a = np.zeros((512, 512, 3), dtype=np.float32)
#                     image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))
#                     image_a[:, :, 1] += label_a  # Green
#                     image_a *= 0.5
#                     image_a[image_a < 0] = 0
#                     image_a[image_a > 1] = 1
#                     writer.add_image(
#                         f"{mode_str}/{series_ndx}_label_{slice_ndx}",
#                         image_a,
#                         self.total_training_samples_count,
#                         dataformats="HWC",
#                     )
#                 writer.flush()
#
#     def log_metrics(self, epoch_ndx, mode_str, metrics_t, threshold=0.5):
#         self.init_tensorboard_writers()
#         log.info(f"E{epoch_ndx} {type(self).__name__}")
#         metrics_a = metrics_t.detach().numpy()
#         sum_a = metrics_a.sum(axis=1)
#         assert np.isfinite(metrics_a).all()
#         all_label_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]
#         metrics_dict = {
#             "loss/all": metrics_a[METRICS_LOSS_SEG_NDX].mean(),
#             "percent_all/tp": sum_a[METRICS_TP_NDX] / (all_label_count or 1) * 100,
#             "percent_all/fn": sum_a[METRICS_FN_NDX] / (all_label_count or 1) * 100,
#             "percent_all/fp": sum_a[METRICS_FP_NDX] / (all_label_count or 1) * 100,
#             "pr/precision": sum_a[METRICS_TP_NDX]
#             / (sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX] or 1),
#             "pr/recall": sum_a[METRICS_TP_NDX]
#             / (sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX] or 1),
#         }
#         metrics_dict["pr/f1_score"] = (
#             2
#             * metrics_dict["pr/precision"]
#             * metrics_dict["pr/recall"]
#             / (metrics_dict["pr/precision"] + metrics_dict["pr/recall"] or 1)
#         )
#
#         log.info(
#             f"E{epoch_ndx} {mode_str} "
#             f"{metrics_dict['loss/all']:.4f} loss, "
#             f"{metrics_dict['pr/precision']:.4f} precision, "
#             f"{metrics_dict['pr/recall']:.4f} recall, "
#             f"{metrics_dict['pr/f1_score']:.4f} f1 score, "
#         )
#         log.info(
#             f"E{epoch_ndx} {mode_str}_all "
#             f"{metrics_dict['loss/all']:.4f} loss, "
#             f"{metrics_dict['percent_all/tp']:-5.1f}% tp, "
#             f"{metrics_dict['percent_all/fn']:-5.1f}% fn, "
#             f"{metrics_dict['percent_all/fp']:-5.1f}% fp, "
#         )
#         writer = getattr(self, mode_str + "_writer")
#         prefix_str = "seg_"
#         for k, v in metrics_dict.items():
#             writer.add_scalar(f"{prefix_str}{k}", v, self.total_training_samples_count)
#         writer.flush()
#         score = metrics_dict["pr/recall"]
#         return score
#
#     def save_model(self, type_str, epoch_ndx, is_best=False):
#         file_path = os.path.join(
#             "data-unversioned",
#             "part2",
#             "models",
#             self.cli_args.tb_prefix,
#             f"{type_str}_{self.time_str}_{self.cli_args.comment}_{self.total_training_samples_count}",
#         )
#         model = self.segmentation_model
#         if isinstance(model, torch.nn.DataParallel):
#             model = model.module
#         state = {
#             "sys_argv": sys.argv,
#             "time": str(datetime.datetime.now()),
#             "model_state": model.state_dict(),
#             "model_name": type(model).__name__,
#             "optimizer_state": self.optimizer.state_dict(),
#             "optimizer_name": type(self.optimizer).__name__,
#             "epoch": epoch_ndx,
#             "total_training_samples_count": self.total_training_samples_count,
#         }
#         torch.save(state, file_path)
#         log.info(f"Saved model params to {file_path}")
#         if is_best:
#             best_path = os.path.join(
#                 "data-unversioned",
#                 "part2",
#                 "models",
#                 self.cli_args.tb_prefix,
#                 f"{type_str}_{self.time_str}_{self.cli_args.comment}.best.state",
#             )
#             shutil.copyfile(file_path, best_path)
#             log.info(f"Saved model params to {best_path}")
#         with open(file_path, "rb") as f:
#             log.info(f"SHA!: {hashlib.sha1(f.read()).hexdigest()}")
#
#     def do_training(self, epoch_ndx, train_dl):
#         trn_metrics_g = torch.zeros(
#             METRICS_SIZE_SEG,
#             len(train_dl.dataset),
#             device=self.device,
#         )
#         self.segmentation_model.train()
#         train_dl.dataset.shuffle_samples()
#
#         batch_iter = enumerate_with_estimate(
#             train_dl,
#             f"E{epoch_ndx} training",
#             start_ndx=train_dl.num_workers,
#         )
#         for batch_ndx, batch_tup in batch_iter:
#             self.optimizer.zero_grad()
#             loss_var = self.compute_batch_loss(
#                 batch_ndx,
#                 batch_tup,
#                 train_dl.batch_size,
#                 trn_metrics_g,
#             )
#             loss_var.backward()
#             self.optimizer.step()
#         self.total_training_samples_count += trn_metrics_g.size(1)
#         return trn_metrics_g.to("cpu")
#
#     def do_validation(self, epoch_ndx, val_dl):
#         with torch.no_grad():
#             val_metrics_g = torch.zeros(
#                 METRICS_SIZE_SEG,
#                 len(val_dl.dataset),
#                 device=self.device,
#             )
#             self.segmentation_model.eval()
#             batch_iter = enumerate_with_estimate(
#                 val_dl,
#                 f"E{epoch_ndx} Validation",
#                 start_ndx=val_dl.num_workers,
#             )
#             for batch_ndx, batch_tup in batch_iter:
#                 self.compute_batch_loss(
#                     batch_ndx,
#                     batch_tup,
#                     val_dl.batch_size,
#                     val_metrics_g,
#                 )
#         return val_metrics_g.to("cpu")
#
#     def main(self):
#         log.info(f"Starting {type(self).__name__}, {self.cli_args}")
#         train_dl = self.init_dl()
#         val_dl = self.init_dl(val=True)
#         self.validation_cadence = 5
#         best_score = 0.0
#         for epoch_ndx in range(1, self.cli_args.epochs + 1):
#             log.info(
#                 f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)} trn, {len(val_dl)} "
#                 f"val batches of size {self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}"
#             )
#             trn_metrics_t = self.do_training(epoch_ndx, train_dl)
#             self.log_metrics(epoch_ndx, "trn", trn_metrics_t)
#
#             if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
#                 val_metrics_t = self.do_validation(epoch_ndx, val_dl)
#                 score = self.log_metrics(epoch_ndx, "val", val_metrics_t)
#                 best_score = max(score, best_score)
#                 self.save_model("seg", epoch_ndx, score == best_score)
#                 self.log_images(epoch_ndx, "trn", train_dl)
#                 self.log_images(epoch_ndx, "val", val_dl)


if __name__ == "__main__":
    # LunaTrainingApp().main()
    # SegmentationTrainingApp().main()
    ClassificationTrainingApp().main()
