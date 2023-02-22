import argparse
import datetime
import os


import logging
import numpy as np
import sys

from torch.utils.tensorboard import SummaryWriter

from torch import nn
import torch
import torch.cuda
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from part2_redo.dsets import LunaDataset
from part2_redo.model import LunaModel
from part2_redo.my_utils import enumerate_with_estimate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, lr=0.001, momentum=0.99, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num_workers",
            help="Number of workers for background data loading",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use for training",
            default=32,
            type=int,
        )
        parser.add_argument(
            "--epochs",
            help="Number of epochs to train for",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--balanced",
            help="Balance the training data to 50/50 pos/neg.",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "--augmented",
            help="Augment the training data",
            action="store_true",
            default=False,
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

        parser.add_argument(
            "--tb-prefix",
            default="part2_redo",
            help="Data prefix to use for Tensorboard run.",
        )
        parser.add_argument(
            "comment",
            help="Comment suffix for Tensorboard run.",
            nargs="?",
            default="dlwpt",
        )
        self.use_cuda = torch.cuda.is_available()
        # self.use_mps = torch.backends.mps.is_available()
        self.use_mps = False  # Conv3D does not currently work on MPS
        self.device = torch.device(
            "cuda" if self.use_cuda else "mps" if self.use_mps else "cpu"
        )
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
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
            self.augmentation_dict["flip"] = 25.0

    def init_model(self):
        model = LunaModel()
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
        return SGD(self.model.parameters(), lr=0.0001, momentum=0.99)

    def init_dl(self, val=False):
        if val:
            ds = LunaDataset(val_stride=10, is_val_set_bool=val)
        else:
            ds = LunaDataset(
                val_stride=10,
                is_val_set_bool=val,
                ratio_int=int(self.cli_args.balanced),
            )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda | self.use_mps,
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
        trn_metrics_g = torch.zeros(
            METRICS_SIZE, len(train_dl.dataset), device=self.device
        )
        batch_iter = enumerate_with_estimate(
            train_dl, f"E{epoch_ndx} training", start_ndx=train_dl.num_workers
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.compute_batch_loss(
                batch_ndx, batch_tup, train_dl.batch_size, trn_metrics_g
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
                val_dl, f"E{epoch_ndx} Validation", start_ndx=val_dl.num_workers
            )
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g
                )
        return val_metrics_g.to("cpu")

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        train_dl = self.init_dl()
        val_dl = self.init_dl(val=True)
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info(
                f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)} trn, {len(val_dl)} "
                f"val batches of size {self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}"
            )
            trn_metrics_t = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, "trn", trn_metrics_t)
            val_metrics_t = self.do_validation(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, "val", val_metrics_t)
        if hasattr(self, "trn_writer"):
            self.trn_writer.close()
            self.val_writer.close()

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_g = loss_func(logits_g, label_g[:, 1])
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()
        return loss_g.mean()

    def log_metrics(self, epoch_ndx, mode_str, metrics_t, threshold=0.5):
        self.init_tensorboard_writers()
        log.info(f"E{epoch_ndx} {type(self).__name__}")
        neg_label_mask = metrics_t[METRICS_LABEL_NDX] <= threshold
        neg_pred_mask = metrics_t[METRICS_PRED_NDX] <= threshold
        pos_label_mask = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask
        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())
        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        pos_correct = int((pos_label_mask & pos_pred_mask).sum())
        pos_incorrect = neg_count - neg_correct
        neg_incorrect = pos_count - pos_correct
        metrics_dict = {
            "loss/all": metrics_t[METRICS_LOSS_NDX].mean(),
            "loss/neg": metrics_t[METRICS_LOSS_NDX, neg_label_mask].mean(),
            "loss/pos": metrics_t[METRICS_LOSS_NDX, pos_label_mask].mean(),
            "correct/all": (
                (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
            ),
            "correct/neg": neg_correct / np.float32(neg_count) * 100,
            "correct/pos": pos_correct / np.float32(pos_count) * 100,
            "pr/precision": pos_correct / np.float32(pos_correct + pos_incorrect),
            "pr/recall": pos_correct / np.float32(pos_correct + neg_incorrect),
        }
        metrics_dict["pr/f1_score"] = (
            2
            * metrics_dict["pr/precision"]
            / (metrics_dict["pr/precision"] + metrics_dict["pr/recall"])
        )

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
            writer.add_scalar(k, v, self.total_training_samples_count)
        writer.add_pr_curve(
            "pr",
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.total_training_samples_count,
        )

        bins = [x / 50.0 for x in range(51)]
        neg_hist_mask = neg_label_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        pos_hist_mask = pos_label_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)
        if neg_hist_mask.any():
            writer.add_histogram(
                "is_neg",
                metrics_t[METRICS_PRED_NDX, neg_hist_mask],
                self.total_training_samples_count,
                bins=bins,
            )
        if pos_hist_mask.any():
            writer.add_histogram(
                "is_pos",
                metrics_t[METRICS_PRED_NDX, pos_hist_mask],
                self.total_training_samples_count,
                bins=bins,
            )


if __name__ == "__main__":
    LunaTrainingApp().main()
