import random

import numpy as np
import torch
from taggle.engine import BaseEngine
from torch.nn import functional as F

from metrics import f1


class BirdSongEngine(BaseEngine):

    def calc_metrics(self, outputs, data):
        metrics = {}
        _, labels = data["targets"].max(axis=1)
        metrics["f1"] = f1(F.softmax(outputs["output1"], dim=1), labels)
        return metrics

    def forward(self, data):
        outputs = self.models["default"](data["image"])
        return outputs

    def backward(self, losses):
        self.update_model(self.models["default"],
                          losses["loss"], self.optimizers["default"])


class BirdSongBCEEngine(BirdSongEngine):

    def calc_losses(self, outputs, data):
        losses = {}
        losses["loss"] = self.criterions["classification"](
            outputs["output1"], data["targets"])
        return losses


class BirdSongCEEngine(BirdSongEngine):

    def calc_losses(self, outputs, data):
        losses = {}
        _, labels = data["targets"].max(axis=1)
        losses["loss"] = self.criterions["classification"](
            outputs["output1"], labels)
        return losses


class BirdSongOUSMEngine(BirdSongEngine):

    def calc_losses(self, outputs, data):
        losses = {}
        losses["loss"] = self.criterions["classification"](
            outputs["output1"], data["targets"])
        if self.epoch >= self.ousm_trigger:
            k = int(len(data["targets"]) * self.ousm_rate)
            if len(losses["loss"].shape) == 2:
                losses["loss"] = losses["loss"].mean(1)
            _, idxs = losses["loss"].topk(
                len(data["targets"]) - k, largest=False)
            losses["loss"] = losses["loss"].index_select(0, idxs)
        losses["loss"] = losses["loss"].mean()
        return losses


class BirdSongOHEMEngine(BirdSongEngine):

    def calc_losses(self, outputs, data):
        losses = {}
        losses["loss"] = self.criterions["classification"](
            outputs["output1"], data["targets"])
        if self.epoch >= self.ousm_trigger:
            k = int(len(data["targets"]) * self.ohem_rate)
            if len(losses["loss"].shape) == 2:
                losses["loss"] = losses["loss"].mean(1)
            _, idxs = losses["loss"].topk(k, largest=False)
            losses["loss"] = losses["loss"].index_select(0, idxs)
        losses["loss"] = losses["loss"].mean()
        return losses


class BirdSongFlipIdEngine(BirdSongEngine):
    # TODO: flipした時にlogitsを片方だけ使うかどうか考える
    # TODO: 現在は片方だけを伝搬させる実装 by pudae and seotao

    def calc_losses(self, outputs, data):
        losses = {}
        class_num = data["targets"].size(1) // 2
        _, labels = data["targets"].max(axis=1)
        is_flip = labels >= class_num
        # print(is_flip.shape, data["targets"][:, class_num:].shape)
        target = torch.where(
            is_flip[:, None], data["targets"][:, class_num:], data["targets"][:, :class_num])
        logits = torch.where(
            is_flip[:, None], outputs["output1"][:, class_num:], outputs["output1"][:, :class_num])
        losses["loss"] = self.criterions["classification"](
            logits, target)
        return losses

    def calc_metrics(self, outputs, data):
        metrics = {}
        class_num = data["targets"].size(1) // 2
        _, labels = data["targets"].max(axis=1)
        is_flip = labels >= class_num
        labels = torch.where(is_flip, labels - class_num, labels)
        logits = torch.where(
            is_flip[:, None], outputs["output1"][:, class_num:], outputs["output1"][:, :class_num])
        metrics["f1"] = f1(F.softmax(logits, dim=1), labels)
        return metrics


class BirdSongMixUpEngine(BirdSongBCEEngine):
    lam = 1
    index = None

    def calc_losses(self, outputs, data):
        losses = {}

        loss_a = super().calc_losses(outputs, data)["loss"]
        data["target"] = data["target"][self.index]
        loss_b = super().calc_losses(outputs, data)["loss"]
        losses["loss"] = self.lam * loss_a + (1 - self.lam) * loss_b
        return losses

    def forward(self, data):
        alpha = random.random() * self.alpha * 0.2 + self.alpha * 0.8
        if alpha > 0:
            self.lam = np.random.beta(alpha, alpha)
        else:
            self.lam = 1
        self.index = torch.randperm(data["image"].shape[0])
        self.index.cuda()
        data["image"] = self.lam * data["image"] + \
            (1 - self.lam) * data["image"][self.index]
        outputs = self.models["default"](data["image"])
        return outputs


def mixup_wrapper(WrappedClass: BirdSongEngine):
    class BirdSongMixUpEngine(WrappedClass):
        lam = 1
        index = None

        def calc_losses(self, outputs, data):
            losses = {}

            loss_a = super().calc_losses(outputs, data)["loss"]
            data["target"] = data["target"][self.index]
            loss_b = super().calc_losses(outputs, data)["loss"]
            losses["loss"] = self.lam * loss_a + (1 - self.lam) * loss_b
            return losses

        def forward(self, data):
            alpha = random.random() * self.alpha * 0.2 + self.alpha * 0.8
            if alpha > 0:
                self.lam = np.random.beta(alpha, alpha)
            else:
                self.lam = 1
            self.index = torch.randperm(data["image"].shape[0])
            self.index.cuda()
            data["image"] = self.lam * data["image"] + \
                (1 - self.lam) * data["image"][self.index]
            outputs = self.models["default"](data["image"])
            return outputs
    return BirdSongMixUpEngine


engine_zoo = {"BirdSongBCEEngine": BirdSongBCEEngine,
              "BirdSongCEEngine": BirdSongCEEngine,
              "BirdSongOUSMEngine": BirdSongOUSMEngine,
              "BirdSongFlipIdEngine": BirdSongFlipIdEngine,
              "BirdSongOHEMEngine": BirdSongOHEMEngine,
              "BirdSongMixupBCEEngine": mixup_wrapper(BirdSongBCEEngine)}