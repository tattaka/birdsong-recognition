import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
import scipy as sp
import torch
import yaml
from taggle.models.sync_batchnorm import convert_model
from torch import nn
from tqdm import tqdm

from dataset import BIRD_CODE, dataset_zoo
from models import BirdSongModelProvider


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main(configs, target_fold=-1):
    seed = set()
    for c in configs.values():
        seed.add(c["seed"])
    assert(len(seed) == 1)
    seed = list(seed)[0]
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_provider = BirdSongModelProvider()
    outputs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c_key, c_dict in tqdm(zip(configs.keys(), configs.values()), total=len(configs.values())):
        csv_dir = os.path.join(c_dict["logdir"], 'csvs')
        output_conf = []
        recon_df = []
        for fold in tqdm(range(c_dict["num_fold"]), leave=False):
            valid_df = pd.read_csv(os.path.join(
                csv_dir, f'valid_fold{fold}.csv'))
            model = model_provider.get_model(c_dict["model"])
            model = convert_model(model)
            output_dir = os.path.join(c_dict["logdir"], f"fold{fold}")
            w = torch.load(
                output_dir + '/checkpoints/best_f1.pth')["default_model_state_dict"]
            model.load_state_dict(w)
            Dataset = dataset_zoo["ClipwiseDataset"]
            model.to(device)
            model = nn.DataParallel(model)
            model.eval()
            if target_fold > 0 and target_fold != fold:
                continue
            for idx in tqdm(range(len(valid_df)), leave=False):
                valid_dataset = Dataset(df=valid_df,
                                        idx=idx,
                                        img_size=c_dict["dataset"]["params"]["img_size"],
                                        data_path=c_dict["dataset"]["params"]["data_path"])
                images = valid_dataset.__getitem__(0)["image"]
                images = torch.from_numpy(images)
                images = images.to(device)
                with torch.no_grad():
                    prediction = model(images)
                    proba = torch.sigmoid(prediction["output1"]).detach(
                    ).cpu().numpy()
                    # output_clip -> (len(clip), 264)

                output_conf.append(proba)
                # print(f"Finish valid fold {fold}  {idx}/{len(valid_df)}")
            recon_df.append(valid_df)
        # output_conf -> (len(train_df), len(clip), 264)
        outputs.append(output_conf)
    # output_conf -> (len(config), len(train_df), len(clip), 264)
    recon_df = pd.concat(recon_df).reset_index(drop=True)
    pred_clips = []
    for di in range(len(recon_df)):
        o = np.zeros_like(outputs[0][di])
        for ci in range(len(outputs)):
            o = o + outputs[ci][di]
        o = o / len(outputs)
        pred_clips.append(o)
    trues = np.array([BIRD_CODE[i] for i in recon_df.ebird_code.values])
    macro_optimizer = ThresholdOptimizer(mode="macro")
    macro_optimizer.fit(pred_clips, trues)
    macro_coef = macro_optimizer.coefficients()
    macro_score = macro_optimizer.calc_score(pred_clips, trues, macro_coef)
    print(
        f"macro threshold {macro_coef} f1 score: {macro_score}")
    print(
        f"default threshold 0.5 f1 score: {macro_optimizer.calc_score(pred_clips, trues, [0.5])}")
    # micro_optimizer = ThresholdOptimizer(mode="micro")
    # micro_optimizer.fit(pred_clips, trues)
    # micro_coef = micro_optimizer.coefficients()
    # micro_score = micro_optimizer.calc_score(pred_clips, trues, micro_coef)
    # print(
    #     f"micro threshold {micro_coef} f1 score: {micro_score}")


def row_wise_f1_score_micro_numpy(y_true, y_pred, threshold=0.5, count=5):
    """
    @author shonenkov

    y_true - 2d npy vector with gt
    y_pred - 2d npy vector with prediction
    threshold - for round labels
    count - number of preds (used sorting by confidence)
    """

    # def meth_agn_v2(x, threshold):
    #     idx, = np.where(x > threshold[0])
    #     return idx[np.argsort(x[idx])[::-1]]

    def event_thresholder(x, threshold):
        # x -> [clip_num, chunk_num, 264]
        x_out = []
        for ci, xi in enumerate(x):
            all_events = set()
            for xii in xi:
                events = (xii >= threshold)[None, :]
                for i in range(len(events)):
                    event = events[i, :]
                    labels = np.argwhere(event).reshape(-1).tolist()
                    for label in labels:
                        all_events.add(label)
            x_out.append(list(all_events))
        return x_out

    F1 = []
    pred_events = event_thresholder(y_pred, threshold)
    for preds, trues in zip(pred_events, y_true):
        TP, FN, FP = 0, 0, 0
        preds = preds[:count]
        trues = [trues]  # TODO: create list pattern
        for true in trues:
            if true in preds:
                TP += 1
            else:
                FN += 1
        for pred in preds:
            if pred not in trues:
                FP += 1
        F1.append(2 * TP / (2 * TP + FN + FP))
    return np.mean(F1)


class ThresholdOptimizer():
    def __init__(self, mode="micro"):
        self.coef_ = {}
        self.mode = mode
        if mode == "micro":
            self.coef_["x"] = [0.5 for i in range(264)]
        elif mode == "macro":
            self.coef_["x"] = [0.5]

    def _f1_loss(self, coef, X, y):
        ll = row_wise_f1_score_micro_numpy(y, X, coef)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._f1_loss, X=X, y=y)
        if self.mode == "micro":
            initial_coef = [0.5 for i in range(264)]
        elif self.mode == "macro":
            initial_coef = [0.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method='nelder-mead')

    def coefficients(self):
        return self.coef_['x']

    def calc_score(self, X, y, coef):
        return row_wise_f1_score_micro_numpy(y, X, coef)


if __name__ == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default='baseline')
    parser.add_argument('--target_fold', type=int, default=-1)
    args = parser.parse_args()
    configs = {}
    for c in args.configs:
        config_path = os.path.join("config", c + '.yaml')
        print(f"config_path:{config_path}")
        with open(config_path, "r+") as f:
            configs.update({c: yaml.load(f)})
    main(configs, args.target_fold)
