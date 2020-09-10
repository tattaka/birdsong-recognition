import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from taggle.models.sync_batchnorm import convert_model
from torch import nn
from tqdm import tqdm

from dataset import dataset_zoo
from models import BirdSongModelProvider


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main(configs, out_name):
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
    recon_df["pseudo_label_path"] = ""
    for di in range(len(recon_df)):
        sample = recon_df.loc[di, :]
        ebird_code = sample["ebird_code"]
        file_prefix = sample["resampled_filename"].split(".")[0]
        o = np.zeros_like(outputs[0][di])
        for ci in range(len(outputs)):
            o = o + outputs[ci][di]
        o = o / len(outputs)
        np.save(os.path.join(c_dict["dataset"]["params"]
                             ["data_path"], ebird_code, f'{file_prefix}_probs'), o)
        recon_df.loc[di, "pseudo_label_path"] = os.path.abspath(os.path.join(c_dict["dataset"]["params"]
                                                                             ["data_path"], ebird_code, f'{file_prefix}_probs.npy'))
    recon_df.to_csv(f"../input/birdsong-recognition/{out_name}.csv")


if __name__ == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default='baseline')
    parser.add_argument('--out_name', default='train_pl')
    args = parser.parse_args()
    configs = {}
    for c in args.configs:
        config_path = os.path.join("config", c + '.yaml')
        print(f"config_path:{config_path}")
        with open(config_path, "r+") as f:
            configs.update({c: yaml.load(f)})
    main(configs, args.out_name)
