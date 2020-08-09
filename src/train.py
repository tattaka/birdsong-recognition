import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import yaml
from taggle.engine import (
    CSVLoggerExtension,
    LineNotifyExtension,
    TensorBoardExtension
)
from taggle.losses import get_losses_dict
from taggle.models import ModelProvider
from taggle.models.sync_batchnorm import convert_model
from taggle.optimizers import get_optimizer
from taggle.utils import stratified_kfold_cross_validator
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR
)
from torch.utils.data import DataLoader

from dataset import dataset_zoo
from engines import engine_zoo


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main(config, one_fold=False):
    seed = config["seed"]
    data_path = config["data_path"]
    df_name = config["train_df"]
    data_df = pd.read_csv(os.path.join(data_path, df_name))
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_provider = ModelProvider()
    if os.path.exists(config["logdir"]):
        import datetime
        dt_now = datetime.datetime.now()
        config["logdir"] = config["logdir"] + \
            dt_now.strftime('_%Y%m%d-%H-%M-%S')
    for fold, (train_df, valid_df) in enumerate(stratified_kfold_cross_validator(df=data_df, target=config["target"], n_splits=config["num_fold"], random_state=seed)):
        print(
            f"######################### fold{fold} Start! #########################")
        csv_dir = os.path.join(config["logdir"], 'csvs')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        train_df.to_csv(os.path.join(
            csv_dir, f'train_fold{fold}.csv'), index=False)
        valid_df.to_csv(os.path.join(
            csv_dir, f'valid_fold{fold}.csv'), index=False)
        model = model_provider.get_model(config["model"])
        model = convert_model(model)
        if "pretrain_path" in config and config["pretrain_path"] is not None:
            w = torch.load(config["pretrain_path"] + '/fold' + str(fold)
                           + '/checkpoints/best_f1.pth')["default_model_state_dict"]
            model.load_state_dict(w)
        Dataset = dataset_zoo[config["dataset"]["type"]]
        train_dataset = Dataset(df=train_df,
                                mode="train",
                                **config["dataset"]["params"])
        valid_dataset = Dataset(df=valid_df,
                                mode="valid",
                                **config["dataset"]["params"])
        train_loader = DataLoader(train_dataset,
                                  batch_size=config["batchsize"],
                                  shuffle=True,
                                  num_workers=config["num_workers"],
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=config["batchsize"],
                                  shuffle=False,
                                  num_workers=config["num_workers"],
                                  drop_last=False,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)
        optimizer = get_optimizer(
            model=model, optimizer=config["optimizer"]["type"], **config["optimizer"]["params"])

        criterions = get_losses_dict(config["loss_fn"])

        optimizer = get_optimizer(
            model=model, optimizer=config["optimizer"]["type"], **config["optimizer"]["params"])
        if config["scheduler"]["type"] == "CosineAnnealingLR":
            scheduler = partial(CosineAnnealingLR,
                                **config["scheduler"]["params"])
        elif config["scheduler"]["type"] == "StepLR":
            scheduler = partial(StepLR, **config["scheduler"]["params"])
        elif config["scheduler"]["type"] == "ReduceLROnPlateau":
            scheduler = partial(ReduceLROnPlateau,
                                **config["scheduler"]["params"])
        else:
            scheduler = None
        Engine = engine_zoo[config["engine"]["type"]]
        engine = Engine(
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
            criterions=criterions,
            output_dir=os.path.join(config["logdir"], f"fold{fold}"),
            train_loader=train_loader,
            valid_loader=valid_loader,
            extensions=[CSVLoggerExtension(), TensorBoardExtension(),
                        LineNotifyExtension()],
            **config["engine"]["params"]
        )
        engine.repeated_run(config["num_epoch"])
        print(
            f"######################### fold{fold} End! ###########################")
        if fold == 0 and one_fold:
            print("One fold stop")
            break


if __name__ == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline')
    parser.add_argument('--one_fold', action='store_true')
    args = parser.parse_args()
    config_path = os.path.join("config", args.config + '.yaml')
    print(f"config_path:{config_path}")
    print(f"one_fold:{args.one_fold}")
    with open(config_path, "r+") as f:
        config = yaml.load(f)
    main(config, args.one_fold)
