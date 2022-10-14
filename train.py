import datetime
import os
import re
from argparse import Namespace
from pathlib import Path
from shutil import copy2

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from crisp_uncertainty.data.camus.data_module import CamusDataModule

from crisp_uncertainty.modules.enet import Enet
from crisp_uncertainty.system.crisp.train import TrainCRISP

if __name__ == '__main__':

    OmegaConf.register_new_resolver(
        "camus_labels", lambda x: '-' + '-'.join([n for n in x if n != 'BG']) if len(x) != 4 else ''
    )
    OmegaConf.register_new_resolver(
        "frac", lambda x: int(x * 100)
    )
    initialize(config_path=".")
    cfg = compose(config_name="default.yaml")

    cfg.seed = seed_everything(cfg.seed, workers=True)
    experiment_name = "experiment-" + datetime.datetime.now().strftime("%d-%m-%y-%H-%M")
    log_dir = os.path.join("log", experiment_name)

    callbacks = [EarlyStopping(patience=100, monitor="val_loss"), ModelCheckpoint(monitor="val_loss")]
    logger = WandbLogger(name=experiment_name, project="CRISP", offline=False, save_dir=log_dir)

    if cfg.ckpt_path:
        trainer = Trainer(resume_from_checkpoint=cfg.ckpt_path, logger=logger, callbacks=callbacks, gpus=1)
    else:
        trainer = Trainer(max_epochs=cfg.num_epoch, logger=logger, callbacks=callbacks, gpus=1)
        trainer.logger.log_hyperparams(Namespace(**cfg))  # Save config to logger.

    # If logger as a logger directory, use it. Otherwise, default to using `default_root_dir`

    # Instantiate datamodule
    datamodule = CamusDataModule(dataset_path=cfg.data.dataset_path,
                                 batch_size=cfg.data.batch_size,
                                 labels=cfg.data.labels,
                                 fold=cfg.data.fold,
                                 use_sequence=cfg.data.use_sequence,
                                 views=cfg.data.views,
                                 )

    input_shape = datamodule.data_params.in_shape
    output_shape = datamodule.data_params.out_shape

    # Instantiate module with respect to datamodule's data params.
    module = Enet(
        input_shape=input_shape,
        output_shape=output_shape,
    )

    # Instantiate model with the created module.
    model = TrainCRISP(module=module,
                       lr=cfg.lr,
                       weight_decay=cfg.weight_decay,
                       save_samples=os.path.join(log_dir, "sample.pth"),
                       output_distribution=None,
                       cross_entropy_weight=0.1,
                       dice_weight=1,
                       clip_weight=1,
                       reconstruction_weight=1,
                       kl_weight=0.5,
                       attr_reg=False,
                       data_params=datamodule.data_params,
                       **cfg.model,
                       **cfg.test_param,
                       )

    if cfg.ckpt_path is not None:  # Load pretrained model if checkpoint is provided
        print(f"Loading model from {cfg.ckpt_path}")
        model = model.load_from_checkpoint(
            str(cfg.ckpt_path), module=module, data_params=datamodule.data_params, strict=False
        )
    elif cfg.weights is not None:
        weights = cfg.weights
        print(f"Loading weights from {weights}")
        model.load_state_dict(torch.load(weights, map_location=model.device)["state_dict"], strict=cfg.strict)

    if cfg.train:
        trainer.fit(model, datamodule=datamodule)

        # Copy best model checkpoint to a predictable path + online tracker (if used)
        best_model_path = Path(os.path.join(log_dir, "best_model.pth"))
        best_model_path.parent.mkdir(exist_ok=True)

        if trainer.checkpoint_callback is not None:
            best_model = trainer.checkpoint_callback.best_model_path
            copy2(best_model, str(best_model_path))

            # Delete checkpoint after copy to avoid filling disk.
            for file in trainer.checkpoint_callback.best_k_models.keys():
                os.remove(file)

            # Ensure we use the best weights (and not the latest ones) by loading back the best model
            model = model.load_from_checkpoint(str(best_model_path), module=module)
        else:  # If checkpoint callback is not used, save current model.
            trainer.save_checkpoint(best_model_path)

        # trainer.logger.experiment.log_model("model", trainer.checkpoint_callback.best_model_path)

    if cfg.test:
        datamodule.setup("fit")
        trainer.test(model, datamodule=datamodule)
