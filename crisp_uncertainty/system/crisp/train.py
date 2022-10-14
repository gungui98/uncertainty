import glob
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple, List
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
from scipy.stats import multivariate_normal
from torch import Tensor
from torchmetrics.utilities.data import to_onehot
from torchvision.utils import make_grid
from tqdm import tqdm
from crisp_uncertainty.data.camus.config import CamusTags
from crisp_uncertainty.data.config import Tags
from crisp_uncertainty.evaluation.datasetevaluator import DiceErrorCorrelation, ThresholdedDice, ThresholdErrorOverlap
from crisp_uncertainty.evaluation.segmentation.metrics import SegmentationMetrics
from crisp_uncertainty.evaluation.uncertainty.calibration import PixelCalibration, SampleCalibration, PatientCalibration
from crisp_uncertainty.evaluation.uncertainty.correlation import Correlation
from crisp_uncertainty.evaluation.uncertainty.distribution import Distribution
from crisp_uncertainty.evaluation.uncertainty.mutual_information import UncertaintyErrorMutualInfo
from crisp_uncertainty.evaluation.uncertainty.stats import Stats
from crisp_uncertainty.evaluation.uncertainty.successerrorhistogram import SuccessErrorHist
from crisp_uncertainty.evaluation.uncertainty.vis import UncertaintyVisualization
from crisp_uncertainty.metrics.train.metric import DifferentiableDiceCoefficient

from crisp_uncertainty.evaluation.data_struct import ViewResult, PatientResult
from crisp_uncertainty.evaluation.uncertainty.overlap import UncertaintyErrorOverlap
from crisp_uncertainty.system.crisp.crisp import CRISP
from crisp_uncertainty.data.camus.data_struct import ViewData, PatientData

from crisp_uncertainty.utils.metrics import Dice


class TrainCRISP(CRISP):

    def __init__(self, module: nn.Module,
                 lr,
                 weight_decay,
                 save_samples,
                 output_distribution=None,
                 cross_entropy_weight=0.1,
                 dice_weight=1,
                 clip_weight=1,
                 reconstruction_weight=1,
                 kl_weight=0.5,
                 attr_reg=False,

                 variance_factor=-1,
                 num_samples=150,
                 uncertainty_threshold=0.25,
                 samples_path=None,
                 iterations=1,
                 decode=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_val_step = False
        self.save_hyperparameters()
        self.module = module
        if self.hparams.decode_img:
            self.img_reconstruction_loss = nn.MSELoss()

        if self.hparams.decode_seg:
            self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")

        self.train_set_features = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch: Any, batch_nb: int):
        self.is_val_step = False
        return self.trainval_step(batch, batch_nb)

    def validation_step(self, batch: Any, batch_nb: int):
        self.is_val_step = True
        result = self.trainval_step(batch, batch_nb)
        self.log("val_loss", result["loss"])
        return result

    def trainval_step(self, batch: Any, batch_nb: int):
        img, seg = batch[Tags.img], batch[Tags.gt]
        if self.trainer.datamodule.data_params.out_shape[0] > 1:
            seg_onehot = to_onehot(seg, num_classes=self.trainer.datamodule.data_params.out_shape[0]).float()
        else:
            seg_onehot = seg.unsqueeze(1).float()

        logs = {}
        batch_size = img.shape[0]

        if self.hparams.output_distribution:
            img_mu, img_logvar = self.img_encoder(img)
            seg_mu, seg_logvar = self.seg_encoder(seg_onehot)
        else:
            img_mu = self.img_encoder(img)
            seg_mu = self.seg_encoder(seg_onehot)

        augmentated_seg_mu = seg_mu

        # Compute CLIP loss
        img_logits, seg_logits = self.clip_forward(img_mu, augmentated_seg_mu)

        labels = torch.arange(batch_size, device=self.device)
        loss_i = F.cross_entropy(img_logits, labels, reduction='none')
        loss_t = F.cross_entropy(seg_logits[:batch_size], labels, reduction='none')

        loss_i = loss_i.mean()
        loss_t = loss_t.mean()
        clip_loss = (loss_i + loss_t) / 2

        img_accuracy = img_logits.argmax(0)[:batch_size].eq(labels).float().mean()
        seg_accuracy = seg_logits.argmax(0).eq(labels).float().mean()

        loss = 0
        loss += self.hparams.clip_weight * clip_loss

        if self.hparams.linear_constraint_weight:
            regression_target = batch[self.hparams.linear_constraint_attr]
            regression = self.regression_module(seg_mu)
            regression_mse = F.mse_loss(regression, regression_target)
            loss += self.hparams.linear_constraint_weight * regression_mse
            logs.update({"regression_mse": regression_mse})

        # Compute VAE loss
        if self.hparams.decode_seg:
            if self.hparams.output_distribution:
                seg_z = self.reparameterize(seg_mu, seg_logvar)
                seg_kld = self.latent_space_metrics(seg_mu, seg_logvar)
            else:
                seg_kld = 0
                seg_z = seg_mu

            seg_recon = self.seg_decoder(seg_z)

            seg_metrics = self.seg_reconstruction_metrics(seg_recon, seg)

            seg_vae_loss = self.hparams.reconstruction_weight * seg_metrics['seg_recon_loss'] + \
                           self.hparams.kl_weight * seg_kld

            logs.update({'seg_vae_loss': seg_vae_loss, 'seg_kld': seg_kld})
            logs.update(seg_metrics)

            if self.is_val_step and batch_nb == 0:
                seg_recon = seg_recon.argmax(1) if seg_recon.shape[1] > 1 else torch.sigmoid(seg_recon).round()
                self.log_images(title='Sample (seg)', num_images=5, axes_content={'Image': img.cpu().detach().numpy(),
                                                                                  'GT': seg.cpu().detach().numpy(),
                                                                                  'Pred': seg_recon.cpu().detach().numpy()})

            loss += seg_vae_loss

        if self.hparams.attr_reg:
            attr_metrics = self._compute_latent_space_metrics(seg_mu, batch)
            attr_reg_sum = sum(attr_metrics[f"{attr}_attr_reg"] for attr in
                               CamusTags.list_available_attrs(self.hparams.data_params.labels))
            loss += attr_reg_sum * 10
            logs.update({'attr_reg_loss': attr_reg_sum})

        if self.hparams.decode_img:
            if self.hparams.output_distribution:
                img_z = self.reparameterize(img_mu, img_logvar)
                img_kld = self.latent_space_metrics(img_mu, img_logvar)
            else:
                img_kld = 0
                img_z = img_mu

            img_recon = self.img_decoder(img_z)
            img_metrics = self.img_reconstruction_metrics(img_recon, img)

            img_vae_loss = self.hparams.reconstruction_weight * img_metrics['img_recon_loss'] + \
                           self.hparams.kl_weight * img_kld

            logs.update({'img_vae_loss': img_vae_loss, 'img_kld': img_kld, })
            logs.update(img_metrics)

            if self.is_val_step and batch_nb == 0:
                self.log_images(title='Sample (img)', num_images=5, axes_content={'Image': img.cpu().detach().numpy(),
                                                                                  'Pred': img_recon.cpu().detach().numpy()})

            loss += img_vae_loss

        logs.update({
            'loss': loss,
            'clip_loss': clip_loss,
            'img_accuracy': img_accuracy,
            'seg_accuracy': seg_accuracy,
        })

        return logs

    def log_images(
            self, title: str, num_images: int, axes_content):
        """Log images to Logger if it is a TensorBoardLogger or CometLogger.

        Args:
            title: Name of the figure.
            num_images: Number of images to log.
            axes_content: Mapping of axis name and image.
            info: Additional info to be appended to title for each image.
        """
        for i in range(num_images):
            fig, axes = plt.subplots(1, len(axes_content.keys()), squeeze=False)
            name = f"{title}_{i}"
            plt.suptitle(name)
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):
                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            plt.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.trainer.logger.log_image("{}_{}".format(title, i), [image], step=self.current_epoch)
            plt.close()

    def clip_forward(self, image_features, seg_features):
        image_features = self.img_proj(image_features)
        seg_features = self.seg_proj(seg_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        seg_features = seg_features / seg_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ seg_features.t()
        logits_per_seg = logit_scale * seg_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_seg

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def seg_reconstruction_metrics(self, recon_x: torch.Tensor, x: torch.Tensor):
        # Segmentation accuracy metrics
        if recon_x.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(recon_x.squeeze(), x.type_as(recon_x))
        else:
            ce = F.cross_entropy(recon_x, x)

        dice_values = self._dice(recon_x, x)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.cross_entropy_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))
        return {"seg_recon_loss": loss, "seg_ce": ce, "dice": mean_dice, **dices}

    def img_reconstruction_metrics(self, recon_x: torch.Tensor, x: torch.Tensor):
        return {"img_recon_loss": self.img_reconstruction_loss(recon_x, x)}

    def latent_space_metrics(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

    def _compute_latent_space_metrics(self, mu, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Computes metrics on the input's encoding in the latent space.
        Adds the attribute regularization term to the loss already computed by the parent's implementation.
        Args:
            out: Output of a forward pass with the autoencoder network.
            batch: Content of the batch of data returned by the dataloader.
        References:
            - Computation of the attribute regularization term inspired by the original paper's implementation:
              https://github.com/ashispati/ar-vae/blob/master/utils/trainer.py#L378-L403
        Returns:
            Metrics useful for computing the loss and tracking the system's training progress:
                - metrics computed by ``super()._compute_latent_space_metrics``
                - attribute regularization term for each attribute (under the "{attr}_attr_reg" label format)
        """
        attr_metrics = {}
        for attr_idx, attr in enumerate(CamusTags.list_available_attrs(self.hparams.data_params.labels)):
            # Extract dimension to regularize and target for the current attribute
            latent_code = mu[:, attr_idx].unsqueeze(1)
            attribute = batch[attr]

            # Compute latent distance matrix
            latent_code = latent_code.repeat(1, latent_code.shape[0])
            lc_dist_mat = latent_code - latent_code.transpose(1, 0)

            # Compute attribute distance matrix
            attribute = attribute.repeat(1, attribute.shape[0])
            attribute_dist_mat = attribute - attribute.transpose(1, 0)

            # Compute regularization loss
            # lc_tanh = torch.tanh(lc_dist_mat * self.hparams.delta)
            lc_tanh = torch.tanh(lc_dist_mat * 1)
            attribute_sign = torch.sign(attribute_dist_mat)
            attr_metrics[f"{attr}_attr_reg"] = F.l1_loss(lc_tanh, attribute_sign)

        return attr_metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def encode_set(self, dataloader):
        train_set_features = []
        train_set_segs = []
        for batch in tqdm(iter(dataloader)):
            seg = batch[Tags.gt].to(self.device)
            train_set_segs.append(seg.cpu())
            if self.hparams.data_params.out_shape[0] > 1:
                seg = to_onehot(seg, num_classes=self.hparams.data_params.out_shape[0]).float()
            else:
                seg = seg.unsqueeze(1).float()
            seg_mu = self.seg_encoder(seg)
            train_set_features.append(seg_mu.detach().cpu())
        return train_set_features, train_set_segs

    def on_fit_end(self) -> None:
        # TODO: remove log_dir
        self.hparams.save_samples = "C:/Users/admin/PycharmProjects/CRISP-uncertainty/log/experiment-14-10-22-16-14/sample.pth"
        if not os.path.exists(self.hparams.save_samples):
            print("Generate train features")

            datamodule = self.trainer.datamodule

            train_set_features, train_set_segs = self.encode_set(datamodule.train_dataloader())
            val_set_features, val_set_segs = self.encode_set(datamodule.val_dataloader())
            train_set_features.extend(val_set_features)
            train_set_segs.extend(val_set_segs)

            self.train_set_features = torch.cat(train_set_features)
            self.train_set_segs = torch.cat(train_set_segs)
            Path(self.hparams.save_samples).parent.mkdir(exist_ok=True)
            torch.save({'features': self.train_set_features,
                        'segmentations': self.train_set_segs}, self.hparams.save_samples)
        else:
            print("load features from cache", self.hparams.save_samples)
            features = torch.load(self.hparams.save_samples)
            self.train_set_features = features["features"]
            self.train_set_segs = features["segmentations"]

    def on_test_start(self) -> None:  # noqa: D102
        self.upload_dir = Path(self.trainer.logger.save_dir) / "upload"
        if not self.upload_dir.exists():
            self.upload_dir.mkdir(parents=True, exist_ok=False)

    def on_test_epoch_start(self) -> None:
        print("Generate test features")

        datamodule = self.trainer.datamodule

        self.to(self.device)

        self.on_fit_end()

        print("Latent features", self.train_set_features.shape)
        print("Latent segmentations", self.train_set_segs.shape)

        self.uncertainty_threshold = self.find_threshold(datamodule)
        self.log('best_uncertainty_threshold', self.uncertainty_threshold)

        seed_everything(0, workers=True)

    def test_epoch_end(self, outputs: List[PatientResult]) -> None:
        """Aggregates results.

        Args:
            outputs: List of results for every patient.
        """
        patient_evaluators = [
            SegmentationMetrics(labels=self.hparams.data_params.labels, upload_dir=self.upload_dir),
            Correlation(Dice(labels=self.hparams.data_params.labels), upload_dir=self.upload_dir),
            PixelCalibration(upload_dir=self.upload_dir),
            SampleCalibration(accuracy_fn=Dice(labels=self.hparams.data_params.labels), upload_dir=self.upload_dir),
            UncertaintyErrorOverlap(uncertainty_threshold=self.uncertainty_threshold, upload_dir=self.upload_dir),
            Stats(uncertainty_threshold=self.uncertainty_threshold, upload_dir=self.upload_dir),
            Distribution(upload_dir=self.upload_dir),
            UncertaintyVisualization(
                uncertainty_threshold=self.uncertainty_threshold, nb_figures=50, upload_dir=self.upload_dir
            ),
            SuccessErrorHist(upload_dir=self.upload_dir),
            PatientCalibration(upload_dir=self.upload_dir),
            UncertaintyErrorMutualInfo(upload_dir=self.upload_dir)
        ]

        dataset_evaluators = [
            DiceErrorCorrelation(upload_dir=self.upload_dir),
            ThresholdedDice(upload_dir=self.upload_dir),
            ThresholdErrorOverlap()]

        metrics = {}
        patient_metrics = []
        for evaluator in patient_evaluators:
            print(f"Generating results with {evaluator.__class__.__name__}...")
            try:
                evaluator_metrics, evaluator_patient_metrics = evaluator(outputs)
                patient_metrics.append(evaluator_patient_metrics)
                metrics.update(evaluator_metrics)
            except Exception as e:
                print(f"Failed with exception {e}")

        patient_metrics = pd.concat([pd.DataFrame(m).T for m in patient_metrics], axis=1)

        patient_metrics.to_csv(self.upload_dir / "patient_metrics.csv")

        for evaluator in dataset_evaluators:
            print(f"Generating results with {evaluator.__class__.__name__}...")
            try:
                evaluator_metrics = evaluator(patient_metrics)
                metrics.update(evaluator_metrics)
            except Exception as e:
                print(f"Failed with exception {e}")
        visualize_files = glob.glob(os.path.join(str(self.upload_dir), "*.png"))
        self.trainer.logger.log_image(key="test_result", images=visualize_files, caption=[os.path.basename(i)[:-4] for i in visualize_files] )

        metrics = {f'test_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics)

    def test_step(self, batch: PatientData, batch_idx: int):
        """Performs test-time inference for a patient and saves the results to an HDF5 file."""
        return self.compute_patient_uncertainty(batch)

    def compute_patient_uncertainty(self, batch: PatientData) -> PatientResult:
        """Computes the uncertainty for one patient.

        Args:
            batch: data for one patient including image, groundtruth and other information.

        Returns:
            PatientResult struct containing uncertainty prediction
        """
        patient_res = PatientResult(id=batch.id)
        for view, data in batch.views.items():
            patient_res.views[view] = self.compute_view_uncertainty(view, data)
        return patient_res

    def compute_view_uncertainty(self, view: str, data: ViewData) -> ViewResult:
        # pred = self.module(data.img_proc.to(self.device))
        # pred = F.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)
        logits = [self.module(data.img_proc.to(self.device)) for _ in range(self.hparams.iterations)]
        if logits[0].shape[1] == 1:
            probs = [torch.sigmoid(logits[i]).detach() for i in range(self.hparams.iterations)]
            pred = torch.stack(probs, dim=-1).mean(-1)
        else:
            probs = [F.softmax(logits[i], dim=1).detach() for i in range(self.hparams.iterations)]
            pred = torch.stack(probs, dim=-1).mean(-1)

        frame_uncertainties = []
        uncertainty_maps = []
        for instant in range(pred.shape[0]):
            uncertainty, uncertainty_map = self.predict_uncertainty(data.img_proc[instant], pred[instant])
            frame_uncertainties.append(uncertainty)
            uncertainty_maps.append(uncertainty_map)

        return ViewResult(
            img=data.img_proc.cpu().numpy(),
            gt=data.gt_proc.cpu().numpy(),
            pred=pred.detach().cpu().numpy(),
            uncertainty_map=np.array(uncertainty_maps),
            frame_uncertainties=np.array(frame_uncertainties),
            view_uncertainty=np.mean(frame_uncertainties),
            voxelspacing=data.voxelspacing,
            instants=data.instants,
        )

    def predict_uncertainty(self, img: Tensor, pred: Tensor) -> Tuple[float, np.array]:
        img, pred = img.to(self.device), pred.to(self.device)
        if self.hparams.data_params.out_shape[0] > 1:
            pred = to_onehot(pred.argmax(0, keepdim=True), num_classes=self.hparams.data_params.out_shape[0])
        else:
            pred = pred.round().unsqueeze(1)

        # Get input image features
        img_mu = self.img_encoder(img[None])
        image_features = self.img_proj(img_mu)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Get sample features
        samples = self.train_set_features.to(self.device).float()
        sample_features = self.seg_proj(samples)
        sample_features = sample_features / sample_features.norm(dim=-1, keepdim=True)
        sample_logits = sample_features @ image_features.t()

        # Get prediction features
        if self.hparams.variance_factor != -1:
            pred_mu = self.seg_encoder(pred.float())
            pred_features = self.seg_proj(pred_mu)
            pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
            pred_logits = pred_features @ image_features.t()

            cov = torch.abs(image_features - pred_features)
            cov = torch.sqrt(torch.sum(torch.square(image_features - pred_features))) * self.hparams.variance_factor
        else:
            sigma = torch.std(sample_features, dim=0)
            # cov = 1.06 * sigma * sample_features.shape[0]**(-1/5)
            q1, q3 = torch.quantile(sample_features, torch.tensor([0.25, 0.75]).to(self.device), dim=0, keepdim=True)
            cov = 0.9 * torch.minimum(sigma, (q3 - q1) / 1.34) * sample_features.shape[0] ** (-1 / 5)

        normal = multivariate_normal(mean=image_features.squeeze().cpu(), cov=cov.cpu().squeeze())
        gaussian_weights = normal.pdf(sample_features.squeeze().cpu()) / normal.pdf(image_features.squeeze().cpu())

        x_hat = torch.mean(sample_features, dim=0, keepdim=True)
        R_hat = torch.norm(x_hat)
        mu = x_hat / R_hat
        kappa = R_hat * (8 - R_hat ** 2) / 1 - R_hat ** 2
        h0 = kappa ** (-1 / 2) * (40 * torch.sqrt(torch.tensor(math.pi)) * sample_features.shape[0]) ** (-1 / 5)
        weights = torch.exp(1 / h0 * (image_features @ sample_features.t() - 1)).squeeze().cpu()

        # Get indices of samples higher than the prediction logits
        sorted_indices = torch.argsort(sample_logits, dim=0, descending=True)

        indices = sorted_indices[:self.hparams.num_samples]

        samples = samples[indices].squeeze()
        weights = weights[indices.cpu()]

        decoded = self.train_set_segs[indices].squeeze()
        if self.seg_channels > 1:
            decoded = to_onehot(decoded, num_classes=self.seg_channels)

        uncertainty_map = []
        for i in range(decoded.shape[0]):
            weight = weights[i]
            # print(weight)
            diff = (~torch.eq(pred.squeeze().cpu(), decoded[i].cpu().squeeze())).float()
            uncertainty_map.append(diff[None] * weight)
            # aleatoric_map.append(decoded[i][None] * weight)

        uncertainty_map = torch.cat(uncertainty_map, dim=0)
        uncertainty_map = uncertainty_map.mean(0).squeeze()

        if self.seg_channels > 1:
            labels_values = [label.value for label in self.hparams.data_params.labels if label.value != 0]
            uncertainty_map = uncertainty_map[labels_values, ...]

        if uncertainty_map.ndim > 2:
            uncertainty_map = uncertainty_map.sum(0)

        uncertainty_map = uncertainty_map / uncertainty_map.max()
        # uncertainty_map = uncertainty_map.crisp(max=1)
        uncertainty_map = uncertainty_map.cpu().detach().numpy()

        # Compute frame uncertainty
        mask = pred.cpu().detach().numpy() != 0
        frame_uncertainty = (np.sum(uncertainty_map) / np.sum(mask))

        return frame_uncertainty, uncertainty_map

    def find_threshold(self, datamodule):
        if self.hparams.uncertainty_threshold == -1:
            print("Finding ideal threshold...")
            datamodule.setup('fit')  # Need to access Validation set
            val_dataloader = datamodule.val_dataloader()
            errors, uncertainties, error_sums = [], [], []

            for _, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Predicting on val set'):
                x, y = data[Tags.img], data[Tags.gt]
                pred = self.module(x.to(self.device))
                pred = F.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)
                for instant in range(pred.shape[0]):
                    _, unc = self.predict_uncertainty(x[instant], pred[instant])

                    err = ~np.equal(pred[instant].argmax(0).cpu().numpy(), y[instant].numpy())
                    errors.append(err)
                    uncertainties.append(unc[None])
                    error_sums.append(err.sum())

            errors, uncertainties, error_sums = np.concatenate(errors), np.concatenate(uncertainties), np.array(
                error_sums)
            print(errors.shape)
            print(uncertainties.shape)
            print(error_sums.shape)

            all_dices = []
            thresholds = np.arange(0.025, 1, 0.025)
            for thresh in tqdm(thresholds, desc='Finding ideal threshold'):
                dices = []
                for e, u in zip(errors, uncertainties):
                    dices.append(UncertaintyErrorOverlap.compute_overlap(e, u, thresh))
                all_dices.append(np.average(dices, weights=error_sums))

            return thresholds[np.argmax(all_dices)]
        else:
            return self.hparams.uncertainty_threshold
