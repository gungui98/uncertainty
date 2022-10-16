import random
from pathlib import Path

import numpy as np

from vital.data.camus.dataset import Camus
from argparse import ArgumentParser

from vital.data.config import Subset
from matplotlib import pyplot as plt
from vital.data.camus.config import View, CamusTags

if __name__ == '__main__':
    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--predict", action="store_true")
    params = args.parse_args()

    ds = Camus(Path(params.path), image_set=Subset.VAL, predict=params.predict, fold=5, data_augmentation='pixel')

    samples = []
    for sample in ds:
        samples.append(sample[CamusTags.img].squeeze().numpy())

    samples = np.array(samples)

    print(samples.min())
    print(samples.max())
    print(samples.mean())
    print(samples.std())

    # if params.predict:
    #     patient = ds[random.randint(0, len(ds) - 1)]
    #     instant = patient.views[View.A4C]
    #     img = instant.img_proc
    #     gt = instant.gt_proc
    #     print("Image shape: {}".format(img.shape))
    #     print("GT shape: {}".format(gt.shape))
    #     print("ID: {}".format(patient.id))
    #
    #     slice = random.randint(0, len(img) - 1)
    #     img = img[slice].squeeze()
    #     gt = gt[slice]
    # else:
    #     sample = ds[random.randint(0, len(ds) - 1)]
    #     img = sample[CamusTags.img].squeeze()
    #     gt = sample[CamusTags.gt]
    #     print("Image shape: {}".format(img.shape))
    #     print("GT shape: {}".format(gt.shape))
    #
    # print(img.min())
    # print(img.max())
    # print(img.mean())
    # print(img.std())
    #
    # f, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(img)
    # ax2.imshow(gt)
    # plt.show(block=False)
    #
    # plt.figure(2)
    # plt.imshow(img, cmap="gray")
    # plt.imshow(gt, alpha=0.2)
    # plt.show()