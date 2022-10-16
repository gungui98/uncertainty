from pathlib import Path

from vital.data.config import Subset

from vital.data.camus.data_module import CamusDataModule

if __name__ == '__main__':
    path = "./camus_point.h5"
    ds = CamusDataModule(Path(path),
                         use_sequence=False,
                         max_patients=None,
                         image_set=Subset.VAL,
                         fold=1,
                         data_augmentation='pixel',
                         batch_size=32,
                         num_workers=1,
                         )
    ds.setup("test")
    samples = []
    for sample in ds.test_dataloader():
        print(sample)

    # samples = np.array(samples)
    #
    # print(samples.min())
    # print(samples.max())
    # print(samples.mean())
    # print(samples.std())

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
