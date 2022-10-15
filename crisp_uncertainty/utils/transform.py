import itertools
from typing import Dict, Mapping, Tuple, Union, overload
from typing import Sequence

import PIL
import numpy as np
import torch
from PIL import Image
from PIL.Image import LINEAR
from PIL.Image import NEAREST
from keras_preprocessing.image import ImageDataGenerator
from skimage.util import crop
from torch import Tensor

from crisp_uncertainty.utils.utils import to_categorical

Shift = Tuple[int, int]
"""Pixel shift along each axis."""

Rotation = float
"""Angle of the rotation."""

Zoom = Tuple[float, float]
"""Zoom along each axis."""

Crop = Tuple[int, int, int, int, int, int]
"""Original shape and coord. of the bbox, in the following order: height, width, row_min, col_min, row_max, col_max."""

RegisteringParameter = Union[Shift, Rotation, Zoom, Crop]


def to_onehot(y: np.ndarray, num_classes: int = None, channel_axis: int = -1, dtype: str = "uint8") -> np.ndarray:
    """Converts a class vector (integers) to binary class matrix.

    Args:
        y: Class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: Total number of classes.
        channel_axis: Index of the channel axis to expand.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...).

    Returns:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype="int")
    input_shape = list(y.shape)
    if input_shape and input_shape[channel_axis] == 1 and len(input_shape) > 1:
        input_shape.pop(channel_axis)
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape[:]
    categorical_axis = channel_axis if channel_axis != -1 else len(output_shape)
    output_shape.insert(categorical_axis, num_classes)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def resize_image(image: np.ndarray, size: Tuple[int, int], resample: PIL.Image = NEAREST) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: Input image to process. Must be in a format supported by PIL.
        size: Width and height dimensions of the processed image to output.
        resample: Resampling filter to use.

    Returns:
        Input image resized to the specified dimensions.
    """
    resized_image = np.array(Image.fromarray(image).resize(size, resample=resample))
    return resized_image


def remove_labels(segmentation: np.ndarray, labels_to_remove: Sequence[int], fill_label: int = 0) -> np.ndarray:
    """Removes labels from the segmentation map, reassigning the affected pixels to `fill_label`.

    Args:
        segmentation: ([N], H, W, [1|C]), Segmentation map from which to remove labels.
        labels_to_remove: Labels to remove.
        fill_label: Label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ([N], H, W, [1]), Categorical segmentation map with the specified labels removed.
    """
    seg = segmentation.copy()
    if seg.max() == 1 and seg.shape[-1] > 1:  # If the segmentation map is in one-hot format
        for label_to_remove in labels_to_remove:
            seg[..., fill_label] += seg[..., label_to_remove]
        seg = np.delete(seg, labels_to_remove, axis=-1)
    else:  # the segmentation map is categorical
        seg[np.isin(seg, labels_to_remove)] = fill_label
    return seg


def segmentation_to_tensor(segmentation: np.ndarray, flip_channels: bool = False, dtype: str = "int64") -> Tensor:
    """Converts a segmentation map to a tensor, including reordering the dimensions.

    Args:
        segmentation: ([N], H, W, [C]), Segmentation map to convert to a tensor.
        flip_channels: If ``True``, assumes that the input is in `channels_last` mode and will automatically convert it
            to `channels_first` mode. If ``False``, leaves the ordering of dimensions untouched.
        dtype: Data type expected for the converted tensor, as a string
            (`float32`, `float64`, `int32`...).

    Returns:
        ([N], [C], H, W), Segmentation map converted to a tensor.

    Raises:
        ValueError: When reordering from `channels_last` to `channel_first`, the segmentation provided is neither 2D nor
            3D (only shapes supported when reordering channels).
    """
    if flip_channels:  # If there is a specific channel dimension
        if len(segmentation.shape) == 3:  # If it is a single segmentation
            dim_to_transpose = (2, 0, 1)
        elif len(segmentation.shape) == 4:  # If there is a batch dimension to keep first
            dim_to_transpose = (0, 3, 1, 2)
        else:
            raise ValueError(
                "Segmentation to convert to tensor is expected to be a single segmentation (2D), "
                "or a batch of segmentations (3D): \n"
                f"The segmentation to convert is {len(segmentation.shape)}D."
            )
        # Change format from `channel_last`, i.e. ([N], H, W, C), to `channel_first`, i.e. ([N], C, H, W)
        segmentation = segmentation.transpose(dim_to_transpose)
    return torch.from_numpy(segmentation.astype(dtype))


def centered_pad(image: np.ndarray, pad_size: Tuple[int, int], pad_val: float = 0) -> np.ndarray:
    """Pads the image, or batch of images, so that (H, W) match the requested `pad_size`.

    Args:
        image: ([N], H, W, C), Data to be padded.
        pad_size: (H, W) of the image after padding.
        pad_val: Value used for padding.

    Returns:
        ([N], H, W, C), Image, or batch of images, padded so that (H, W) match `pad_size`.
    """
    im_size = np.array(pad_size)

    if image.ndim == 4:
        to_pad = (im_size - image.shape[1:3]) // 2
        to_pad = ((0, 0), (to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))
    else:
        to_pad = (im_size - image.shape[:2]) // 2
        to_pad = ((to_pad[0], to_pad[0]), (to_pad[1], to_pad[1]), (0, 0))

    return np.pad(image, to_pad, mode="constant", constant_values=pad_val)


def centered_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """Crops the image, or batch of images, so that (H, W) match the requested `crop_size`.

    Args:
        image: ([N], H, W, C), Data to be cropped.
        crop_size: (H, W) of the image after the crop.

    Returns:
         ([N], H, W, C), Image, or batch of images, cropped so that (H, W) match `crop_size`.
    """
    if image.ndim == 4:
        to_crop = (np.array(image.shape[1:3]) - crop_size) // 2
        to_crop = ((0, 0), (to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    else:
        to_crop = (np.array(image.shape[:2]) - crop_size) // 2
        to_crop = ((to_crop[0], to_crop[0]), (to_crop[1], to_crop[1]), (0, 0))
    return crop(image, to_crop)


def centered_resize(image: np.ndarray, size: Tuple[int, int], pad_val: float = 0) -> np.ndarray:
    """Centers image around the requested `size`, either cropping or padding to match the target size.

    Args:
        image:  ([N], H, W, C), Data to be adapted to fit the target (H, W).
        size: Target (H, W) for the input image.
        pad_val: The value used for the padding.

    Returns:
        ([N], H, W, C), Image, or batch of images, adapted so that (H, W) match `size`.
    """
    if image.ndim == 4:
        height, width = image.shape[1:3]
    else:
        height, width = image.shape[:2]

    # Check the height to select if we crop of pad
    if size[0] - height < 0:
        image = centered_crop(image, (size[0], width))
    elif size[0] - height > 0:
        image = centered_pad(image, (size[0], width), pad_val)

    # Check if we crop or pad along the width dim of the image
    if size[1] - width < 0:
        image = centered_crop(image, size)
    elif size[1] - width > 0:
        image = centered_pad(image, size, pad_val)

    return image


class AffineRegisteringTransformer:
    """Register image/segmentation pairs based on the structures in the segmentations."""

    registering_steps = ["shift", "rotation", "zoom", "crop"]

    def __init__(self, num_classes: int, crop_shape: Tuple[int, int] = None):
        """Initializes class instance.

        Args:
            num_classes: Number of classes in the dataset from which the image/segmentation pairs come from.
            crop_shape: (height, width) Shape at which to resize the bbox around the ROI after crop.
                This argument is only used if the `crop` registering step applies.
        """
        self.num_classes = num_classes
        self.crop_shape = crop_shape
        self.transformer = ImageDataGenerator(fill_mode="constant", cval=0)
        self.registering_step_fcts = {
            "shift": self._center,
            "rotation": self._rotate,
            "zoom": self._zoom_to_fit,
            "crop": self._crop_resize,
        }

    @staticmethod
    def _get_default_parameters(segmentation: np.ndarray) -> Dict[str, RegisteringParameter]:
        return {
            "shift": (0, 0),
            "rotation": 0,
            "zoom": (1, 1),
            "crop:": segmentation.shape + (0, 0, segmentation.shape[0] - 1, segmentation.shape[1] - 1),
        }

    @overload
    def register_batch(
            self, segmentations: np.ndarray, images: None
    ) -> Tuple[Mapping[str, RegisteringParameter], np.ndarray]:  # noqa: D102
        pass

    def register_batch(
            self, segmentations: np.ndarray, images: np.ndarray = None
    ) -> Tuple[Mapping[str, RegisteringParameter], np.ndarray, np.ndarray]:
        """Registers the segmentations (and images) based on the positioning of the structures in the segmentations.

        Args:
            segmentations: Segmentations to register based on the positioning of their structures.
            images: Images to register based on the positioning of the structures of their associated segmentation.

        Returns:
            - Parameters of the transformations applied to register the segmentations and images.
            - Registered segmentations.
            - Registered images (if `images` is not None).

        Raises:
            ValueError: If the provided images do not match the shape of the segmentations.
        """
        registering_parameters = {step: [] for step in self.registering_steps}
        registered_segmentations, registered_images = [], []
        if images is None:
            images = []
        elif images.shape[:3] != segmentations.shape[:3]:
            # If `images` are provided, ensure they match `segmentations` in every dimension except number of channels
            raise ValueError(
                "Provided `images` parameter does not match first 3 dimensions of `segmentations`. \n"
                f"`images` has shape {images.shape}, \n"
                f"`segmentations` has shape {segmentations.shape}."
            )

        for idx, (segmentation, image) in enumerate(itertools.zip_longest(segmentations, images)):
            if image is not None:
                segmentation_registering_parameters, registered_segmentation, registered_image = self.register(
                    segmentation, image
                )
                registered_images.append(registered_image)
            else:
                segmentation_registering_parameters, registered_segmentation = self.register(segmentation)

            registered_segmentations.append(registered_segmentation)

            # Memorize the parameters used to register the current segmentation
            for registering_parameter, values in registering_parameters.items():
                values.append(segmentation_registering_parameters[registering_parameter])

        out = registering_parameters, np.array(registered_segmentations)
        if registered_images:
            out += (np.array(registered_images),)

        return out

    @overload
    def register(
            self, segmentation: np.ndarray, image: None
    ) -> Tuple[Mapping[str, RegisteringParameter], np.ndarray]:  # noqa: D102
        pass

    def register(
            self, segmentation: np.ndarray, image: np.ndarray = None
    ) -> Tuple[Mapping[str, RegisteringParameter], np.ndarray, np.ndarray]:
        """Registers the segmentation (and image) based on the positioning of the structures in the segmentation.

        Args:
            segmentation: Segmentation to register based on the positioning of its structures.
            image: Image to register based on the positioning of the structures of its associated segmentation.

        Returns:
            - Parameters of the transformation applied to register the image and segmentation.
            - Registered segmentation.
            - Registered image (if `image` is not None).
        """
        # Ensure that the input is in a supported format
        segmentation, original_segmentation_format = self._check_segmentation_format(segmentation)
        if image is not None:
            image, original_image_format = self._check_image_format(image)

        # Register the image/segmentation pair step-by-step
        registering_parameters = {}
        for registering_step in self.registering_steps:
            registering_step_fct = self.registering_step_fcts[registering_step]
            if image is None:
                registering_step_parameters, segmentation = registering_step_fct(segmentation)
            else:
                registering_step_parameters, segmentation, image = registering_step_fct(segmentation, image)
            registering_parameters[registering_step] = registering_step_parameters

        # Restore the image/segmentation to their original formats
        segmentation = self._restore_segmentation_format(segmentation, original_segmentation_format)
        out = registering_parameters, segmentation
        if image is not None:
            image = self._restore_image_format(image, original_image_format)
            out += (image,)

        return out

    def undo_batch_registering(
            self, segmentations: np.ndarray, registering_parameters: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Undoes the registering on the segmentations using the parameters saved during the registration.

        Args:
            segmentations: Registered segmentations to restore to their original value.
            registering_parameters: Parameter names (should appear in `self.registering_steps`) and their values applied
                to register each segmentation.

        Returns:
            Categorical un-registered segmentations.

        Raises:
            ValueError:
                - If any of the keys in ``registering_parameters`` isn't a registering step ``self`` supports.
                - If the provided images do not match the shape of the segmentations.
        """
        # Check that provided parameters correspond to supported registering operations
        for parameter in registering_parameters.keys():
            if parameter not in self.registering_steps:
                raise ValueError(
                    f"Provided `{parameter}` parameter does not match any of the following supported registering "
                    f"steps: \n"
                    f"{self.registering_steps}"
                )

        # Check that parameters for supported registering operations match the provided data
        for registering_step, parameter_values in registering_parameters.items():
            if len(registering_parameters[registering_step]) != len(segmentations):
                raise ValueError(
                    f"Provided `{registering_step}` parameter does not match the number of elements in "
                    f"segmentations. \n"
                    f"`{registering_step}` has length {len(parameter_values)}, "
                    f"`segmentations` has length {len(segmentations)}."
                )

        if "crop" in self.registering_steps:
            # Create an array of the original size in case crop was applied to receive the unregistered output
            unregistered_segmentations = np.empty((len(segmentations), *registering_parameters["crop"][0][:2]))
        else:
            # Create an empty array similar to the segmentations to receive the unregistered output
            unregistered_segmentations = np.empty_like(segmentations)

        for idx, segmentation in enumerate(segmentations):
            seg_registering_parameters = {
                registering_step: values[idx] for registering_step, values in registering_parameters.items()
            }
            unregistered_segmentations[idx] = self.undo_registering(segmentation, seg_registering_parameters)
        return unregistered_segmentations

    def undo_registering(
            self, segmentation: np.ndarray, registering_parameters: Mapping[str, RegisteringParameter]
    ) -> np.ndarray:
        """Undoes the registering on the segmentation using the parameters saved during the registration.

        Args:
            segmentation: Registered segmentation to restore to its original value.
            registering_parameters: Parameter names (should appear in `self.registering_steps`) and their values applied
                to register the segmentation.

        Returns:
            Categorical un-registered segmentation.
        """
        # Get default registering parameters for steps that were not provided
        registering_parameters = dict(registering_parameters)
        registering_parameters.update(
            {
                registering_step: self._get_default_parameters(segmentation)[registering_step]
                for registering_step in AffineRegisteringTransformer.registering_steps
                if registering_step not in registering_parameters.keys()
            }
        )

        # Ensure that the segmentation is in categorical format and of integer type
        segmentation, original_segmentation_format = self._check_segmentation_format(segmentation)

        # Start by restoring the segmentation to its original dimension
        if "crop" in self.registering_steps:
            segmentation = self._restore_crop(segmentation, registering_parameters["crop"])

        # Format the transformations' parameters
        shift = registering_parameters["shift"]
        rotation = registering_parameters["rotation"]
        zoom = registering_parameters["zoom"]
        transformation_parameters = {
            "shift": {"tx": -shift[0], "ty": -shift[1]},
            "rotation": {"theta": -rotation},
            "zoom": {"zx": 1 / zoom[0], "zy": 1 / zoom[1]},
        }

        # Apply each inverse transformation step corresponding to an original transformation,
        # and in the reverse order they were first applied (except for crop that's already been undone)
        registering_steps_wo_crop = [
            registering_step for registering_step in self.registering_steps if registering_step != "crop"
        ]
        for registering_step in reversed(registering_steps_wo_crop):
            segmentation = self._transform_segmentation(segmentation, transformation_parameters[registering_step])

        # Restore the segmentation to its original format
        return self._restore_segmentation_format(segmentation, original_segmentation_format)

    def _check_segmentation_format(self, segmentation: np.ndarray) -> Tuple[np.ndarray, Tuple[bool, bool]]:
        """Ensures that segmentation is in categorical format and of integer type.

        Args:
            segmentation: Segmentation of unknown shape and type.

        Returns:
            - Segmentation in categorical format and of integer type.
            - Flags indicating the original shape of the segmentation.
        """
        # Check if image is a categorical 2D array
        is_categorical_2d = segmentation.ndim == 2

        # Check if image is a categorical 3D array (with last dim of size 1)
        is_categorical_3d = not is_categorical_2d and segmentation.shape[2] == 1

        if is_categorical_2d or is_categorical_3d:  # If the image is not already in categorical format
            segmentation = to_onehot(segmentation, num_classes=self.num_classes)

        return segmentation.astype(np.uint8), (is_categorical_2d, is_categorical_3d)

    @staticmethod
    def _check_image_format(image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Ensures that image has a channels dimension.

        Args:
            image: Image of unknown shape and type.

        Returns:
            - Image with channels dimension.
            - Flag indicating the original shape of the segmentation.
        """
        # Check if image is a categorical 2D array
        is_2d = image.ndim == 2

        if is_2d:  # If the image has not already a channels dimension
            image = image[..., np.newaxis]

        return image, is_2d

    @staticmethod
    def _restore_segmentation_format(segmentation: np.ndarray, format: Tuple[bool, bool]) -> np.ndarray:
        """Restore a segmentation in categorical format to its original shape.

        Args:
            segmentation: Segmentation in categorical format and of integer type.
            format: Flags indicating the original shape of the segmentation.

        Returns:
            Segmentation in its original format.
        """
        is_categorical_2d, is_categorical_3d = format  # Unpack original shape info

        if is_categorical_2d or is_categorical_3d:  # If the segmentation was originally categorical
            segmentation = to_categorical(segmentation)
            if is_categorical_3d:  # If the segmentation had an empty dim of size 1
                segmentation = segmentation[..., np.newaxis]
        return segmentation

    @staticmethod
    def _restore_image_format(image: np.ndarray, is_2d: bool) -> np.ndarray:
        """Restore an image with channels dimension to its original shape.

        Args:
            image: Image with channels dimension.
            is_2d: Flag indicating the original shape of the segmentation.

        Returns:
            Image in its original format.
        """
        is_2d = is_2d  # Unpack original shape info
        if is_2d:  # If the segmentation was originally categorical
            image = np.squeeze(image)
        return image

    def _compute_shift_parameters(self, segmentation: np.ndarray) -> Shift:
        """Computes the pixel shift to apply along each axis to center the segmentation.

        Args:
            segmentation: Segmentation for which to compute shift parameters.

        Returns:
             Pixel shift to apply along each axis to center the segmentation.
        """
        return self._get_default_parameters(segmentation)["shift"]

    def _compute_rotation_parameters(self, segmentation: np.ndarray) -> Rotation:
        """Computes the angle of the rotation to apply to align the segmentation along the desired axis.

        Args:
            segmentation: Segmentation for which to compute rotation parameters.

        Returns:
            Angle of the rotation to apply to align the segmentation along the desired axis.
        """
        return self._get_default_parameters(segmentation)["rotation"]

    def _compute_zoom_to_fit_parameters(self, segmentation: np.ndarray, margin: float = 0.1) -> Zoom:
        """Computes the zoom to apply along each axis to fit the bounding box surrounding the segmented classes.

        Args:
            segmentation: Segmentation for which to compute zoom to fit parameters.
            margin: Ratio of image shape to ignore when computing zoom so as to leave empty border around the image when
                fitting.

        Returns:
            Zoom to apply along each axis to fit the bounding box surrounding the segmented classes.
        """
        return self._get_default_parameters(segmentation)["zoom"]

    def _compute_crop_parameters(self, segmentation: np.ndarray, margin: float = 0.05) -> Crop:
        """Computes the coordinates of a bounding box (bbox) around a region of interest (ROI).

        Args:
            segmentation: Segmentation for which to compute crop parameters.
            margin: Ratio by which to enlarge the bbox from the closest possible fit, so as to leave a slight margin at
                the edges of the bbox.

        Returns:
            Original shape and coordinates of the bbox, in the following order:
            height, width, row_min, col_min, row_max, col_max.
        """
        return self._get_default_parameters(segmentation)["crop"]

    @overload
    def _center(self, segmentation: np.ndarray, image: None) -> Tuple[Shift, np.ndarray]:
        pass

    def _center(self, segmentation: np.ndarray, image: np.ndarray = None) -> Tuple[Shift, np.ndarray, np.ndarray]:
        """Applies a pixel shift along each axis to center the segmentation (and image).

        Args:
            segmentation: Segmentation to center based on the positioning of its structures.
            image: Image to center based on the positioning of the structures of its associated segmentation.

        Returns:
            - Pixel shift applied along each axis to center the segmentation.
            - Centered segmentation.
            - Centered image (if `image` is not None).
        """
        pixel_shift_by_axis = self._compute_shift_parameters(segmentation)
        shift_parameters = {"tx": pixel_shift_by_axis[0], "ty": pixel_shift_by_axis[1]}
        out = pixel_shift_by_axis, self._transform_segmentation(segmentation, shift_parameters)
        if image is not None:
            out += (self._transform_image(image, shift_parameters),)
        return out

    @overload
    def _rotate(self, segmentation: np.ndarray, image: None) -> Tuple[Rotation, np.ndarray]:
        pass

    def _rotate(self, segmentation: np.ndarray, image: np.ndarray = None) -> Tuple[Rotation, np.ndarray, np.ndarray]:
        """Applies a rotation to align the segmentation (and image) along the desired axis.

        Args:
            segmentation: Segmentation to rotate based on the positioning of its structures.
            image: Image to rotate based on the positioning of the structures of its associated segmentation.

        Returns:
            - Angle of the rotation applied to align the segmentation along the desired axis.
            - Rotated segmentation.
            - Rotated image (if `image` is not None).
        """
        rotation_angle = self._compute_rotation_parameters(segmentation)
        rotation_parameters = {"theta": rotation_angle}
        out = rotation_angle, self._transform_segmentation(segmentation, rotation_parameters)
        if image is not None:
            out += (self._transform_image(image, rotation_parameters),)
        return out

    @overload
    def _zoom_to_fit(self, segmentation: np.ndarray, image: None) -> Tuple[Zoom, np.ndarray]:
        pass

    def _zoom_to_fit(self, segmentation: np.ndarray, image: np.ndarray = None) -> Tuple[Zoom, np.ndarray, np.ndarray]:
        """Applies a zoom along each axis to fit the segmentation (and image) to the area of interest.

        Args:
            segmentation: Segmentation to zoom to fit based on the positioning of its structures.
            image: Image to zoom to fit based on the positioning of the structures of its associated segmentation.

        Returns:
            - Zoom applied along each axis to fit the segmentation.
            - Fitted segmentation.
            - Fitted image (if `image` is not None).
        """
        zoom_to_fit = self._compute_zoom_to_fit_parameters(segmentation)
        zoom_to_fit_parameters = {"zx": zoom_to_fit[0], "zy": zoom_to_fit[1]}
        out = zoom_to_fit, self._transform_segmentation(segmentation, zoom_to_fit_parameters)
        if image is not None:
            out += (self._transform_image(image, zoom_to_fit_parameters),)
        return out

    @overload
    def _crop_resize(self, segmentation: np.ndarray, image: None) -> Tuple[Crop, np.ndarray]:
        pass

    def _crop_resize(self, segmentation: np.ndarray, image: np.ndarray = None) -> Tuple[Crop, np.ndarray, np.ndarray]:
        """Applies a zoom along each axis to fit the segmentation (and image) to the area of interest.

        Args:
            segmentation: Segmentation to crop based on the positioning of its structures.
            image: Image to crop based on the positioning of the structures of its associated segmentation.

        Returns:
            - Original shape (2) and crop coordinates (4) applied to get a bbox around the segmentation.
            - Cropped and resized segmentation.
            - Cropped and resized image (if `image` is not None).
        """

        def _crop(image: np.ndarray, bbox: tuple) -> np.ndarray:
            row_min, col_min, row_max, col_max = bbox

            # Pad the image if it is necessary to fit the bbox
            row_pad = max(0, 0 - row_min), max(0, row_max - image.shape[0])
            col_pad = max(0, 0 - col_min), max(0, col_max - image.shape[1])
            image = np.pad(image, (row_pad, col_pad), mode="constant", constant_values=0)

            # Adjust bbox coordinates to new padded image
            row_min += row_pad[0]
            row_max += row_pad[0]
            col_min += col_pad[0]
            col_max += col_pad[0]

            return image[row_min:row_max, col_min:col_max]

        # Compute cropping parameters
        crop_parameters = self._compute_crop_parameters(segmentation)

        # Crop the segmentation around the bbox and resize to target shape
        segmentation = _crop(to_categorical(segmentation), crop_parameters[2:])
        segmentation = to_onehot(resize_image(segmentation, self.crop_shape[::-1]))

        out = crop_parameters, segmentation

        if image is not None:
            # Crop the image around the bbox and resize to target shape
            image = _crop(np.squeeze(image), crop_parameters[2:])
            image = resize_image(image, self.crop_shape[::-1], resample=LINEAR)[..., np.newaxis]
            out += (image,)

        return out

    def _restore_crop(self, segmentation: np.ndarray, crop_parameters: Crop) -> np.ndarray:
        """Restores a cropped region of an segmentation to its original size and location.

        Args:
            segmentation: Cropped region of the original segmentation, to replace in its original position in the
                segmentation.
            crop_parameters: Original shape (2) and crop coordinates (4) applied to get a bbox around the segmentation.

        Returns:
            Segmentation where the cropped region was resized and placed in its original position.
        """
        # Extract shape before crop and crop coordinates from crop parameters
        og_shape = np.hstack((crop_parameters[:2], segmentation.shape[-1]))
        row_min, col_min, row_max, col_max = crop_parameters[2:]

        # Resize the resized cropped segmentation to the original shape of the bbox
        bbox_shape = (row_max - row_min, col_max - col_min)
        segmentation = to_onehot(
            resize_image(to_categorical(segmentation), bbox_shape[::-1]), num_classes=segmentation.shape[-1]
        )

        # Place the cropped segmentation at its original location, inside an empty segmentation
        og_segmentation = np.zeros(og_shape, dtype=np.uint8)
        row_pad = max(0, 0 - row_min), max(0, row_max - og_shape[0])
        col_pad = max(0, 0 - col_min), max(0, col_max - og_shape[1])
        og_segmentation[
        max(0, row_min): min(row_max, og_shape[0]), max(0, col_min): min(col_max, og_shape[1]), :
        ] = segmentation[row_pad[0]: bbox_shape[0] - row_pad[1], col_pad[0]: bbox_shape[1] - col_pad[1], :]

        return og_segmentation

    def _transform_image(
            self, image: np.ndarray, transform_parameters: Mapping[str, RegisteringParameter]
    ) -> np.ndarray:
        """Applies transformations on an image.

        Args:
            image: Image to transform.
            transform_parameters: Parameters describing the transformation to apply.
                Must follow the format required by Keras' ImageDataGenerator (see `ImageDataGenerator.apply_transform`).

        Returns:
            Transformed image.
        """
        return self.transformer.apply_transform(image, transform_parameters)

    def _transform_segmentation(
            self, segmentation: np.ndarray, transform_parameters: Mapping[str, RegisteringParameter]
    ) -> np.ndarray:
        """Applies transformations on a segmentation.

        Args:
            segmentation: Segmentation to transform.
            transform_parameters: Parameters describing the transformation to apply.
                Must follow the format required by Keras' ImageDataGenerator (see `ImageDataGenerator.apply_transform`).

        Returns:
            Transformed segmentation.
        """
        segmentation = self.transformer.apply_transform(segmentation, transform_parameters)

        # Compute the background class as the complement of the other classes
        # (this fixes a bug where some pixels had no class)
        background = ~segmentation.any(2)
        segmentation[background, 0] = 1
        return segmentation
