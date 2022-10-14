from crisp_uncertainty.data.camus.config import Label
from crisp_uncertainty.metrics.evaluate.anatomical_structure import Anatomical2DStructureMetrics
from crisp_uncertainty.metrics.evaluate.segmentation import Segmentation2DMetrics


class LeftAtriumMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the left atrium."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.ATRIUM.value)
