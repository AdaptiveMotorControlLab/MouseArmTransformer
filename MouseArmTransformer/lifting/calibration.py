import os

import numpy as np
import PIL.Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import dataclasses
import itertools

import glob
import PIL.Image

from MouseArmTransformer.lifting import opencv
from MouseArmTransformer.lifting.camera import StereoCamera
from MouseArmTransformer.lifting.utils import _f32, _f64


class CameraCalibration:
    """Compute view extrinsics and camera intrinsics from calibration points.

    Given object points measured with different cameras, compute intrinsics
    and extrinsics for the different views. To use this function, you need
    a collection of (paired) measurements from different views and cameras.

    Your `image_points` should be an array of shape
    (num_cameras, num_views, num_points, 2), the object points should be the
    corresponding object points of shape `(num_views, num_points, 3)` and are
    typically initialized using a meshgrid.

    This class will compute the intrinsics of each cameras, and additionally one
    set of extrinsics for each view. This means that you will get `num_cameras`
    intrinsic matrices, and `num_cameras x num_views` extrinsic matrices.
    """

    def __init__(self, board):
        self.board = board
        self._cameras = {}
        self._compute_intrinsics_and_views()

    @property
    def image_points(self):
        """The 2D image points used for calculating the calibration result."""
        return self.board.image_points

    @property
    def object_points_3d(self):
        """The 3D object points used for calculating the calibration result."""
        return self.board.object_points_3d

    @property
    def frame_sizes(self):
        """Frame sizes for every camera."""
        return self.board.frame_sizes

    @property
    def num_cameras(self):
        return self.board.num_cameras

    def _compute_intrinsics_and_views(self):
        object_points = self.object_points_3d[np.newaxis].repeat(9, axis=0)
        for camera_id, img_points_2d in enumerate(self.image_points):
            imgpoints = list(_f32(img_points_2d[..., np.newaxis, :]))
            objpoints = list(_f32(object_points))
            size = tuple(self.frame_sizes[camera_id])
            self._cameras[
                camera_id
            ] = opencv.CameraCalibrationResult.from_calibration_points(
                objpoints, imgpoints, size
            )

    def stereo_calibration(self, camera_1=0, camera_2=1) -> StereoCamera:
        """Return a stereo calibration computed from two cameras."""
        calibration_results = opencv.StereoCalibrationResult.from_camera_pair(
            self[camera_1], self[camera_2]
        )
        return StereoCamera(
            calibration_results.camera_1,
            calibration_results.camera_2,
            config=calibration_results,
        )

    def __getitem__(self, camera_id):
        return self._cameras[camera_id]

    def __len__(self):
        return len(self._cameras)
