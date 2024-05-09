import numpy as np
import glob
import os

import cv2
import PIL.Image
import PIL.ImageOps
import dataclasses
import itertools

import PIL.Image

from mausspaun.data_processing.lifting import opencv, utils


@dataclasses.dataclass
class Camera:
    """A simple perspective camera model without distortion."""

    intrinsics: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def _check_intrinsics(self):
        assert self.intrinsics[1, 0] == 0
        assert self.intrinsics[0, 1] == 0
        assert (self.intrinsics[2, :2] == 0).all()

    def _check_extrinsics(self):
        assert self.rotation.shape == (3, 3)
        assert np.allclose(self.rotation.T @ self.rotation, np.eye(3))

    def __post_init__(self):
        self._check_intrinsics()
        self._check_extrinsics()

    @property
    def translation_matrix(self):
        return self.translation.reshape(3, 1)

    @property
    def extrinsics(self):
        return np.concatenate([self.rotation, self.translation_matrix], axis=1)

    @property
    def fx(self):
        return self.intrinsics[0, 0]

    @property
    def fy(self):
        return self.intrinsics[1, 1]

    @property
    def cx(self):
        return self.intrinsics[0, 2]

    @property
    def cy(self):
        return self.intrinsics[1, 2]

    @property
    def projection(self):
        return self.intrinsics @ self.extrinsics

    def world_to_view(self, points3d):
        """Project a world point into the camera coordinate system.

        Note: This operation does *not* return the location of a point
        on the camera image. If you want this, have a look at `project_points`.
        """
        R = self.rotation
        T = self.translation
        return points3d @ R.T + T.T

    def view_to_world(self, points3d):
        """Project a point from camera coordinates into the world coordinate system.

        By convention, the world coordinate system is the camera coordinate system
        of a camera with idenity rotation and zero translation.
        """
        R = self.rotation
        T = self.translation
        return (points3d - T.T) @ np.linalg.inv(R.T)

    def project_points(self, world_points, variant="re"):
        """Project world points into image coordinates."""

        assert variant in ["cv2", "re", "np"]
        assert world_points.shape[0] == 4

        if variant == "cv2":
            return opencv.project_points(self, world_points)

        if variant == "re":

            # An attempt to re-implement
            # https://github.com/opencv/opencv/blob/
            #  e655083e3c09e3c28b336326d2fc6c88fdaeffd0/
            #  modules/calib3d/src/calibration.cpp#L522,
            # although without non-linearities.

            X = world_points[:3]

            x, y, z = self.rotation @ X + self.translation
            z = np.where(z == 0, 1.0, 1.0 / z)
            x *= z
            y *= z

            x = x * self.fx + self.cx
            y = y * self.fy + self.cy

            return np.stack([x, y], axis=0)

        raise ValueError(variant)


@dataclasses.dataclass
class StereoCamera:
    """A stereo rig consisting of two calibrated cameras."""

    camera_1: Camera
    camera_2: Camera
    config: object = None

    def triangulate(self, points_1, points_2):
        points_1, shape_1 = utils.pack_array(points_1)
        points_2, shape_2 = utils.pack_array(points_2)
        assert shape_1 == shape_2

        points3d = opencv.triangulate(self, points_1, points_2)
        return utils.unpack_array(points3d, shape_1)

    def __getitem__(self, index):
        if index == 0:
            return self.camera_1
        if index == 1:
            return self.camera_2
        raise StopIteration()

    def __len__(self):
        return 2

    def reprojection(self, points):
        """Triangulate and re-project the given point array

        Returns a list of dictionaries, containing reprojected points
        as well as distances to the original points for both cameras.
        """

        reprojection = []
        points3d = self.triangulate(*points)
        points3d_h = utils.to_homographic(points3d.reshape(-1, 3).T)
        for camera_id, points in enumerate(points):
            reproject = (
                self[camera_id]
                .project_points(points3d_h, "cv2")
                .T.reshape(points.shape)
            )
            error = np.linalg.norm(reproject - points, axis=-1)
            reprojection.append(dict(error=error, points=reproject))
        return reprojection
