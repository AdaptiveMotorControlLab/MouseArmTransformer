"""Wrappers around OpenCV functions used for camera calibration."""

import numpy as np
import glob
import os

import cv2
import dataclasses
import itertools
import PIL.Image
import PIL.ImageOps

from MouseArmTransformer.lifting.camera import Camera
from MouseArmTransformer.lifting.utils import _f32, _f64


def project_points(camera, world_points):
    p, _ = cv2.projectPoints(
        _f32(world_points[:3]),
        _f64(camera.rotation),
        _f32(camera.translation),
        _f32(camera.intrinsics),
        distCoeffs=None,
    )
    return p.squeeze().T


def triangulate(stereo, points_1, points_2):
    """Triangulate points for a given stereo rig."""

    assert points_1.shape[1] == 2, points_1.shape
    assert points_2.shape[1] == 2, points_2.shape

    points3d_h = cv2.triangulatePoints(
        projMatr1=_f32(stereo.camera_1.projection),
        projMatr2=_f32(stereo.camera_2.projection),
        projPoints1=_f32(points_1.T),
        projPoints2=_f32(points_2.T),
    )
    points3d = (points3d_h[:3] / points3d_h[3]).T
    return points3d


@dataclasses.dataclass
class CameraCalibrationResult:
    """An initial estimate for a camera and a calibration view.

    Holds information about the estimated intrinsic parameters of the
    camera long with extrinsic parameters for each view the camera is
    calibrated with.
    """

    @classmethod
    def from_calibration_points(cls, object_points, image_points, size):
        parameters = cv2.calibrateCamera(object_points, image_points, size, None, None)
        return cls(*parameters, object_points, image_points, size)

    ret: np.ndarray
    intrinsics: np.ndarray
    distances: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    object_points: list
    image_points: list
    size: list

    @property
    def rotation_matrix(self):
        R, _ = cv2.Rodrigues(rvecs[0], jacobian=None)
        return R

    def project_points(self, view_id):
        imgpoints_proj, _ = cv2.projectPoints(
            self.object_points[view_id],
            self.rotation_vector[view_id],
            self.translation_vector[view_id],
            self.intrinsics,
            self.distances,
        )
        return imgpoints_proj


@dataclasses.dataclass
class StereoCalibrationResult:
    """A stereo calibration setup constructed from two camera views"""

    @classmethod
    def from_camera_pair(cls, camera_1, camera_2):
        assert all(
            (i == j).all()
            for i, j in zip(camera_1.object_points, camera_2.object_points)
        )
        parameters = cv2.stereoCalibrate(
            objectPoints=camera_1.object_points,
            imagePoints1=camera_1.image_points,
            imagePoints2=camera_2.image_points,
            cameraMatrix1=camera_1.intrinsics,
            distCoeffs1=camera_1.distances,
            cameraMatrix2=camera_2.intrinsics,
            distCoeffs2=camera_2.distances,
            imageSize=None,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )

        return cls(*parameters)

    retval: np.array

    camera_matrix_1: np.array

    distortion_coeffs_1: np.array

    camera_matrix_2: np.array

    distortion_coeffs_2: np.array

    rotation_matrix: np.array
    """Output rotation matrix.

    Together with the translation vector T, this matrix brings points given in the
    first camera's coordinate system to points in the second camera's coordinate system.
    In more technical terms, the tuple of R and T performs a change of basis from the
    first camera's coordinate system to the second camera's coordinate system. Due to
    its duality, this tuple is equivalent to the position of the first camera with
    respect to the second camera coordinate system.
    """

    translation_matrix: np.array
    """Output translation vector.

    See description above.
    """

    essential_matrix: np.array
    fundamental_matrix: np.array

    @property
    def camera_1(self):
        return Camera(
            distortion=self.distortion_coeffs_1,
            intrinsics=self.camera_matrix_1,
            translation=np.zeros((3, 1)),
            rotation=np.eye(3),
        )

    @property
    def camera_2(self):
        return Camera(
            distortion=self.distortion_coeffs_2,
            intrinsics=self.camera_matrix_2,
            rotation=self.rotation_matrix,
            translation=self.translation_matrix,
        )

    def __getitem__(self, key):
        camera_id, key = key
        if key in ["camera_matrix", "distance_coeffs", "projection"]:
            return getattr(self, f"{key}_{camera_id+1}")
        raise ValueError(key)


@dataclasses.dataclass
class StereoRectificationResults:
    """Rectification.

    This only makes sense for proper stereo camera setups, e.g. when the
    cameras are close to parallel to each other.
    """

    @classmethod
    def from_stereo_calibration(cls, stereo_calib):
        parameters = cv2.stereoRectify(
            cameraMatrix1=stereo_calib.camera_matrix_1,
            distCoeffs1=stereo_calib.distance_coeffs_1,
            cameraMatrix2=stereo_calib.camera_matrix_1,
            distCoeffs2=stereo_calib.distance_coeffs_2,
            imageSize=None,
            R=stereo_calib.rotation_matrix,
            T=stereo_calib.translation_matrix,
            alpha=0.5,
        )
        return cls(*parameters)

    rectification_1: np.ndarray
    """Output 3x3 rectification transform (rotation matrix) for the first camera.

    This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified
    first camera's coordinate system. In more technical terms, it performs a change of basis from the
    unrectified first camera's coordinate system to the rectified first camera's coordinate system.
    """

    rectification_2: np.ndarray
    """Output 3x3 rectification transform (rotation matrix) for the second camera.

    This matrix brings points given in the unrectified second camera's coordinate system to points in the rectified
    second camera's coordinate system. In more technical terms, it performs a change of basis from the
    unrectified second camera's coordinate system to the rectified second camera's coordinate system.
    """

    projection_1: np.ndarray
    """Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.

    It projects points given in the rectified first camera coordinate system into the
    rectified first camera's image.
    """

    projection_2: np.ndarray
    """Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera

    It projects points given in the rectified first camera coordinate system into the
    rectified second camera's image.
    """

    disparity_to_depth_mapping: np.ndarray
    """Output 4x4 disparity-to-depth mapping matrix (see @ref reprojectImageTo3D)."""

    valid_pix_roi_1: np.ndarray
    """Optional output rectangles inside the rectified images where all the pixels

    are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
    (see the picture below).
    """

    valid_pix_roi_2: np.ndarray
    """Optional output rectangles inside the rectified images where all the pixels are valid.

    If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
    (see the picture below).
    """
