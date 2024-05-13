"""Calibration boards

Calibration boards contain points in image and world space and can be
created either manually, from paired images, or paired video. The calibration
board should be pre-defined. If you have custom calibration boards in use,
you can add them to this file.
"""

import glob
import os

import cv2
import dataclasses
import itertools
import PIL.Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class Board:
    """Base class for the calibration board."""

    def __post_init__(self):
        self.images = {}
        self.frame_sizes = {k: None for k in range(self.num_cameras)}

    @property
    def object_points_3d(self):
        return NotImplemented

    @property
    def image_points(self):
        return NotImplemented

    @property
    def num_points(self):
        return NotImplemented

    num_cameras: int


class ManualBoard(Board):
    """Manually specify calibration points."""

    pass


@dataclasses.dataclass
class ImageFilesBoard(Board):
    """Load calibration board from a file list."""

    file_names: list
    ignore_shape_mismatch = "topleft"

    def load_calibration_image(self, fname, subpix=False, show_image=False):
        return NotImplemented

    @property
    def num_frames(self):
        return len(self.file_names)


@dataclasses.dataclass
class Checkerboard(ImageFilesBoard):
    """Checkerboard calibration pattern."""

    width: int
    """Number of corner points in horizontal image direction."""

    height: int
    """Number of corner points in vertical image direction."""

    width_units: float = 1.0
    """Optional."""

    height_units: float = 1.0
    """Optional."""

    unit_name: str = "units"
    """Optional. Name the units of your calibration pattern (e.g. mm)."""

    def __post_init__(self):
        super().__post_init__()
        self._image_points = self._init_calibration_points_2d(nan=True)
        self.calibration_images = {}
        for image_id, image_list in enumerate(self.file_names):
            assert len(image_list) == self.num_cameras, len(image_list)
            for camera_id, camera_image in enumerate(image_list):
                self._load_image_for_camera(camera_id, image_id, camera_image)
        assert not np.isnan(self.image_points).any()

    def load_calibration_image(self, fname, subpix=False, show_image=False):

        calib_img = PIL.Image.open(fname).convert("L")
        calib_img = np.array(calib_img)
        success, corners = cv2.findChessboardCorners(
            calib_img,
            (self.width, self.height),
            None,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH,
        )
        assert success
        if subpix:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(calib_img, corners, (1, 1), (-1, -1), criteria)
        return corners, calib_img.shape, calib_img

    @property
    def num_points(self):
        return self.width * self.height

    @property
    def image_points(self):
        return self._image_points

    @property
    def object_points_3d(self):
        objp = np.zeros((self.num_points, 3), np.float64)
        objp[:, :2] = np.mgrid[0 : self.width, 0 : self.height].T.reshape(-1, 2)
        return objp

    def _init_calibration_points_2d(self, nan=True):
        points = np.zeros((self.num_cameras, self.num_frames, self.num_points, 2))
        if nan:
            points[:] = np.nan
        return points

    def _load_image_for_camera(self, camera_id, image_id, camera_image):
        corners, image_shape, calib_img = self.load_calibration_image(camera_image)
        self.images[camera_id, image_id] = calib_img
        self.image_points[camera_id, image_id, :, :] = corners[:, 0, :]

        if self.frame_sizes[camera_id] is not None:
            if self.frame_sizes[camera_id] != image_shape:
                print(
                    f"Shape of {camera_image} is {image_shape}, while "
                    f"at leat one previous image was of shape "
                    f"{self.frame_sizes[camera_id]}."
                )
                if self.ignore_shape_mismatch == "topleft":
                    h, w = self.frame_sizes[camera_id]
                    h_, w_ = image_shape
                    self.frame_sizes[camera_id] = (max(h, h_), max(w, w_))
                    print(
                        f"Ignoring the mismatch and keeping the larger image: "
                        f"{self.frame_sizes[camera_id]}."
                    )
        else:
            self.frame_sizes[camera_id] = image_shape

    def __getitem__(self, key):
        """Return the calibration image of a particular view, along with the detected keypoints."""
        return self.images(key)
