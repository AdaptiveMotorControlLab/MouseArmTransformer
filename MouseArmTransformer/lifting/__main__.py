"""Main entry point for lifting and calibration.

This script is the main CLI interface and also contains some
commented code to get a better understanding of how the full pipeline
works. In general, the necessary steps are:

    (1) Calibration of a camera pair and initialization of the stereo
        rig.
    (2) Triangulation of all points.
    (3) Supervised training of a lifting network.

Optionally, manual correction is performed between steps (2) and (3).
Currently, steps (1) and (2) are implemented.
"""

import logging

logging.basicConfig(level=logging.INFO)

import os
import PIL.Image
import numpy as np
import PIL.ImageOps
import cv2
import matplotlib.pyplot as plt
import dataclasses
import itertools
import glob
import PIL.Image
import argparse

import board
import calibration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", "-l", nargs="+", required=True)
    parser.add_argument("--right", "-r", nargs="+", required=True)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--no-sort", action="store_true")
    parser.add_argument("--log-level", default="info")

    args = parser.parse_args()
    if not args.no_sort:
        args.left = sorted(args.left)
        args.right = sorted(args.right)
        print(f"{args.left=}")
        print(f"{args.right=}")
        assert len(args.left) == len(args.right)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logging.basicConfig(level=numeric_level)

    return args


def main():
    args = parse_args()
    logging.info(f"Starting calibration with args {args}.")

    # We start with initializing the checkerboard pattern. You can
    # replace this part by your own configurations. If you already
    # pre-computed your calibration points, you can also make use of
    # the lifting.ManualBoard class!
    images = [[left, right] for left, right in zip(args.left, args.right)]
    checkerboard = board.Checkerboard(
        file_names=images, num_cameras=2, width=args.width, height=args.height
    )

    # These are the two most important function of the calibration
    # board, specifying paired (noisy) measurements.
    logging.info(f"Loaded object points: {checkerboard.object_points_3d.shape}")
    logging.info(f"Loaded image points: {checkerboard.image_points.shape}")

    # The calibration module leverages these correspondences between
    # the two boards and computes camera intrinsics along with
    # extrinsics for each camera and view pair.
    camera_calibration = calibration.CameraCalibration(checkerboard)
    print(camera_calibration.num_cameras)

    logging.debug(f"Computed intrinsics for {camera_calibration.num_cameras}.")

    # For any pair of two cameras, we can now compute the extrinsic parameters.
    # By convention, the first camera in the stereo rig will serve as the reference
    # cameras, i.e., its rotation matrix will be the identity matrix and its
    # translation vector will be zero.
    stereo_rig = camera_calibration.stereo_calibration(camera_1=0, camera_2=1)

    logging.info(
        f"Computed extrinsic parameters. Full camera configuration:\n"
        f"Camera1: {stereo_rig[0]}\n"
        f"Camera2: {stereo_rig[1]}\n"
    )

    # We can now check the quality of each step. We will first project the calibration
    # points into world space, and check the reprojection error.
    result = stereo_rig.reprojection(checkerboard.image_points)
    for camera_id, result_dict in enumerate(result):
        logging.info(
            f"Camera {camera_id} has reproject error {result_dict['error'].mean():.1f} px."
        )


if __name__ == "__main__":
    main()
