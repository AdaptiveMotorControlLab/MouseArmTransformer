import glob

import numpy as np
import matplotlib.pyplot as plt

from MouseArmTransformer.lifting.board import Checkerboard
from MouseArmTransformer.lifting.calibration import CameraCalibration


class RigConfig:
    def __init__(self, images, width, height):

        self.board = Checkerboard(
            file_names=images, num_cameras=2, width=width, height=height
        )
        self.camera_calibration = CameraCalibration(self.board)
        self.stereo_rig = self.camera_calibration.stereo_calibration(
            camera_1=0, camera_2=1
        )

    def visualize_calibration(self, max_view=None):
        points3d = self.stereo_rig.triangulate(*self.board.image_points)

        for view_id, view in enumerate(points3d):

            world_points = view.T
            world_points = np.concatenate(
                [world_points, np.ones((1, world_points.shape[1]))], axis=0
            )

            cam_point_1 = self.stereo_rig[0].project_points(world_points)
            cam_point_2 = self.stereo_rig[1].project_points(world_points)

            fig, axes = plt.subplots(1, 2, figsize=(7, 3))

            axes[0].imshow(self.board.images[0, view_id], cmap="gray")
            axes[0].scatter(*cam_point_1, s=5, c="C1")

            axes[1].imshow(self.board.images[1, view_id], cmap="gray")
            axes[1].scatter(*cam_point_2, s=5, c="C1")

            for ax in axes:
                ax.axis("off")

            fig.show()

            if max_view is not None and view_id + 1 >= max_view:
                break

        return fig

    def triangulate_calibration(self):
        """Run triangulation on the calibration images"""

        points = self.board.image_points
        points3d = self.stereo_rig.triangulate(*points)
        return points3d


class MouseReachingRig5_2018(RigConfig):

    def __init__(self, data_folder):

        front_calibration = glob.glob(
            f"{data_folder}/CAL_front*.jpg"
        )
        side_calibration = glob.glob(
            f"{data_folder}/CAL_side*.jpg"
        )
        images = list(zip(sorted(side_calibration), sorted(front_calibration)))
        width = 6
        height = 8
        super().__init__(images=images, width=6, height=8)
