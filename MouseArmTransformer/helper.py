import glob
import os
import pickle
import re

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

try:
    from MouseArmTransformer.lifting.rig import MouseReachingRig5_2018

    from mausspaun.data_processing.dlc import (
        DLC_TO_MUJOCO_MAPPING,
        align_data_with_rig_markers,
        get_dlc_data,
    )
    from mausspaun.data_processing.lifting.utils import _f32, _f64
    from mausspaun.visualization import plot_3D_video

except (ImportError, ModuleNotFoundError):
    import warnings
    warnings.warn(
        "Could not import the mausspaun package. "
        "This is fine/expected for inference-only, but will raise "
        "errors if you want to retrain a model."
    )

connections = [
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('right_wrist', 'right_wrist_top'),
    ('right_wrist', 'right_wrist_bottom'),
    ('right_wrist', 'right_hand'),
    ('right_hand', 'right_finger0_0'),
    ('right_finger0_0', 'right_finger0_1'),
    ('right_finger0_1', 'right_finger0_2'),
    ('right_finger0_2', 'right_finger0_3'),
    ('right_hand', 'right_finger1_0'),
    ('right_finger1_0', 'right_finger1_1'),
    ('right_finger1_1', 'right_finger1_2'),
    ('right_finger1_2', 'right_finger1_3'),
    ('right_hand', 'right_finger2_0'),
    ('right_finger2_0', 'right_finger2_1'),
    ('right_finger2_1', 'right_finger2_2'),
    ('right_finger2_2', 'right_finger2_3'),
    ('right_hand', 'right_finger3_0'),
    ('right_finger3_0', 'right_finger3_1'),
    ('right_finger3_1', 'right_finger3_2'),
    ('right_finger3_2', 'right_finger3_3'),
    ('left_wrist', 'left_hand'),
    ('left_hand', 'left_finger0'),
    ('left_hand', 'left_finger1'),
    ('left_hand', 'left_finger2'),
    ('left_hand', 'left_finger3'),
]

mausspaun_keys = [
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'right_wrist_top',
    'right_wrist_bottom',
    'right_hand',
    'right_finger0_0',
    'right_finger0_1',
    'right_finger0_2',
    'right_finger0_3',
    'right_finger1_0',
    'right_finger1_1',
    'right_finger1_2',
    'right_finger1_3',
    'right_finger2_0',
    'right_finger2_1',
    'right_finger2_2',
    'right_finger2_3',
    'right_finger3_0',
    'right_finger3_1',
    'right_finger3_2',
    'right_finger3_3',
    'nose',
    'watertube',
    'joystick',
    'lick',
    'left_wrist',
    'left_hand',
    'left_finger0',
    'left_finger1',
    'left_finger2',
    'left_finger3',
]

DLC_TO_MUJOCO_MAPPING = {
    "R_shoulder": "right_shoulder",
    "Right_elbow": "right_elbow",
    "Right_wrist": "right_wrist",
    "R_Wrist_Top": "right_wrist_top",
    "R_Wrist_Bottom": "right_wrist_bottom",
    "Right_backofhand": "right_hand",
    # for fingers DLC starts at index 1, Mujoco model at index 0
    "R_Finger1_Base": "right_finger0_0",
    "R_Finger1": "right_finger0_1",
    "R_Finger1_Int": "right_finger0_2",
    "R_Finger1_Tip": "right_finger0_3",
    "R_Finger2_Base": "right_finger1_0",
    "R_Finger2": "right_finger1_1",
    "R_Finger2_Int": "right_finger1_2",
    "R_Finger2_Tip": "right_finger1_3",
    "R_Finger3_Base": "right_finger2_0",
    "R_Finger3": "right_finger2_1",
    "R_Finger3_Int": "right_finger2_2",
    "R_Finger3_Tip": "right_finger2_3",
    "R_Finger4_Base": "right_finger3_0",
    "R_Finger4": "right_finger3_1",
    "R_Finger4_Int": "right_finger3_2",
    "R_Finger4_Tip": "right_finger3_3",
    "nose": "nose",
    "water_tube": "watertube",
    "joystick": "joystick",
    "lick": "lick",
    # Add left hand
    "Left_wrist": "left_wrist",
    "left_backofhand": "left_hand",
    "L_Finger1": "left_finger0",
    "L_Finger2": "left_finger1",
    "L_Finger3": "left_finger2",
    "L_Finger4": "left_finger3",
    "Left_elbow": "left_elbow"
}


def sliding_windows(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        xs.append(_x)

    return np.array(xs)


def dlc_to_mujoco(mouse_markers):
    rig_markers = {
        0: np.array([1.3102248, -7.7451963, 101.0002]),
        1: np.array([5.16718, -7.587261, 101.77739]),
        2: np.array([4.965026, -4.473334, 102.719795]),
        7: np.array([3.4012034, 5.8962784, 98.65744]),
        8: np.array([18.320026, 4.855262, 118.24137])
    }
    cam_positions, dlc_c, dlc_s, T, mujoco_c, mujoco_s = align_data_with_rig_markers(data=mouse_markers,
                                                                                     dlc_markers=rig_markers)
    del cam_positions['mujoco_markers']
    del cam_positions['dlc_markers']

    return cam_positions, (dlc_c, dlc_s, T, mujoco_c, mujoco_s)


def mujoco_to_dlc(mouse_markers, dlc_c, dlc_s, T, mujoco_c, mujoco_s):
    # To transform back
    # np.dot((data[key] + dlc_c) * dlc_s, T) / mujoco_s - mujoco_c
    # muj = np.dot((dlc + dlc_c) * dlc_s, T) / mujoco_s - mujoco_c
    # muj + mujoco_c = np.dot((dlc + dlc_c) * dlc_s, T) / mujoco_s
    # (muj + mujoco_c) * mujoco_s = np.dot((dlc + dlc_c) * dlc_s, T)
    # np.dot((muj + mujoco_c) * mujoco_s), T_inv) = (dlc + dlc_c) * dlc_s
    # (np.dot((muj + mujoco_c) * mujoco_s), T_inv) / dlc_s) - dlc_c

    for key in mouse_markers.keys():
        mouse_markers[key] = (np.dot(((mouse_markers[key] + mujoco_c) * mujoco_s), np.linalg.inv(T)) / dlc_s) - dlc_c
    return mouse_markers


def get_rig(calibration_folder="/data/markus/mausspaun/calibration/data/2018_mouse_reaching_rig5/calibration/"):
    return MouseReachingRig5_2018(calibration_folder)


def project_both_cameras(rig, points_3D):
    points_cam1 = project_points(points_3D, _f32(rig.stereo_rig[0].intrinsics), _f64(rig.stereo_rig[0].rotation),
                                 _f32(rig.stereo_rig[0].translation))
    points_cam2 = project_points(points_3D, _f32(rig.stereo_rig[1].intrinsics), _f64(rig.stereo_rig[1].rotation),
                                 _f32(rig.stereo_rig[1].translation))
    return (points_cam1, points_cam2)


def project_from_3D_dlc(inference_dlc_np):
    all_cam1, all_cam2 = [], []
    rig = get_rig()
    for i in range(0, inference_dlc_np.shape[1]):
        (cam1, cam2) = project_both_cameras(rig, inference_dlc_np[:, i:i + 1, :])
        all_cam1.append(cam1)
        all_cam2.append(cam2)
    all_cam1 = np.transpose(np.array(all_cam1), axes=(2, 0, 1))
    all_cam2 = np.transpose(np.array(all_cam2), axes=(2, 0, 1))
    return (all_cam1, all_cam2)


def project_points(points_3D, intrinsic_matrix, rotation_matrix, translation_vector):
    p, _ = cv2.projectPoints(
        _f32(points_3D),
        rotation_matrix,
        translation_vector,
        intrinsic_matrix,
        distCoeffs=None,
    )
    return p.squeeze().T


def calculate_euclidean_error(array1, array2):
    squared_diff = np.sum(np.square(array1 - array2), axis=-1)
    return np.sqrt(squared_diff)


def get_training_data(cam1_path,
                      cam2_path,
                      likelihood_cutoff=0.9,
                      convert_to_mujoco=True,
                      base_path='/data/markus/mausspaun/calibration/'):
    # To be able to play around with the parameters used during filtering
    dlc_dataset_params = {
        "cam1": cam1_path,
        "cam2": cam2_path,
        "likelihood_cutoff": [0 for ii in range(33)],
        "median_cutoff": None,
    }

    dlc_dataset = triangulate_data(calibration_folder=base_path + "data/2018_mouse_reaching_rig5/calibration/",
                                   alignment_points=base_path + "honeybee_reference_points.jl",
                                   plot=False,
                                   **dlc_dataset_params)

    FPS = 75
    start = 0 * FPS
    stop = 25 * 60 * FPS

    # dataset = joblib.load(fname)
    mouse_markers, rig_markers, c1_markers, c2_markers, l1, l2 = get_dlc_data(dlc_dataset, start, stop)

    if convert_to_mujoco:
        rig_markers = {
            0: rig_markers[0],
            1: rig_markers[1],
            2: rig_markers[2],
            7: rig_markers[3],
            8: rig_markers[4],
        }
        # transform DLC data to Mujoco frame of reference
        cam_positions, dlc_c, dlc_s, T, mujoco_c, mujoco_s = align_data_with_rig_markers(data=mouse_markers,
                                                                                         dlc_markers=rig_markers)

        mujoco_markers = cam_positions['mujoco_markers']
        dlc_markers = cam_positions['dlc_markers']
        del cam_positions['mujoco_markers']
        del cam_positions['dlc_markers']
    else:
        cam_positions = mouse_markers.copy()
    # Get rid of left elbow
    del cam_positions['left_elbow']
    del c1_markers['left_elbow']
    del c2_markers['left_elbow']
    del l1['left_elbow']
    del l2['left_elbow']

    X_3d = np.array([cp for key, cp in cam_positions.items()])
    X_2d_c1 = np.array([cp for key, cp in c1_markers.items()])
    X_2d_c2 = np.array([cp for key, cp in c2_markers.items()])
    likelihood_c1 = np.array([cp for key, cp in l1.items()])
    likelihood_c2 = np.array([cp for key, cp in l2.items()])
    likelihood = np.minimum(likelihood_c1, likelihood_c2)

    print('Nan Values in 3D: ', np.sum(np.isnan(X_3d)))
    print('Nan Values in 2D Camera 1: ', np.sum(np.isnan(X_2d_c1)))
    print('Nan Values in 2D Camera 2: ', np.sum(np.isnan(X_2d_c2)))

    # Cut on likelihood
    #X_3d_nans = (likelihood < likelihood_cutoff) | (np.any(np.isnan(X_3d), axis=2))
    #X_3d_nans = X_3d_nans[:, :, np.newaxis]
    X_3d_train = np.transpose(X_3d, axes=(1, 0, 2))  #[~X_3d_nans]
    X_2d_c1_train = np.transpose(X_2d_c1, axes=(1, 0, 2))  #[~X_3d_nans]
    X_2d_c2_train = np.transpose(X_2d_c2, axes=(1, 0, 2))  #[~X_3d_nans]

    # Set NaN based on likelihood
    mask = likelihood.transpose() < likelihood_cutoff  # Create a mask where likelihood is below threshold
    mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    X_3d_train[mask] = np.nan

    print('Total 3D {}, Remaining 3D points {}, 2D Camera 1 {}, 2D Camera 2 {}'.format(
        X_3d.shape, X_3d_train.shape, X_2d_c1_train.shape, X_2d_c2_train.shape))

    return X_3d_train, X_2d_c1_train, X_2d_c2_train, cam_positions, likelihood_c1, likelihood_c2


def get_paths(base_path="/data/markus/emissions/", get_full_paths=False):
    cam1_paths = []
    cam2_paths = []
    for file in os.listdir(base_path):
        if '.h5' not in file:
            continue
        if not 'camera-1' in file:
            continue
        match = re.search(r'day-(\d+)_attempt-(\d+)_camera-(\d+)_part-(\d+)', file)
        day, attempt, camera, part = map(int, match.groups())
        other_camera = 1 if camera == 2 else 2
        other_camera_file = re.sub(r'_camera-\d+', f'_camera-{other_camera}', file)
        if not os.path.exists(base_path + other_camera_file):
            if not get_full_paths:
                print(f"Skipping {file} as {other_camera_file} does not exist")
                continue
        cam1_paths.append(base_path + file)
        cam2_paths.append(base_path + other_camera_file)
    return cam1_paths, cam2_paths


def load_data(cutoff=0.99, reload=False, path='/data/markus/mausspaun/nn_training_data/data.pkl'):
    if not reload and os.path.exists(path):
        full_X_c1, full_X_c2, full_y = pickle.load(open(path, 'rb'))
    else:
        cam1_paths, cam2_paths = get_paths()
        for idx, (cam1_path, cam2_path) in enumerate(zip(cam1_paths, cam2_paths)):
            try:
                X_3d_train, X_2d_c1_train, X_2d_c2_train, cam_positions, likelihood_c1, likelihood_c2 = get_training_data(
                    cam1_path, cam2_path, likelihood_cutoff=cutoff)
            except ValueError as e:
                print(e)
                print('Could not load {}'.format(cam1_path))
            if idx == 0:
                full_X_c1 = X_2d_c1_train
                full_X_c2 = X_2d_c2_train
                full_y = X_3d_train
            else:
                full_X_c1 = np.vstack((full_X_c1, X_2d_c1_train))
                full_X_c2 = np.vstack((full_X_c2, X_2d_c2_train))
                full_y = np.vstack((full_y, X_3d_train))
        pickle.dump((full_X_c1, full_X_c2, full_y), open(path, 'wb'))
    return full_X_c1, full_X_c2, full_y


def load_test_data(dlc_base="/data/markus/emissions/"):
    cam1_path = (
        dlc_base +
        "rigVideo_mouse-Jaguar_day-19_attempt-1_camera-1_part-0_doe-20180813_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"
        #"rigVideo_mouse-HoneyBee_day-77_attempt-1_camera-1_part-6_doe-20180803_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"
    )
    cam2_path = (
        dlc_base +
        "rigVideo_mouse-Jaguar_day-19_attempt-1_camera-2_part-0_doe-20180813_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"
        #"rigVideo_mouse-HoneyBee_day-77_attempt-1_camera-2_part-6_doe-20180803_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"
    )
    X_3d_test, X_2d_c1_test, X_2d_c2_test, cam_positions, likelihood_c1, likelihood_c2 = get_training_data(
        cam1_path, cam2_path)
    return X_3d_test, X_2d_c1_test, X_2d_c2_test, cam_positions, likelihood_c1, likelihood_c2


def load_single_session(
    dlc_base="/data/markus/emissions/",
    cam1="rigVideo_mouse-HoneyBee_day-77_attempt-1_camera-1_part-0_doe-20180803_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5",
    cam2="rigVideo_mouse-HoneyBee_day-77_attempt-1_camera-2_part-0_doe-20180803_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"
):
    cam1_path = (dlc_base + cam1)
    cam2_path = (dlc_base + cam2)
    X_3d_test, X_2d_c1_test, X_2d_c2_test, cam_positions, likelihood_c1, likelihood_c2 = get_training_data(
        cam1_path, cam2_path)
    return X_3d_test, X_2d_c1_test, X_2d_c2_test, cam_positions, likelihood_c1, likelihood_c2


def calculate_euclidean_distance(cam_positions):
    distances = {}

    for connection in connections:
        point1 = connection[0]
        point2 = connection[1]
        if point1 in cam_positions and point2 in cam_positions:
            data1 = cam_positions[point1]
            data2 = cam_positions[point2]

            if data1.shape == data2.shape:  # ensure that the data dimensions match
                dist = np.sqrt(np.nansum((data1 - data2)**2, axis=1))  # calculate Euclidean distance
                distances[connection] = dist

    return distances


def median_distances_to_tensor(median_distances, connections):
    median_tensor = torch.zeros(len(connections))
    for i, connection in enumerate(connections):
        median_tensor[i] = torch.tensor(median_distances[connection])
    return median_tensor


def calculate_euclidean_distance_4D_torch(data_array):
    distances = torch.zeros(data_array.shape[0], data_array.shape[1], len(connections), device=data_array.device)

    key_to_index = {key: i for i, key in enumerate(mausspaun_keys)}

    for i, connection in enumerate(connections):
        point1 = connection[0]
        point2 = connection[1]
        if point1 in mausspaun_keys and point2 in mausspaun_keys:
            idx1 = key_to_index[point1]
            idx2 = key_to_index[point2]

            data1 = data_array[:, :, idx1, :]
            data2 = data_array[:, :, idx2, :]

            dist = torch.sqrt(torch.nansum((data1 - data2)**2, axis=-1))  # calculate Euclidean distance
            distances[:, :, i] = dist

    return distances


def calculate_relative_displacements_median(cam_positions):
    median_displacements = {}

    for joint, positions in cam_positions.items():
        # Calculate displacement vectors between consecutive positions
        displacements = positions[1:] - positions[:-1]

        # Calculate Euclidean distances of displacement vectors
        euclidean_distances = np.sqrt(np.nansum(displacements**2, axis=-1))

        # Get median displacement for this joint
        median_displacements[joint] = np.nanmedian(euclidean_distances)

    return median_displacements


def get_relative_displacements(cam_positions):
    distances = calculate_euclidean_distance(cam_positions)
    all_aggregated = {}
    for key, item in distances.items():
        all_aggregated[key] = np.median(item)
    all_aggregated_tensor = median_distances_to_tensor(all_aggregated, connections).to('cuda')

    relative_displacements = median_distances_to_tensor(calculate_relative_displacements_median(cam_positions),
                                                        cam_positions.keys()).to('cuda')
    return all_aggregated_tensor, relative_displacements


def save_video(
    test_preds,
    cam_positions,
    info,
    smoothing_window=21,
    labeled_2d_video='/data/markus/mausspaun/labeled_videos/rigVideo_mouse-HoneyBee_day-77_attempt-1_camera-1_part-6_doe-20180803_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.mp4'
):
    all_pred_positions = {key: test_preds[:, -1, i, :] for i, key in enumerate(cam_positions.keys())}

    if smoothing_window > 0:
        for key, item in all_pred_positions.items():
            all_pred_positions[key] = savgol_filter(item, smoothing_window, 3, axis=0)

    run_name = "{epoch}_loss{loss:.3f}_seq{seq_length}_cutoff{cutoff}_lossweights{loss_weights}_smoothing{smoothing}".format(
        **info)

    plot_3D_video.plot_split_3d_video(labeled_2d_video,
                                      all_pred_positions,
                                      cam_positions=cam_positions,
                                      dpi=150,
                                      frames=(0, 500),
                                      fn_save="/data/markus/mausspaun/3D/withleft_{}".format(run_name))

    return run_name


# --- Ground truth ---
import pandas as pd
from scipy.spatial.distance import euclidean


def calculate_gt_error(frame, marker, numpy_array, df, marker_mapping):
    """
    Calculate the Euclidean error for a given frame and marker.
    """
    df_coords = np.squeeze(df[(df['frame'] == frame) & (df['bodypart'] == marker)][['x', 'y', 'z']].values)
    if marker_mapping[marker] in mausspaun_keys:
        marker_index = mausspaun_keys.index(marker_mapping[marker])
    else:
        return -1
    numpy_coords = numpy_array[frame, marker_index, :]
    error = euclidean(df_coords, numpy_coords)
    return error


def load_and_process_ground_truth(mouse_name, day, attempt, part, to_mujoco=True, gt_path="/data/mausspaun/labeled/"):
    """
    Load and process the ground truth data.
    """
    pattern = f"{gt_path}rigVideo_mouse-{mouse_name}_day-{day}_attempt-{attempt}_part-{part}_*_points3d_diff.csv"
    matching_files = glob.glob(pattern)
    assert len(matching_files) <= 1

    if not matching_files:
        print(f"Found no labels for {pattern}")
        return None
    if part != 0:
        print("Labels only valid for part 0")
        return None
    fn_ground_truth = matching_files[0]
    print('--- Found labels at {}'.format(fn_ground_truth))

    labeled_3d = pd.read_csv(fn_ground_truth)
    if to_mujoco:
        _, (dlc_c, dlc_s, T, mujoco_c, mujoco_s) = dlc_to_mujoco({})
        labeled_3d[['x', 'y', 'z']] = np.dot((labeled_3d[['x', 'y', 'z']] + dlc_c) * dlc_s, T) / mujoco_s - mujoco_c
    return labeled_3d


def compute_gt_statistics(errors_df, verbose):
    """
    Compute and print statistical information about errors.
    """
    average_error_per_marker = errors_df.groupby('marker')['error'].mean()
    max_error_per_marker = errors_df.groupby('marker')['error'].max()
    min_error_per_marker = errors_df.groupby('marker')['error'].min()
    std_deviation_per_marker = errors_df.groupby('marker')['error'].std()
    overall_average_error = errors_df['error'].mean()

    if verbose > 0:
        print("\nOverall Average Error:", overall_average_error)
        if verbose > 1:
            print("Average Error Per Marker:\n", average_error_per_marker)
            print("\nMax Error Per Marker:\n", max_error_per_marker)
            print("\nMin Error Per Marker:\n", min_error_per_marker)
            print("\nStandard Deviation Per Marker:\n", std_deviation_per_marker)


def triangulate_data(
    calibration_folder,
    alignment_points,
    save_path=None,
    plot=False,
    **dataset_kwargs,
):
    rig = MouseReachingRig5_2018(f"{calibration_folder}/")

    if plot:
        fig1 = rig.visualize_calibration(max_view=1)
        plt.show()

    dlc_data = DeepLabCutDataset(**dataset_kwargs)
    points3d = dlc_data.triangulate(rig)

    if len(dlc_data.joints) != points3d.shape[1]:
        raise Exception("Number of labels does not match number of markers")

    # get the DLC rig marker data, and triangulate to 3D
    reference_points = joblib.load(alignment_points)

    # filter for shared points
    cam1_align = dict(zip(
        reference_points["camera_1"]["labels"],
        reference_points["camera_1"]["points"],
    ))
    cam2_align = dict(zip(
        reference_points["camera_2"]["labels"],
        reference_points["camera_2"]["points"],
    ))

    shared_ids = list(set(cam1_align.keys()).intersection(set(cam2_align.keys())))
    cam1_align = np.stack([cam1_align[i] for i in shared_ids], axis=0)
    cam2_align = np.stack([cam2_align[i] for i in shared_ids], axis=0)

    arrays = [np.random.uniform(0, 1, (np.random.randint(10, 100), 5, 7)) for _ in range(10)]
    # TODO: replace with exception and message
    assert all((a == b).all() for a, b in zip(unpack(*pack(*arrays)), arrays))

    num_timepoints, keypoints, _ = dlc_data.points_camera_1.shape

    # these are the calibration points
    cpoints_1, cpoints_2 = rig.board.image_points.reshape(2, -1, 2)

    # these are the points in the video
    points_1 = dlc_data.points_camera_1.reshape(-1, 2)
    points_2 = dlc_data.points_camera_2.reshape(-1, 2)

    # package the tracked points, calibration points and alignment points
    p1, l1 = pack(points_1, cpoints_1, cam1_align)
    p2, l2 = pack(points_2, cpoints_2, cam2_align)
    # TODO: replace with exception and message
    assert l1 == l2

    # run triangulation
    points3d = rig.stereo_rig.triangulate(p1, p2)

    # unpack and reshape everything
    points3d, cpoints3d, align_points3d = unpack(points3d, l1)
    points3d = points3d.reshape(num_timepoints, keypoints, 3)

    # TODO: why is 33 hard coded?
    points_1 = points_1.reshape(-1, dlc_data.points_camera_1.shape[1], 2)
    points_2 = points_2.reshape(-1, dlc_data.points_camera_2.shape[1], 2)

    if plot:
        ax = plt.subplot(111, projection="3d")
        # TODO: ideally we're reading the labels from the dlc file
        ax.plot(*align_points3d[:3].T, label="0-2", linewidth=3, c="k")
        ax.plot(*align_points3d[3:].T, label="7-8", linewidth=3, c="k")
        ax.scatter(*points3d[:, 18].T, label="joystick (for ref)")
        ax.scatter(*points3d[:, 9].T, label="hand (for ref)")
        ax.legend()
        plt.show()

        ax = plt.subplot(111, projection="3d")
        ax.plot(*align_points3d[:3].T, label="0-2", linewidth=3, c="k")
        ax.plot(*align_points3d[3:].T, label="7-8", linewidth=3, c="k")
        ax.scatter(*cpoints3d.T, label="calibration")
        ax.legend()
        plt.show()

        def overlay_plot(camera_data):
            plt.imshow(camera_data["image"])
            for i, (x, y) in zip(camera_data["labels"], camera_data["points"]):
                plt.text(x, y, str(i), color="red")

        overlay_plot(reference_points["camera_1"])
        plt.scatter(*points_1[::10, 18].T, alpha=0.5, c=dlc_data.likelihoods[::10, 18, 0])

        plt.show()
        overlay_plot(reference_points["camera_2"])
        plt.scatter(*points_2[::10, 18].T, alpha=0.5, c=dlc_data.likelihoods[::10, 18, 1])
        plt.show()

    data = {
        "joint_links": dlc_data.joint_links,
        "points_3d": points3d,
        "alignment_points_3d": align_points3d,
        "alignment_points_camera_1": cam1_align,
        "alignment_points_camera_2": cam2_align,
        "points_camera_1": dlc_data.points_camera_1,
        "points_camera_2": dlc_data.points_camera_2,
        "likelihoods_camera_1": dlc_data.likelihoods[:, :, 0],
        "likelihoods_camera_2": dlc_data.likelihoods[:, :, 1],
        "joint_names": dlc_data.joints,
    }

    if save_path is not None:
        joblib.dump(data, save_path)
        print(f"Saved .jl files under : {save_path}")

    return data
