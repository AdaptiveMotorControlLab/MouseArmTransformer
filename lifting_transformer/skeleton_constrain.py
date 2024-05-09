import numpy as np

import joblib
import matplotlib.pyplot as plt
import pickle

from mausspaun.data_processing.dlc import align_data_with_rig_markers, get_dlc_data
from lifting_transformer.helpers import triangulate_data

from mausspaun.arm import MouseArm

from scipy.optimize import least_squares

def load_honeybee():
    base_path = '/data/markus/mausspaun/calibration/'
    likelihood_cutoff = [0.99 for ii in range(33)]

    # We have to find these in DJ
    cam1_path = base_path + "data/2018_mouse_reaching_rig5/dlc/rigVideo_mouse-HoneyBee_day-77_attempt-1_camera-1_part-0_doe-20180803_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"
    cam2_path = base_path + "data/2018_mouse_reaching_rig5/dlc/rigVideo_mouse-HoneyBee_day-77_attempt-1_camera-2_part-0_doe-20180803_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"

    # Convert to Mujoco reference frame
    dlc_dataset_params = {
        "cam1": cam1_path,
        "cam2": cam2_path,
        "likelihood_cutoff": likelihood_cutoff,
    }

    # Or just the triangulated data
    dataset = triangulate_data(
        calibration_folder=base_path + 'data/2018_mouse_reaching_rig5/calibration/',
        alignment_points=base_path + "honeybee_reference_points.jl",
        plot=False,
        **dlc_dataset_params
    )

    FPS = 75
    start = 0 * FPS
    stop = 25 * 60 * FPS

    # dataset = joblib.load(fname)
    mouse_markers, rig_markers = get_dlc_data(
        dataset, start, stop, confidence_threshold=0.99
    )

    # TODO: need to specify the corresponding rig markers identification numbers
    # need to update the way rig markers are saved when the joblib is created
    # here, we're using (0, 1, 2, 7, 8) (identified in impala.png)
    rig_markers = {
        0: rig_markers[0],
        1: rig_markers[1],
        2: rig_markers[2],
        7: rig_markers[3],
        8: rig_markers[4],
    }

    # transform DLC data to Mujoco frame of reference
    cam_positions = align_data_with_rig_markers(data=mouse_markers, dlc_markers=rig_markers)

    mujoco_markers = cam_positions['mujoco_markers']
    dlc_markers = cam_positions['dlc_markers']
    del cam_positions['mujoco_markers']
    del cam_positions['dlc_markers']

    key_mapping = {
        'finger0_0': 'finger0_0_dlc',
        'finger0_1': 'finger0_1_dlc',
        'finger0_2': 'finger0_2_dlc',
        'finger0_3': 'finger0_3_dlc',
        'finger1_0': 'finger1_0_dlc',
        'finger1_1': 'finger1_1_dlc',
        'finger1_2': 'finger1_2_dlc',
        'finger1_3': 'finger1_3_dlc',
        'finger2_0': 'finger2_0_dlc',
        'finger2_1': 'finger2_1_dlc',
        'finger2_2': 'finger2_2_dlc',
        'finger2_3': 'finger2_3_dlc',
        'finger3_0': 'finger3_0_dlc',
        'finger3_1': 'finger3_1_dlc',
        'finger3_2': 'finger3_2_dlc',
        'finger3_3': 'finger3_3_dlc',
        'hand': 'backofhand',
    }
    def rename_keys(mapping, target_dict):
        for old_key, new_key in mapping.items():
            if old_key in target_dict:
                target_dict[new_key] = target_dict.pop(old_key)

    rename_keys(key_mapping, cam_positions)

    return cam_positions

def load_mousearm():
    mousearm = MouseArm(
        use_muscles=False,
        use_sim_state=False,
        xml_kwargs={
            "add_rig": True,
            "add_rig_markers": False,
            "add_joystick": False,
            "add_joint_targets": True
        },
        visualize=False,
   )
    return mousearm

def get_positions(mousearm, q):
    body_parts = ["shoulder", "elbow", "wrist", "wrist_top", "wrist_bottom", "backofhand",
                "finger0_0_dlc", "finger0_1_dlc", "finger0_2_dlc", "finger0_3_dlc",
                "finger1_0_dlc", "finger1_1_dlc", "finger1_2_dlc", "finger1_3_dlc",
                "finger2_0_dlc", "finger2_1_dlc", "finger2_2_dlc", "finger2_3_dlc",
                "finger3_0_dlc", "finger3_1_dlc", "finger3_2_dlc", "finger3_3_dlc"]

    positions = {}
    for part in body_parts:
        positions[part] = mousearm.config.Tx(part, object_type="body", q=q)

    return positions

def optimize_3D_fit(x, mousearm, cam_positions, t=0, prev_positions=None, exclude=['wrist_bottom', 'wrist_top']):
    positions = get_positions(mousearm, x)
    residuals = np.array(list(calc_residuals(positions, cam_positions, t=t, exclude=exclude).values()))

    if prev_positions is not None:
        for prev_position in prev_positions:
            prev_residuals = []
            assert len(prev_position) == len(positions.keys())
            for idx, key in enumerate(positions.keys()):
                if key not in exclude:
                    distance_to_prev = np.linalg.norm(np.array(positions[key]) - np.array(prev_position[idx]))
                    prev_residual = (0.5**(idx+1)) * distance_to_prev
                    prev_residuals.append(prev_residual)
            residuals = residuals + np.array(prev_residuals)

    return residuals

def calc_residuals(arm_positions, cam_positions, t=0, exclude=[None]):
    # Calculate residuals
    residuals = {}
    for key in arm_positions.keys():
        if (key in cam_positions) and (key not in exclude):
            arm_position = np.array(arm_positions[key])
            cam_position = np.array(cam_positions[key][t,:])
            distance = np.linalg.norm(arm_position - cam_position)
            residuals[key] = distance
    return residuals

def get_positions_from_result(mousearm, result, cam_positions, t):
    arm_position = get_positions(mousearm, result.x)
    cam_position = {key: cam_positions[key][t, :] for key in cam_positions}
    return (arm_position, cam_position)

def run_optimization(mousearm, cam_positions):
    initial_guess = [0, -1, 0 ,  0, 0, -0.5,  0.5,  0,  0, 0]

    all_arm_positions_list = []
    # Get keys:
    initial_positions = get_positions(mousearm, initial_guess)
    keys_list = list(initial_positions.keys())
    tolerance = 1e-4

    for t in range(0, cam_positions['shoulder'].shape[0]):
        print('{} / {}'.format(t, cam_positions['shoulder'].shape[0] - 1), end='\r')

        prev_positions_list = None
        if t > 0:
            start_idx = max(0, t - 5)
            prev_positions_list = [all_arm_positions_list[idx] for idx in range(start_idx, t)][::-1]

        result = least_squares(
            optimize_3D_fit,
            initial_guess,
            args=(mousearm, cam_positions, t, prev_positions_list),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
        )
        (arm_position, cam_position) = get_positions_from_result(mousearm, result, cam_positions, t)
        arm_position_list = [arm_position[key] for key in keys_list]
        all_arm_positions_list.append(arm_position_list)

        initial_guess = result.x

    all_arm_positions = {key: np.array([pos_list[i] for pos_list in all_arm_positions_list]) for i, key in enumerate(keys_list)}
    return all_arm_positions

def main():
    cam_positions = load_honeybee()
    mousearm = load_mousearm()

    print('Run optimization')
    all_arm_positions = run_optimization(mousearm, cam_positions)

    with open('/data/markus/mausspaun/3D/HoneyBee_day-77_attempt-1_camera-1_part-0.pkl', 'wb') as f:
        pickle.dump(all_arm_positions, f)

if __name__ == "__main__":
    main()
