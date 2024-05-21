import datetime
import getpass
import glob
import json
import os
import random
import tkinter as tk

import cv2
import joblib
import numpy as np
import pandas as pd
from matplotlib.path import Path
from sklearn.decomposition import PCA

from mausspaun.data_processing.dlc import align_data_with_rig_markers
from mausspaun.data_processing.lifting.rig import MouseReachingRig5_2018
from mausspaun.data_processing.utils import interp_nan, pack, unpack


# --- For Triangulation ---
def load_rig_and_calibration_points(alignment_points, calibration_folder):
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

    # these are the calibration points
    rig = MouseReachingRig5_2018(f"{calibration_folder}/")
    cpoints_1, cpoints_2 = rig.board.image_points.reshape(2, -1, 2)

    return rig, cpoints_1, cpoints_2, cam1_align, cam2_align


def triangulate_points_from_rig(points1, points2, triangulation_data):
    rig, cpoints_1, cpoints_2, cam1_align, cam2_align = triangulation_data

    num_timepoints, keypoints, _ = points1.shape

    # these are the points in the video
    points_1 = points1.reshape(-1, 2)
    points_2 = points2.reshape(-1, 2)

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

    return points3d


def center_window_relative_to_parent(parent, child):
    # wait for the child window to be drawn so that we can get its dimensions
    child.update_idletasks()

    # get parent window width, height and position
    parent_width = parent.winfo_width()
    parent_height = parent.winfo_height()
    parent_x = parent.winfo_rootx()
    parent_y = parent.winfo_rooty()

    # get child window width and height
    child_width = child.winfo_width()
    child_height = child.winfo_height()

    # calculate position x and y coordinates
    x = parent_x + (parent_width / 2) - (child_width / 2)
    y = parent_y + (parent_height / 2) - (child_height / 2)

    child.geometry('%dx%d+%d+%d' % (child_width, child_height, x, y))


def align_to_mujoco(markers):
    alignment_points = np.array(
        [[1.3102256, -7.7451963, 101.00019], [5.1671805, -7.587261, 101.777374], [4.965027, -4.473334, 102.71978],
         [3.401204, 5.896279, 98.65743], [18.320026, 4.8552613, 118.24135]],
        dtype=np.float32)

    rig_markers = {
        0: alignment_points[0],
        1: alignment_points[1],
        2: alignment_points[2],
        7: alignment_points[3],
        8: alignment_points[4],
    }

    markers_mujoco, dlc_c, dlc_s, T, mujoco_c, mujoco_s = align_data_with_rig_markers(
        data=markers,
        dlc_markers=rig_markers,
    )
    del markers_mujoco['mujoco_markers']
    del markers_mujoco['dlc_markers']
    return markers_mujoco


def draw_connections(ax, positions, c='w'):
    connections_right = [('R_shoulder', 'Right_elbow'), ('Right_elbow', 'Right_wrist'), ('Right_wrist', 'R_Wrist_Top'),
                         ('Right_wrist', 'R_Wrist_Bottom'), ('Right_wrist', 'Right_backofhand'),
                         ('Right_backofhand', 'R_Finger1_Base'), ('R_Finger1_Base', 'R_Finger1_Int'),
                         ('R_Finger1_Int', 'R_Finger1_Tip'), ('Right_backofhand', 'R_Finger2_Base'),
                         ('R_Finger2_Base', 'R_Finger2_Int'), ('R_Finger2_Int', 'R_Finger2_Tip'),
                         ('Right_backofhand', 'R_Finger3_Base'), ('R_Finger3_Base', 'R_Finger3_Int'),
                         ('R_Finger3_Int', 'R_Finger3_Tip'), ('Right_backofhand', 'R_Finger4_Base'),
                         ('R_Finger4_Base', 'R_Finger4_Int'), ('R_Finger4_Int', 'R_Finger4_Tip')]

    # Add left side
    connections_left = [('Left_elbow', 'Left_wrist'), ('Left_wrist', 'left_backofhand'),
                        ('left_backofhand', 'L_Finger1'), ('left_backofhand', 'L_Finger2'),
                        ('left_backofhand', 'L_Finger3'), ('left_backofhand', 'L_Finger4')]

    for con_str, connection in zip(['connections_left', 'connections_right'], [connections_left, connections_right]):
        for start, end in connection:
            # Check if both points exist
            if start not in positions or end not in positions:
                continue
            xs = [positions[start][0], positions[end][0]]
            ys = [positions[start][1], positions[end][1]]
            zs = [positions[start][2], positions[end][2]]
            if con_str == 'connections_left':
                ax.plot(xs, ys, zs, '{}-'.format(c), alpha=0.5)
            else:
                ax.plot(xs, ys, zs, '{}-'.format(c), alpha=1)


def bodypart_to_color(bodypart):
    colors = {
        'nose': "#fffffc",
        'Left_wrist': "#0000ff",
        'left_backofhand': "#3333ff",
        'L_Finger1': "#6666ff",
        'L_Finger2': "#9999ff",
        'L_Finger3': "#99ccff",
        'L_Finger4': "#66ccff",
        'Left_elbow': "#33ccff",
        'Right_wrist': "#ff0000",
        'Right_backofhand': "#ff3300",
        'R_Finger1': "#ff6600",
        'R_Finger2': "#ff9900",
        'R_Finger3': "#ffcc00",
        'R_Finger4': "#ffcc66",
        'Right_elbow': "#ffcc99",
        'water_tube': "#00ff00",
        'lick': "#800080",
        'R_shoulder': "#ff33cc",
        'joystick': "#999999",
        'R_Finger1_Int': "#ff6600",
        'R_Finger2_Int': "#ff9900",
        'R_Finger3_Int': "#ffcc00",
        'R_Finger4_Int': "#ffcc66",
        'R_Finger1_Base': "#ff6633",
        'R_Finger2_Base': "#ff9933",
        'R_Finger3_Base': "#ffcc33",
        'R_Finger4_Base': "#ffcc99",
        'R_Finger1_Tip': "#ff6633",
        'R_Finger2_Tip': "#ff9933",
        'R_Finger3_Tip': "#ffcc33",
        'R_Finger4_Tip': "#ffcc99",
        'R_Wrist_Top': "#ff3300",
        'R_Wrist_Bottom': "#ff0000",
    }
    return colors.get(bodypart, "#ffffff")


# --- Save ---
def save(self):
    # Define filename
    fname = f"rigVideo_mouse-{self.args.mouse_name}_day-{self.args.day}_attempt-{self.args.attempt}_part-{self.args.part}_"
    # Add the current date up to minutes
    fname_with_date = fname + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M_")

    # Output dir saves the most recent version of the labeled data, output dir save stores all versions
    output_dir = os.path.join(self.base_path, 'labeled')
    output_dir_save = os.path.join(self.base_path, 'labeled', 'save')

    # Get dataframes
    cam1_df, cam2_df, points3d_df, cam1_diff, cam2_diff, points3d_df_diff = get_dataframes_for_saving(self)

    # Save into output_dir_save
    save_dataframes_csv(output_dir_save, fname_with_date, cam1_df, cam2_df, points3d_df, cam1_diff, cam2_diff,
                        points3d_df_diff)

    # Check if session exists in output_dir, then delete and save new version
    files = os.listdir(output_dir)
    files = [f for f in files if fname in f]
    if len(files) > 0:  # Previous version exists
        # Delete previous version
        for f in files:
            os.remove(os.path.join(output_dir, f))
    # Save into output_dir
    save_dataframes_csv(output_dir, fname_with_date, cam1_df, cam2_df, points3d_df, cam1_diff, cam2_diff,
                        points3d_df_diff)

    print(f"Saved labeled data to {output_dir} as {fname_with_date}cam[1,2].csv and {fname_with_date}points3d.csv")


def save_dataframes_csv(output_dir, fname_with_date, cam1_df, cam2_df, points3d_df, cam1_diff, cam2_diff,
                        points3d_df_diff):
    files = [
        f"{fname_with_date}cam1.csv", f"{fname_with_date}cam2.csv", f"{fname_with_date}points3d.csv",
        f"{fname_with_date}cam1_diff.csv", f"{fname_with_date}cam2_diff.csv", f"{fname_with_date}points3d_diff.csv"
    ]

    # Change permissions too
    for file, df in zip(files, [cam1_df, cam2_df, points3d_df, cam1_diff, cam2_diff, points3d_df_diff]):
        file_path = os.path.join(output_dir, file)
        df.to_csv(file_path, na_rep='NaN')
        os.chmod(file_path, 0o777)  # rwxrwxrwx permissions


def get_dataframes_for_saving(self):
    # Convert lists of points to DataFrames
    cam1_df = self.camera1.points
    cam2_df = self.camera2.points
    bodypart_names = cam1_df.columns.get_level_values(1).unique()  # Get unique body part names

    # Reshape data to 2D
    reshaped_points = self.all_points3d.reshape(-1, self.all_points3d.shape[-1])
    index = pd.MultiIndex.from_product([range(self.all_points3d.shape[0]), bodypart_names], names=['frame', 'bodypart'])
    points3d_df = pd.DataFrame(reshaped_points, index=index, columns=['x', 'y', 'z'])

    # Check difference to self.camera1.original_points and then only save the points that are different
    _, _, cam1_diff, cam2_diff = get_labeled_frames(self)

    # Filter the points3d_df DataFrame
    diff_indices = cam1_diff.index.union(cam2_diff.index)
    points3d_df_diff = points3d_df.loc[diff_indices].dropna() # Drop NaN values due to the camera shift

    return cam1_df, cam2_df, points3d_df, cam1_diff, cam2_diff, points3d_df_diff


def get_labeled_frames(self):
    cam1_diff = self.camera1.points[self.camera1.points != self.camera1.original_points]
    labeled_cam1 = cam1_diff.notna().any(axis=1)

    cam2_diff = self.camera2.points[self.camera2.points != self.camera2.original_points]
    labeled_cam2 = cam2_diff.notna().any(axis=1)

    # For fixing mislabeled frames:
    # labeled_cam2[1] = False
    # self.camera2.points.iloc[1] = self.camera2.original_points.iloc[1]

    return labeled_cam1.values, labeled_cam2.values, cam1_diff[labeled_cam1], cam2_diff[labeled_cam2]


def get_camera_offset(mouse_name, day):
    """Used for aligning the frames of camera 1 and camera 2"""
    day = int(day)
    if mouse_name == 'HoneyBee' and day == 77:
        return 5
    elif mouse_name == 'Jaguar' and day == 19:
        return 3
    elif mouse_name == 'HoneyBee' and (day in [81, 82, 83, 84, 86]):
        return 3
    else:
        return 0  # Default value if no conditions are met


# --- I/O ---
def get_base_path():
    current_username = getpass.getuser()

    # Construct the home directory path and the desired path
    home_directory = os.path.expanduser(f"~{current_username}")
    s5_base = os.path.join(home_directory, "remoteS5", "mausspaun/")

    # Create the dictionary
    path_dict = {'s5': s5_base}
    return path_dict


def generate_filepaths(base_path, mouse_name, day, attempt, part):
    common_file_part = f"rigVideo_mouse-{mouse_name}_day-{day}_attempt-{attempt}_camera-"
    h5_file_pattern = f"_part-{part}_doe-*_rig-5DLC_resnet50_MackenzieJan21shuffle1_700000.h5"
    mp4_file_pattern = f"_part-{part}_doe-*_rig-5.mp4"

    filepaths = find_paths_on_server(base_path, base_path, common_file_part, h5_file_pattern, mp4_file_pattern)

    if not filepaths:
        # If empty check for video in videos_base
        print('Part not found - Checking base folder...')
        base_paths = get_base_path()
        s5_base_h5 = base_paths['s5'] + 'emissions/'
        s5_videos = base_paths['s5'] + 'videos/videos_base/'

        # For cheking original videos
        #s5_videos = '/home/markus/remoteS1/rawdata/video/'
        #mp4_file_pattern = f"_part-{part}_doe-*_rig-5.avi"

        filepaths = find_paths_on_server(s5_videos, s5_base_h5, common_file_part, h5_file_pattern, mp4_file_pattern)

    return filepaths


def find_paths_on_server(video_path, h5_path, common_file_part, h5_file_pattern, mp4_file_pattern):
    filepaths = []
    for i in range(1, 3):
        h5_file_paths = glob.glob(h5_path + common_file_part + str(i) + h5_file_pattern)
        mp4_file_paths = glob.glob(video_path + common_file_part + str(i) + mp4_file_pattern)
        print(video_path + common_file_part + str(i) + mp4_file_pattern)
        if h5_file_paths and mp4_file_paths:
            filepaths.append([mp4_file_paths[0], h5_file_paths[0]])

    return filepaths


def load_action_labels(labels_path, mouse_name, day, attempt, part):
    fname = f"rigVideo_mouse-{mouse_name}_day-{day}_attempt-{attempt}_camera-1_part-{part}_"
    # Check if fname exists in labels_path
    files = os.listdir(labels_path)
    files = [f for f in files if ((fname in f) and (f.endswith('.tsv')))]
    assert len(files) < 2, f"Found multiple files with the same name: {files}"

    if len(files) > 0:  # Found action labels
        # Load the action labels from tsv file
        action_labels = pd.read_csv(os.path.join(labels_path, files[0]), sep='\t', index_col=0)
        return action_labels
    return None


def get_color_from_label(label):
    options = [
        "on joy top", "on joy middle", "slightly pulled", "midway pulled", "pulled", "at base", "grooming", "reaching"
    ]
    colors = ["#ff0000", "#ff8000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#8000ff"]
    return colors[options.index(label)]


# --- Embedding ---
def on_select(self, verts):
    # Extract the first index from the entry fields
    first_index = int(self.first_index_entry.get())

    path = Path(verts)
    ind = np.nonzero(path.contains_points(self.embedding))[0]

    # Reset all colors to white
    self.colors[:] = 0

    # Return if nothing selected
    if len(ind) == 0:
        self.points_embedding.set_array(self.colors)
        self.canvas_embedding.draw_idle()
        return

    # Calculate the center of the selected points
    selected_points = self.embedding[ind]
    center_x = np.mean(selected_points[:, 0])
    center_y = np.mean(selected_points[:, 1])

    # Find the index of the point closest to the center
    distances = np.sqrt((self.embedding[:, 0] - center_x)**2 + (self.embedding[:, 1] - center_y)**2)
    closest_index = np.argmin(distances)

    # Update the colors of the selected points
    self.colors[ind] = 1  # Set to red
    self.points_embedding.set_array(self.colors)

    # Update the canvas to reflect the color changes
    self.canvas_embedding.draw_idle()

    # Update the slider to the corresponding frame
    self.slider.set(closest_index + first_index)

    # Update views if needed (e.g., 2D views or other dependent elements)
    self.update_2d_views()


def get_embedding(self, start_frame=0, end_frame=50):
    camera1_frames = np.array(self.camera1.get_all_frames(start_frame, end_frame))
    camera2_frames = np.array(self.camera2.get_all_frames(start_frame, end_frame))
    print("camera1_frames.shape", camera1_frames.shape)

    # Reshape the frames to be 2D and perform PCA
    camera1_frames = camera1_frames.reshape((camera1_frames.shape[0], -1))
    camera2_frames = camera2_frames.reshape((camera2_frames.shape[0], -1))
    print("camera1_frames.shape", camera1_frames.shape)

    # Perform PCA
    pca_c1 = PCA(n_components=1)
    camera1_embedding = pca_c1.fit_transform(camera1_frames)
    pca_c2 = PCA(n_components=1)
    camera2_embedding = pca_c2.fit_transform(camera2_frames)

    # Concatenate the embeddings
    embedding = np.concatenate((camera1_embedding, camera2_embedding), axis=1)
    print("embedding.shape", embedding.shape)
    return embedding


# --- Fun ---
def add_quotes(self, font):
    # Quotes to display
    self.quotes = self.quotes = [
        "Don't watch the clock; do what it does. Keep going. - Sam Levenson",
        "The future depends on what you do today. - Mahatma Gandhi",
        "You are never too old to set another goal or to dream a new dream. - C.S. Lewis",
        "The only limit to our realization of tomorrow will be our doubts of today. - Franklin D. Roosevelt",
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Don't count the days, make the days count. - Muhammad Ali",
        "The journey of a thousand miles begins with one step. - Lao Tzu",
        "Success is not in what you have, but who you are. - Bo Bennett",
        "Your time is limited, don't waste it living someone else's life. - Steve Jobs",
        "The best way to predict the future is to create it. - Peter Drucker",
        "It does not matter how slowly you go as long as you do not stop. - Confucius",
        "All progress takes place outside the comfort zone. - Michael John Bobak",
        "Success usually comes to those who are too busy to be looking for it. - Henry David Thoreau",
        "Don't be afraid to give up the good to go for the great. - John D. Rockefeller",
        "I find that the harder I work, the more luck I seem to have. - Thomas Jefferson",
        "Success is not just about making money. It's about making a difference. - Unknown",
        "Life is what happens when you're busy making other plans. - John Lennon",
        "You miss 100% of the shots you don't take. - Wayne Gretzky",
        "I am always doing that which I cannot do, in order that I may learn how to do it. - Pablo Picasso",
        "Opportunity is missed by most people because it is dressed in overalls and looks like work. - Thomas Edison",
        "I told my wife she should embrace her mistakes. She gave me a hug.",
    ]

    # Create a frame to hold the quote
    self.quote_frame = tk.Frame(self.root, bg='white')
    self.quote_frame.grid(row=3, column=2, sticky='se')

    # Add a label to display a random quote
    self.quote_label = tk.Label(self.quote_frame,
                                text=random.choice(self.quotes),
                                wraplength=400,
                                justify='center',
                                bg='white',
                                fg='black',
                                font=font)
    self.quote_label.pack(padx=5, pady=5)
