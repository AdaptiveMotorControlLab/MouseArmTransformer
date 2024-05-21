import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
import os
import sys

import numpy as np
import cv2
import pandas as pd

from MouseArmTransformer.gui import utils, camera, points
import json
import argparse
from copy import deepcopy

import mplcursors

class App:
    def __init__(self, root, camera1, camera2, triangulation_data, action_labels, base_path, args, font=('Arial', 15)):
        # create the elements as instance variables
        self.camera1 = camera1
        self.camera2 = camera2
        self.root = root
        self.slider = tk.Scale(root, from_=0, to=500, length=600, orient=tk.HORIZONTAL, font=font)
        self.slider.grid(row=1, column=0, columnspan=2)
        self.triangulation_data = triangulation_data
        self.args = args
        self.base_path = base_path
        self.action_labels = action_labels
        self.camera_offset = utils.get_camera_offset(args.mouse_name, args.day)

        self.fig1 = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=root)
        self.canvas1.get_tk_widget().grid(row=0, column=0)
        self.ax1.grid(False)

        self.fig2 = Figure(figsize=(5, 5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=root)
        self.canvas2.get_tk_widget().grid(row=0, column=1)
        self.ax2.grid(False)

        self.fig3d = Figure(figsize=(5, 5), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.ax3d.xaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))  
        self.ax3d.yaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
        self.ax3d.zaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=root)
        self.canvas3d.get_tk_widget().grid(row=0, column=2)

        # Plot the embedding
        self.fig_embedding = Figure(figsize=(5, 5), dpi=100)
        self.ax_embedding = self.fig_embedding.add_subplot(111)
        self.ax_embedding.set_facecolor((0.3, 0.3, 0.3, 1.0))
        self.canvas_embedding = FigureCanvasTkAgg(self.fig_embedding, master=root)
        self.canvas_embedding.get_tk_widget().grid(row=0, column=3)    
        self.lasso = LassoSelector(self.ax_embedding, self.on_select, props={'color': 'white', 'linewidth': 2})

        # Add markers
        self.points1 = []
        self.points2 = []
        self.points3d = []

        frame_no = self.slider.get()
        self.current_frame = self.slider.get()
        self.update_points(frame_no)

        # Add toggle of markers
        self.root.bind('<Control_L>', self.toggle_marker_visibility)  # CTRL key pressed
        self.root.bind('<KeyRelease-Control_L>', self.toggle_marker_visibility)  # CTRL key released

        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(root, bg='white')
        self.button_frame.grid(row=2, columnspan=2, pady=10)

        # Add a button for starting the triangulation process
        self.button = tk.Button(self.button_frame, text="Triangulate points", command=self.update, font=font)
        self.button.pack(side=tk.LEFT, padx=5)

        # Add a button for saving the points
        self.save_button = tk.Button(self.button_frame, text="Save points", command=self.save, font=font)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.frame_entry = tk.Entry(self.button_frame, width=5, font=font)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind('<Return>', self.load_frame)

        # Create a frame to hold the entries
        self.entry_frame = tk.Frame(root, bg='white')
        self.entry_frame.grid(row=1, column=3, pady=10)

        # Add entry for the first index
        self.first_index_label = tk.Label(self.entry_frame, text="First index:", font=font)
        self.first_index_label.pack(side=tk.LEFT, padx=5)
        self.first_index_entry = tk.Entry(self.entry_frame, width=5, font=font)
        self.first_index_entry.pack(side=tk.LEFT, padx=5)
        self.first_index_entry.insert(0, "0")  # Default value
        self.first_index_entry.bind('<Return>', self.draw_embedding)

        # Add entry for the last index
        self.last_index_label = tk.Label(self.entry_frame, text="Last index:", font=font)
        self.last_index_label.pack(side=tk.LEFT, padx=5)
        self.last_index_entry = tk.Entry(self.entry_frame, width=5, font=font)
        self.last_index_entry.pack(side=tk.LEFT, padx=5)
        self.last_index_entry.insert(0, "50")  # Default value
        self.last_index_entry.bind('<Return>', self.draw_embedding)

        # Call the draw_embedding function to update the embedding
        self.draw_embedding(None)

        # Add quotes
        utils.add_quotes(self, font)

        # Update the views when the slider value changes
        self.slider.configure(command=lambda _: self.update_2d_views())

        # Style the layout
        root.configure(bg='white')
        self.slider.configure(bg='white', fg='black', troughcolor='#cccccc', sliderrelief='flat', highlightbackground='white')
        self.button.configure(bg='#007BFF', fg='white', activebackground='#0069d9', activeforeground='white')
        self.save_button.configure(bg='#007BFF', fg='white', activebackground='#0069d9', activeforeground='white')

        # More styling for fitting the GUI to the screen
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)
        root.grid_columnconfigure(3, weight=1)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.canvas2.get_tk_widget().grid(row=0, column=1, sticky='nsew')
        self.canvas3d.get_tk_widget().grid(row=0, column=2, sticky='nsew')
        self.canvas_embedding.get_tk_widget().grid(row=0, column=3, sticky='nsew')

        self.update_2d_views()

    def on_select(self, verts):
        utils.on_select(self, verts)

    def draw_embedding(self, event):
        # Extract the values from the entry fields
        first_index = int(self.first_index_entry.get())
        last_index = int(self.last_index_entry.get())

        # Get embedding and update the plot
        self.embedding = utils.get_embedding(self, first_index, last_index)

        # Clear the canvas
        self.ax_embedding.clear()

        self.colors = np.zeros(len(self.embedding))  # All black to start
        self.cmap = plt.cm.colors.ListedColormap(['white', 'red'])
        self.norm = plt.cm.colors.Normalize(vmin=0, vmax=1)

        # Define edge colors based on labeled frames
        labeled_cam1, labeled_cam2, _, _ = utils.get_labeled_frames(self)
        labeled_frames = (labeled_cam1 & labeled_cam2[0:len(labeled_cam1)])
        self.edge_colors = []
        for cam1_label, cam2_label, frame_label in zip(labeled_cam1, labeled_cam2[0:len(labeled_cam1)], labeled_frames):
            if frame_label:
                color = 'red'
            elif cam1_label:
                color = 'green'
            elif cam2_label:
                color = 'blue'
            else:
                color = 'black'
            self.edge_colors.append(color)     
        self.points_embedding = self.ax_embedding.scatter(self.embedding[:, 0], self.embedding[:, 1], edgecolors=self.edge_colors[0:last_index],
                                                          c=self.colors, cmap=self.cmap, norm=self.norm)

        # Redraw the canvas to apply the changes
        self.canvas_embedding.draw()
        
    def load_frame(self, event=None):
        frame_no = int(self.frame_entry.get())
        self.update_points(frame_no)
        # Update the canvas
        self.canvas1.draw()
        self.canvas2.draw()
        # Overwrite current frame in this case
        self.current_frame = self.slider.get()
        self.update_current_frame(frame_no) # This overwrites the current frame with frame_no
        self.current_frame = self.slider.get()

    def update_views(self):
        self.update_2d_views()
        self.update_3d_views()

    def update_2d_views(self):
        # Get the frame number from the slider
        frame_no = self.slider.get()

        # Read the frames from the videos
        self.camera1.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.camera2.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - self.camera_offset)
        _, frame1 = self.camera1.cap.read()
        _, frame2 = self.camera2.cap.read()

        # Convert the frames to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Draw the frames
        self.ax1.imshow(frame1)
        self.ax2.imshow(frame2)

        # Add action label
        if self.action_labels is not None:
            self.ax1.set_title(self.action_labels[frame_no], fontsize=10)

        # Update points on the frames
        self.update_current_frame(frame_no)
        self.update_points(frame_no)

        # Update the text of the color to reflect label status
        self.quote_label.configure(fg=self.edge_colors[frame_no])

        # Update the canvas
        self.canvas1.draw()
        self.canvas2.draw()

    def update_3d_views(self):
        # Clear the 3D view and plot the new points
        self.ax3d.clear()
        for bodypart, p in self.points3d.items():
            # With color
            color = utils.bodypart_to_color(bodypart)
            sc = self.ax3d.scatter(*p, label=bodypart, c=color)
    
            cursor = mplcursors.cursor(sc, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

        # Draw connections in 3D view
        if len(self.points3d) > 0:
            utils.draw_connections(self.ax3d, self.points3d)
        self.canvas3d.draw()

    def update_points(self, frame_no):
        # Initialize lists for new points
        self.bodyparts = self.camera1.points.columns.get_level_values(1)[::3]
        
        # Update points on the frames
        for i, bodypart in enumerate(self.bodyparts):
            # The 'x' and 'y' positions are column 0 and 1, and 'likelihood' is column 2.
            x1, y1, likelihood1 = self.camera1.points.loc[frame_no, (slice(None), bodypart)]
            x2, y2, likelihood2 = self.camera2.points.loc[frame_no, (slice(None), bodypart)]


            if len(self.points1) >= len(self.bodyparts): 
                self.points1[i].point.set_data([x1], [y1])
                self.points1[i].x = x1
                self.points1[i].y = y1
            else: 
                self.points1.append(points.DraggablePoint(self.ax1, x1, y1, bodypart=bodypart))
                
            if len(self.points2) >= len(self.bodyparts):
                self.points2[i].point.set_data([x2], [y2])
                self.points2[i].x = x2
                self.points2[i].y = y2
            else: 
                self.points2.append(points.DraggablePoint(self.ax2, x2, y2, bodypart=bodypart))
        
        self.current_frame = frame_no

    def update_current_frame(self, frame_no):
        # Initialize lists for new points
        self.bodyparts = self.camera1.points.columns.get_level_values(1)[::3]

        for i, bodypart in enumerate(self.bodyparts):
            if len(self.points1) >= len(self.bodyparts): 
                _, _, current_likelihood1 = self.camera1.points.loc[self.current_frame, (slice(None), bodypart)]
                self.camera1.points.loc[self.current_frame, (slice(None), bodypart)] = [np.squeeze(self.points1[i].x), np.squeeze(self.points1[i].y), current_likelihood1]
            if len(self.points2) >= len(self.bodyparts):
                _, _, current_likelihood2 = self.camera2.points.loc[self.current_frame, (slice(None), bodypart)]
                self.camera2.points.loc[self.current_frame, (slice(None), bodypart)] = [np.squeeze(self.points2[i].x), np.squeeze(self.points2[i].y), current_likelihood2]
        self.current_frame = frame_no

    def triangulate(self):
        # First update points so all updates are reflected in the triangulation
        frame_no = self.slider.get()
        self.update_current_frame(frame_no)

        x_values = self.camera1.points.xs('x', level='coords', axis=1).values
        y_values = self.camera1.points.xs('y', level='coords', axis=1).values
        points1 = np.dstack((x_values, y_values))

        x_values = self.camera2.points.xs('x', level='coords', axis=1).values
        y_values = self.camera2.points.xs('y', level='coords', axis=1).values
        points2 = np.dstack((x_values, y_values))

        # Cut to 500 points TODO ALIGN POINTS
        points1 = points1[:500]
        points2 = points2[:500]

        self.all_points3d = utils.triangulate_points_from_rig(points1, points2, self.triangulation_data)

        # Get bodyparts
        assert [p1.bodypart for p1 in self.points1] == [p2.bodypart for p2 in self.points2]
        self.bodyparts = [p1.bodypart for p1 in self.points1]

    def update(self):
        # Make sure everything is up to date
        self.triangulate()
        frame_no = self.slider.get()

        self.points3d = self.all_points3d[frame_no]

        # Align to mujuco
        points_3d_dict = {key: np.array(self.points3d[i]) for i, key in enumerate(self.bodyparts)}
        self.points3d = utils.align_to_mujoco(points_3d_dict)

        # Update the 3D view to include the triangulated points
        self.update_3d_views()

    def save(self):
        # Make sure everything is up to date
        self.triangulate()

        # Save the triangulated points
        utils.save(self)

    def toggle_marker_visibility(self, event=None):
        for point in self.points1 + self.points2:
            point.point.set_visible(not point.point.get_visible())  # Toggle visibility
        self.canvas1.draw()
        self.canvas2.draw()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_name", default="HoneyBee")
    parser.add_argument("--day", default="77")
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--part", default="0")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if int(args.part) != 0:
        raise ValueError("Part has to be 0 as video files are not synchronized, i.e. part-5-camera-1 shows a different timerange than part-5-camera-2")

    # Get paths
    base_path = utils.get_base_path()['s5']
    original_filepaths = utils.generate_filepaths(base_path + 'videos/videos_dlc2/', args.mouse_name, args.day, args.attempt, args.part)

    # Check if session has been labeled before
    output_dir = os.path.join(base_path, 'labeled')
    fname = f"rigVideo_mouse-{args.mouse_name}_day-{args.day}_attempt-{args.attempt}_part-{args.part}_"

    # Check if session exists in output_dir
    try:
        files = os.listdir(output_dir)
        files = [f for f in files if fname in f]
        if len(files) > 0: # Previous version exists
            filepaths = deepcopy(original_filepaths)
            # Find file with cam1.csv and cam2.csv suffixes
            cam1_file = [f for f in files if 'cam1.csv' in f][0]
            cam2_file = [f for f in files if 'cam2.csv' in f][0]
            print('Previous labeling session found, loading {} & {}'.format(cam1_file, cam2_file))
            filepaths[0][1] = os.path.join(output_dir, cam1_file)
            filepaths[1][1] = os.path.join(output_dir, cam2_file)
        else:
            filepaths = original_filepaths
        print('Original Filepaths:')
        print(original_filepaths)
        print('Loading Files...')
        print(filepaths)
        camera1 = camera.Camera(filepaths[0][0], filepaths[0][1], original_filepaths[0][1])
        camera2 = camera.Camera(filepaths[1][0], filepaths[1][1], original_filepaths[1][1], camera_offset=utils.get_camera_offset(args.mouse_name, args.day))
        
    except FileNotFoundError:
        print("No files found for the given mouse name, day, attempt, and part.")
        print('Did you remember to mount the network drive?')
        sys.exit()

    # Load action labels
    action_labels_path = base_path + 'action_labels/'
    action_labels = utils.load_action_labels(action_labels_path, args.mouse_name, args.day, args.attempt, args.part)
    if action_labels is None:
        print("No action labels found for the given mouse name, day, attempt, and part.")
    else:
        action_labels = action_labels['annotation']

    alignment_points = base_path + 'calibration/' + "honeybee_reference_points.jl"
    calibration_folder = base_path + 'calibration/' + "data/2018_mouse_reaching_rig5/calibration/"

    rig, cpoints_1, cpoints_2, cam1_align, cam2_align = utils.load_rig_and_calibration_points(alignment_points, calibration_folder)

    root = tk.Tk()
    root.wm_title("3D Point Labeler")
    app = App(root, camera1, camera2, (rig, cpoints_1, cpoints_2, cam1_align, cam2_align), action_labels, base_path, args)
    tk.mainloop()