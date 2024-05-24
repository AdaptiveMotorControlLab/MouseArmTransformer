# Steps to get the GUI running:

```bash
git clone https://github.com/AdaptiveMotorControlLab/mouse-arm.git
cd mouse-arm
./setup.sh
```
This will create the mausspaun conda environment, we need some additional dependencies which we can add to the setup but for now you can install them manually via:

```bash
conda install -c anaconda tk
pip install mplcursors
pip install tables
```

For running the GUI, first mount the data partition of server 5:
```bash
mkdir ~/remoteS5
sshfs user@128.178.84.37:/data/ /home/user/remoteS5/
```

Then to start the GUI use:

```bash
cd mausspaun/visualization/gui
python app.py
```
This will start the GUI with the default session (Honeybee, day 77, attempt 1, part 0). For other sessions use one of the following:
```bash
python app.py --mouse_name Jaguar --day 19
python app.py --mouse_name HoneyBee --day 81
python app.py --mouse_name HoneyBee --day 82
python app.py --mouse_name HoneyBee --day 83
python app.py --mouse_name HoneyBee --day 84
python app.py --mouse_name HoneyBee --day 86
```

If you want to label into the combined pool of all labeled frames, use the following command:
```bash
python app.py --mouse_name HoneyBee --day 77 --user combined
```
otherwise the default user is the username of the computer.


# Usage

The slider is used to step through frames of the video, which adapts both camera views. Adjust the the 2D markers in both camera frames to the correct locations and then hit 'Triangulate Points' to see the result in 3D and 'Save points' to save all labeled frames back to file. 

The textbox next to 'Save points' copies previously labeled frames to the currently selected one, e.g. if the user labeled frame 7 then steps to frame 8 and wants to reuse the points from frame 7 (instead of the original ones from frame 8), the user would put 7 into the textbox (while on frame 8) and hit Enter. 

The panel on the right side can be used to label distinct frames by selecting them in PCA space. The textboxes below the panel specify the start and end indices of the frames which are used to calculate the PCA. For faster startup this is set to 0 and 50, but its recommended to set it to 0 and 500 and then hit Enter. This will calculate the PCA for the first 500 frames which can be selected by selecting points within the plot. The frame which is subsequently labeled is the cluster center of the selected points. Previously labeled frames are highlighted with different edge colors. If the edge is in green only Camera 1 has been labeled, if it is shown in blue then only Camera 2 has been labeled. If both cameras have been labeled the edge is shown in red. 

When closing and reopening, previously labeled frames will be reinstated directly from the files on the server (only if they have been saved in the previous session.)
