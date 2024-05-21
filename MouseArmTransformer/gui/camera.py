import cv2
import pandas as pd
import numpy as np

class Camera:
    def __init__(self, video_path, dlc_path, original_dlc_path, camera_offset=None):
        self.cap = cv2.VideoCapture(video_path)
        # Check if HDF or CSV
        if dlc_path.endswith('.csv'):
            self.points = pd.read_csv(dlc_path, header=[0,1,2], index_col=0)
        else:
            self.points = pd.read_hdf(dlc_path)        
        self.original_points = pd.read_hdf(original_dlc_path)

        # Shift camera points to align camera1 with camera2, which has an offset of 3-5 frames depending on the session
        if camera_offset is not None:
            print('Shifting markers by {} frames'.format(camera_offset))
            self.original_points = self.original_points.shift(+camera_offset)

            # h5 are the original files, csv are the labeled ones, so we make sure to only shift when we load original files
            if ('.h5' in dlc_path): 
                self.points = self.points.shift(+camera_offset)
            
            # Until we fix the labeled files we also need to shift them | Uncomment below and resave all previously labeled files, should only be run once
            # self.points = self.points.shift(+camera_offset)

        # Force same precision to make comparison between csv and hdf possible
        self.points = self.points.astype('float32')
        self.original_points = self.original_points.astype('float32')

    def get_frame(self, frame_no):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        _, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_all_frames(self, start_frame=0, end_frame=50):
        frames = []
        for i in range(start_frame, end_frame):
            frames.append(self.get_frame(i))
        return frames