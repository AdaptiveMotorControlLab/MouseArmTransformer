import numpy as np

import tkinter as tk
from tkinter import messagebox
from MouseArmTransformer.gui import utils

class DraggablePoint:
    """ Represents a 2D point that can be dragged with the mouse. """
    active_point = None

    def __init__(self, ax, x, y, bodypart=None):
        self.x = x
        self.y = y
        self.press = None
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.bodypart = bodypart
        self.background = None

        color = utils.bodypart_to_color(self.bodypart)
        self.point = self.plot(x, y, 'o', color=color, markersize=2.5, picker=3)

        self.cidpress = self.canvas.mpl_connect(
            "button_press_event", self.on_press)
        self.cidrelease = self.canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = self.canvas.mpl_connect(
            "motion_notify_event", self.on_motion)

    def plot(self, x, y, *args, **kwargs):
        point, = self.ax.plot(x, y, *args, **kwargs)
        return point

    def on_press(self, event):
        """ Called when the left mouse button is pressed. """
        if event.inaxes != self.ax: return
        contains, attrd = self.point.contains(event)

        # If the right button (button 3) is pressed
        if event.button == 3:
            if contains and self.bodypart is not None:
                messagebox.showinfo('Information', 
                                    f'Body part: {self.bodypart}\n'
                                    f'x-coordinate: {np.squeeze(self.x):.3f}\n'
                                    f'y-coordinate: {np.squeeze(self.y):.3f}')
                return

        if event.button == 1:
            if not contains: return

            # If there's an active point and it's not this one, don't do anything
            if DraggablePoint.active_point is not None and DraggablePoint.active_point != self:
                return
            
            self.press = (self.point.get_data(), event.xdata, event.ydata)

            # Draw everything but the draggable point and store the pixel buffer
            self.point.set_animated(True)
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.point.axes.bbox)

            # Set this point as the active one
            DraggablePoint.active_point = self

    def on_motion(self, event):
        """ Called when the mouse is moved. """
        if self.press is None: return
        if event.inaxes != self.ax: return
        dx = event.xdata - self.press[1]
        dy = event.ydata - self.press[2]
        self.point.set_data([self.x+dx], [self.y+dy])
        
        # Restore the pixel buffer, redraw the point, and blit just the redrawn area
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.point)
        self.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        """ Called when the left mouse button is released. """
        self.press = None
        
        # Turn off the point animation property and force a full redraw of the figure
        self.point.set_animated(False)
        # self.background = None
        # self.canvas.draw()
        # Or blit everything
        self.canvas.blit(self.ax.bbox)

        # Update the point coordinates
        self.x, self.y = self.point.get_data()

        if DraggablePoint.active_point == self:
            DraggablePoint.active_point = None