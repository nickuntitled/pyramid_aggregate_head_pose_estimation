# The utils.py is from Ruiz's HopeNet source code.
# It is from this repository
# https://github.com/natanielruiz/deep-head-pose
import numpy as np
import scipy.io as sio
import cv2
from math import cos, sin

def get_ypr_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pre_pose_params = mat['Pose_Para'][0]
    pose_params = pre_pose_params[:3]
    return pose_params

# Get 3D landmarks from AFLW2000
def get_pt3d_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt3d_68']
    return pt2d

# Get 2D landmarks
def get_pt2d_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img