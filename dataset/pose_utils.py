import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points
import pc_util
import math

type2class={'brick':0, 'bunny':1, 'candlestick':2, 'coffee_cup':3, 'gear':4, 'IPAGearShaft':5,
            'IPARingScrew':6, 'pepper':7, 'tless_20':8, 'tless_22':9, 'tless_29':10}
            
class2type = {type2class[t]:t for t in type2class}

class poseObject(object):
    def __init__(self, line):
        self.poses = []
        data = line.split(' ')
        self.classname = data[0]
        data[1:7] = [float(x) for x in data[1:]]
        pose = [data[1],data[2],data[3],data[4],data[5],data[6]] # x,y,x,rx,ry,rz
        self.poses.append(pose)
        self.instance_id = int(data[7])
        self.num_parts = int(data[8])

def load_pointcloud(pc_filename):
    pointcloud = pc_util.read_xyzrgb_ply(pc_filename)
    return pointcloud

def load_label(pose_filename, num_pose):
    lines = [line.rstrip() for line in open(pose_filename)]
    pose_lines = []
    obj_name = ''
    objects = []
    for line in lines[1:]:
        data = line.split(' ')
        obj = poseObject(line)
        objects.append(obj)

    return objects

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, angle):
    R = rotz(-1*angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    print(box3d_roi_inds)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def get_object_points(pc, instance_id):
    g=pc[:,4]
    inds=g[:]==instance_id
    return pc[inds,:], inds

def get_part_points(pc, part_id):
    b=pc[:,5]
    inds=b[:]==part_id
    return pc[inds,:], inds
