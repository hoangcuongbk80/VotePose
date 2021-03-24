import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class poseDatasetConfig(object):
    def __init__(self):
        self.num_class = 10
        self.num_angle_bin = 12
        self.num_viewpoint = 36

        self.type2class={'brick':0, 'bunny':1, 'candlestick':2, 'coffee_cup':3, 'gear':4, 'IPAGearShaft':5,
                        'IPARingScrew':6, 'pepper':7, 'tless_20':8, 'tless_22':9, 'tless_22-tless_29':10}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'brick':0, 'bunny':1, 'candlestick':2, 'coffee_cup':3, 'gear':4, 'IPAGearShaft':5,
                                'IPARingScrew':6, 'pepper':7, 'tless_20':8, 'tless_22':9, 'tless_29':10}
    
    def angle2class(self, angle):
        
        num_class = self.num_angle_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):

        num_class = self.num_angle_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2pose(self, sem_cls, center, viewpoint_class, angle_class, angle_residual, quality, width):
        angle = self.class2angle(angle_class, angle_residual) * 180/np.pi
        object_name = self.class2type[int(sem_cls)]
        pose = []
        pose.append(object_name)
        pose.append(center[0])
        pose.append(center[1])
        pose.append(center[2])
        pose.append(viewpoint_class)
        pose.append(angle)
        pose.append(quality)
        pose.append(width)
        return pose
