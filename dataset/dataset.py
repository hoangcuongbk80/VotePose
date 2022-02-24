import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import pose_utils
from model_util import poseDatasetConfig

DC = poseDatasetConfig()
MAX_NUM_POSE = 64
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5])

class poseVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, augment=False, scan_idx_list=None):

        assert(num_points<=50000)
        self.data_path = os.path.join(ROOT_DIR, 'dataset/data/%s'%(split_set))

        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):

        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] 
        poses = np.load(os.path.join(self.data_path, scan_name)+'_pose.npy')
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_object_votes.npz')['point_object_votes'] 
        point_part_votes = np.load(os.path.join(self.data_path, scan_name)+'_part_votes.npz')['point_part_votes']

        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        # ------------------------------- LABELS ------------------------------
        label_mask = np.zeros((MAX_NUM_POSE))
        label_mask[0:poses.shape[0]] = 1

        target_poses_mask = label_mask 
        target_poses = np.zeros((MAX_NUM_POSE, 6))
        for i in range(poses.shape[0]):
            pose = poses[i]
            target_pose = pose[0:6]
            target_poses[i,:] = target_pose

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        point_part_votes_mask = point_part_votes[choices,0]
        point_part_votes = point_part_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_part_label'] = point_part_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)

        ret_dict['center_label'] = target_poses.astype(np.float32)[:,0:3]
        ret_dict['rot_label'] = target_poses.astype(np.float32)[:,3:6]

        target_poses_semcls = np.zeros((MAX_NUM_POSE))
        target_poses_semcls[0:poses.shape[0]] = poses[:,-1]
        ret_dict['sem_cls_label'] = target_poses_semcls.astype(np.int64)
        ret_dict['object_label_mask'] = target_poses_mask.astype(np.float32)
        
        return ret_dict