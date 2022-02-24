import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
import pc_util
import pose_utils

parser = argparse.ArgumentParser()
parser.add_argument('--viz', action='store_true', help='Run data visualization.')
parser.add_argument('--gen_data', action='store_true', help='Generate training dataset.')
parser.add_argument('--num_sample', type=int, default=8, help='Number of samples [default: 90000]')
parser.add_argument('--num_vote', type=int, default=1, help='Number of vote per object or part [default: 1]')
parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 50000]')

args = parser.parse_args()

DEFAULT_TYPE_WHITELIST = ["brick", "bunny", "candlestick", "coffee_cup", "gear", "IPAGearShaft", "IPARingScrew", "pepper", "tless_20", "tless_22", "tless_29"]

class pose_object(object):
    ''' Load and parse object data '''
    def __init__(self, data_dir):

        self.pointcloud_dir = os.path.join(data_dir, 'pointcloud')
        self.pose_dir = os.path.join(data_dir, 'pose')
        self.num_samples = args.num_sample
        self.num_vote = args.num_vote
        
    def __len__(self):
        return self.num_samples

    def get_pointcloud(self, idx):
        pc_filename = os.path.join(self.pointcloud_dir, '%d.ply'%(idx))
        print(pc_filename)
        return pose_utils.load_pointcloud(pc_filename)

    def get_label_objects(self, idx): 
        pose_filename = os.path.join(self.pose_dir, '%d.txt'%(idx))
        print(pose_filename)
        return pose_utils.load_label(pose_filename, self.num_vote)
    
def extract_data(data_dir, idx_filename, output_folder, num_point=20000,
    type_whitelist=DEFAULT_TYPE_WHITELIST):
    
    dataset = pose_object(data_dir)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_label_objects(data_idx)

        # Save pointcloud
        pc = dataset.get_pointcloud(data_idx)
        xyz_pc=pc[:,0:3]
        np.savez_compressed(os.path.join(output_folder,'%06d_pc.npz'%(data_idx)), pc=xyz_pc)
        # Save poses and votes
        pose_list = []
        N = pc.shape[0]
        point_object_votes = np.zeros((N,4)) # 1 votes and 1 vote mask 1*3+1 
        point_part_votes = np.zeros((N,4)) # 1 votes and 1 vote mask 1*3+1 
        point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
        indices = np.arange(N)
        for obj in objects:
         
            ## Compute gt votes for object center
         
            object_pc, inds=pose_utils.get_object_points(pc, obj.instance_id)

            # Add pose
            for grp in obj.poses:
                pose = np.zeros((8))
                pose[0:6] = np.array([grp[0], grp[1], grp[2], grp[3], grp[4], grp[5]]) # pose_position
                pose[6] = pose_utils.type2class[obj.classname] # semantic class id
                pose_list.append(pose)           

            # Assign first dimension to indicate it belongs an object
            point_object_votes[inds,0] = 1
            for pose_idx, grp in enumerate(obj.poses):
                pose_position = np.array([grp[0], grp[1], grp[2]])
                # Add the votes (all 0 if the point is not in any object's OBB)
                votes = np.expand_dims(pose_position,0) - object_pc[:,0:3]
                sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
                for i in range(len(sparse_inds)):
                    j = sparse_inds[i]
                    point_object_votes[j, int(pose_idx*3+1):int((pose_idx+1)*3+1)] = votes[i,:]

            ## Compute gt votes for part center

            for part_id in range(0, obj.num_parts):

                part_pc, part_inds=pose_utils.get_part_points(pc, part_id)
                part_center = np.mean(part_pc[:,0:3], axis=0)

                # Assign first dimension to indicate it belongs an object
                point_part_votes[inds,0] = 1
                # Add the votes (all 0 if the point is not in any part's surface)
                votes = np.expand_dims(part_center,0) - part_pc[:,0:3]
                sparse_inds = indices[part_inds] # turn dense True,False inds to sparse number-wise inds
                for i in range(len(sparse_inds)):
                    j = sparse_inds[i]
                    point_part_votes[j, 1:4] = votes[i,:]

        np.savez_compressed(os.path.join(output_folder, '%06d_object_votes.npz'%(data_idx)), point_object_votes = point_object_votes)
        np.savez_compressed(os.path.join(output_folder, '%06d_part_votes.npz'%(data_idx)), point_part_votes = point_part_votes)
        if len(pose_list)==0:
            poses = np.zeros((0,8))
        else:
            poses = np.vstack(pose_list)
        np.save(os.path.join(output_folder, '%06d_pose.npy'%(data_idx)), poses)

    return 0

    
if __name__=='__main__':
    
    if args.viz:
        data_viz(os.path.join(BASE_DIR, 'data'))
        exit()

    if args.gen_data:
        idxs = np.array(range(0,args.num_sample))
        np.random.seed(0)
        np.random.shuffle(idxs)
        np.savetxt(os.path.join(BASE_DIR, 'data', 'train_data_idx.txt'), idxs[:6], fmt='%i')
        np.savetxt(os.path.join(BASE_DIR, 'data', 'val_data_idx.txt'), idxs[6:], fmt='%i')
        
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        extract_data(DATA_DIR, os.path.join(DATA_DIR, 'train_data_idx.txt'),
            output_folder = os.path.join(DATA_DIR, 'train'), num_point=50000)
        extract_data(DATA_DIR, os.path.join(DATA_DIR, 'val_data_idx.txt'),
            output_folder = os.path.join(DATA_DIR, 'val'), num_point=50000)