import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import tools

DUMP_CONF_THRESH = 0.5 # Dump poses with object prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def dump_results(end_points, dump_dir, config, inference_switch=False):

    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)
    if 'vote_object_xyz' in end_points:
        aggregated_vote_object_xyz = end_points['aggregated_vote_object_xyz'].detach().cpu().numpy()
        vote_object_xyz = end_points['vote_object_xyz'].detach().cpu().numpy() # (B,num_seed,3)
        aggregated_vote_object_xyz = end_points['aggregated_vote_object_xyz'].detach().cpu().numpy()
    objectness_scores = end_points['objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['center'].detach().cpu().numpy() # (B,K,3)
    pred_rot = end_points['rot_6d'] # (B,K,3)
    predict_rot = pred_rot.reshape(pred_rot.shape[0]*pred_rot.shape[1], pred_rot.shape[2])
    predict_rmat = tools.compute_rotation_matrix_from_ortho6d(predict_rot)
    predict_rmat = predict_rmat.reshape(pred_rot.shape[0], pred_rot.shape[1], 9)

    # OTHERS
    #pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    save_pose_file = os.path.join(dump_dir, 'pred_poses.txt')
    f = open(save_pose_file, "w")

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        if 'vote_object_xyz' in end_points:
            pc_util.write_ply(end_points['vote_object_xyz'][i,:,:], os.path.join(dump_dir, '%06d_vgen_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_object_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
            pc_util.write_ply(aggregated_vote_object_xyz[i,:,:], os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))

        # Dump predicted poses
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            for j in range(num_proposal):
                loc = pred_center[i,j,0:3]
                for ite in loc:
                    str_num = '{:.6f}'.format(ite)
                    f.write(str_num)
                    f.write(' ')
                f.write("\n")

            for j in range(num_proposal):
                rmat = predict_rmat[i,j,0:9]
                for ind, ite in enumerate(rmat):
                    str_num = '{:.6f}'.format(ite)
                    f.write(str_num)
                    if (ind+1)%3==0:
                        f.write('\n')
                    else:
                        f.write(' ')
    f.close()