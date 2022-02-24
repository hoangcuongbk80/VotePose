import torch
import torch.nn as nn
import numpy as np
import sys
import os
import tools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.06
GT_VOTE_FACTOR = 1
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_vote_object_loss(end_points):

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1]
    vote_object_xyz = end_points['vote_object_xyz']
    seed_inds = end_points['seed_inds'].long()

    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,GT_VOTE_FACTOR)

    # Compute the min of min of distance
    vote_object_xyz_reshape = vote_object_xyz.view(batch_size*num_seed, -1, 3)
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3)
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_object_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_vote_part_loss(end_points):

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1]
    vote_part_xyz = end_points['vote_part_xyz'] 
    seed_inds = end_points['seed_inds'].long()

    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_part_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,GT_VOTE_FACTOR)

    # Compute the min of min of distance
    vote_part_xyz_reshape = vote_part_xyz.view(batch_size*num_seed, -1, 3)
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3)
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_part_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) 
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_part_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_part_loss


def compute_objectness_loss(end_points):

    aggregated_vote_object_xyz = end_points['aggregated_vote_object_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_object_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_object_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_rot_loss(end_points, config):

    pred_rot = end_points['rot_6d']
    predict_rot = pred_rot.reshape(pred_rot.shape[0]*pred_rot.shape[1], pred_rot.shape[2])
    predict_rmat = tools.compute_rotation_matrix_from_ortho6d(predict_rot)
    predict_rmat = predict_rmat.reshape(pred_rot.shape[0], pred_rot.shape[1], 9)

    gt_rot = end_points['rot_label'][:,:,0:3]
    gt_rott = gt_rot.reshape(gt_rot.shape[0]*gt_rot.shape[1], gt_rot.shape[2])
    gt_rmat = tools.compute_rotation_matrix_from_euler(gt_rott)
    gt_rmat = gt_rmat.reshape(gt_rot.shape[0], gt_rot.shape[1], 9)

    dist1, ind1, dist2, _ = nn_distance(predict_rmat, gt_rmat)
    object_label_mask = end_points['object_label_mask']
    rot_loss = torch.sum(dist2*object_label_mask)/(torch.sum(object_label_mask)+1e-6)
    print('rot_loss', rot_loss)

    return rot_loss

def compute_location_loss(end_points, config):

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    objectness_label = end_points['objectness_label'].float()

    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)
    object_label_mask = end_points['object_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*object_label_mask)/(torch.sum(object_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2
    
    return center_loss

def compute_sem_cls_loss(end_points, config):

    object_assignment = end_points['object_assignment']
    objectness_label = end_points['objectness_label'].float()
   
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return sem_cls_loss

def get_loss(end_points, config):

    # Vote object loss
    vote_object_loss = compute_vote_object_loss(end_points)
    end_points['vote_loss'] = vote_object_loss

    # Vote part loss
    vote_part_loss = compute_vote_part_loss(end_points)
    end_points['vote_part_loss'] = vote_part_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # location loss
    loc_loss = compute_location_loss(end_points, config)

    # rotation loss
    rot_loss = compute_rot_loss(end_points, config)

    # semantic loss
    sem_loss = compute_sem_cls_loss(end_points, config)

    # Final loss function
    loss = 0.5*vote_object_loss + 0.5*vote_part_loss + loc_loss + 0.1*rot_loss + 0.5*objectness_loss + 0.1*sem_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
