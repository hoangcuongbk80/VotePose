import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
from CGNL import SpatialCGNL

def decode_scores(net, end_points, num_class, num_angle_bin, num_viewpoint):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_object_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center

    rot_6d = net_transposed[:,:,5:11] # (batch_size, num_proposal, 3)
    end_points['rot_6d'] = rot_6d

    width = net_transposed[:,:,11:12] # (batch_size, num_proposal, 1)
    end_points['width'] = width

    quality = net_transposed[:,:,12:13] # (batch_size, num_proposal, 1)
    end_points['quality'] = quality # or grasp quality or grasp score


    angle_scores = net_transposed[:,:,13:13+num_angle_bin]
    angle_residuals_normalized = net_transposed[:,:,13+num_angle_bin:13+num_angle_bin*2]
    end_points['angle_scores'] = angle_scores # Bxnum_proposalxnum_angle_bin
    end_points['angle_residuals_normalized'] = angle_residuals_normalized # Bxnum_proposalxnum_angle_bin (should be -1 to 1)
    end_points['angle_residuals'] = angle_residuals_normalized * (np.pi/num_angle_bin) # Bxnum_proposalxnum_angle_bin

    viewpoint_scores = net_transposed[:,:,13+num_angle_bin*2:13+num_angle_bin*2+num_viewpoint]
    end_points['viewpoint_scores'] = viewpoint_scores

    sem_cls_scores = net_transposed[:,:,13+num_angle_bin*2+num_viewpoint:] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_angle_bin, num_viewpoint, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_angle_bin = num_angle_bin
        self.num_viewpoint = num_viewpoint
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Grasp detection/proposal
        # Objectness-> class (2), center-> residual (3), width-> residual (1), quality(score)-> residual (1)
        # in-plane rotation-> class+residual (num_angle_bin*2), 
        # viewpoint-> class (num_viewpoint)
        self.conv1 = torch.nn.Conv1d(256,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+1+1+num_angle_bin*2+num_viewpoint+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.sa_object = SpatialCGNL(128, int(128 / 2), use_scale=False, groups=4)
        self.sa_part = SpatialCGNL(128, int(128 / 2), use_scale=False, groups=4)

    def forward(self, xyz_object, features_object, xyz_part,  features_part, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz_object, features_object, fps_inds = self.vote_aggregation(xyz_object, features_object)
            sample_inds = fps_inds

            xyz_part, features_part, fps_inds_part = self.vote_aggregation(xyz_part, features_part)
            sample_inds_part = fps_inds_part
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz_object, features_object, _ = self.vote_aggregation(xyz_object, features_object, sample_inds)

            sample_inds_part = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz_part, features_part, _ = self.vote_aggregation(xyz_part, features_part, sample_inds_part)

        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz_object, features_object, _ = self.vote_aggregation(xyz_object, features_object, sample_inds)
            sample_inds_part = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz_part, features_part, _ = self.vote_aggregation(xyz_part, features_part, sample_inds_part)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_object_xyz'] = xyz_object # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        end_points['aggregated_vote_part_xyz'] = xyz_part # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_part_inds'] = sample_inds_part # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal


        # --------- VOTE OBJECT CENTER AND SELF-ATTENTION ---------

        feature_dim = features_object.shape[1]
        batch_size = features_object.shape[0]
        features_object = features_object.contiguous().view(batch_size, feature_dim, 16, 16)
        net = self.sa_object(features_object)
        net = net.contiguous().view(batch_size, feature_dim, self.num_proposal)

        # --------- VOTE PART CENTER AND SELF-ATTENTION ---------

        feature_part_dim = features_part.shape[1]
        features_part = features_part.contiguous().view(batch_size, feature_part_dim, 16, 16)
        net_part = self.sa_part(features_part)
        net_part = net.contiguous().view(batch_size, feature_part_dim, self.num_proposal)

        # --------- OBJECT POSE ESTIMATION ---------

        net = torch.cat((net, net_part), 1)
        net = F.relu(self.bn1(self.conv1(net))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+1+1+num_angle_bin*2+num_viewpoint+self.num_class, num_proposal)

        end_points = decode_scores(net, end_points, self.num_class, self.num_angle_bin, self.num_viewpoint)
        return end_points