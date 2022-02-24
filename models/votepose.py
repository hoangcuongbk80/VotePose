import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss

class votepose(nn.Module):

    def __init__(self, num_class, input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        self.vgen = VotingModule(self.vote_factor, 256)
        self.pnet = ProposalModule(num_class, num_proposal, sampling)

    def forward(self, inputs):

        # --------- Backbone point feature learning ---------
        
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]  
        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        # ---------------- Deep Hough voting ----------------
        
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']   
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
          
        # --------- Vote object instance center ----------        
        
        xyz_object, features_object = self.vgen(xyz, features)
        features_norm = torch.norm(features_object, p=2, dim=1)
        features_object = features_object.div(features_norm.unsqueeze(1))
        end_points['vote_object_xyz'] = xyz_object
        end_points['vote_object_features'] = features_object

        # ----------- Vote object part center -------------        
        
        xyz_part, features_part = self.vgen(xyz, features)
        features_norm = torch.norm(features_part, p=2, dim=1)
        features_part = features_part.div(features_norm.unsqueeze(1))
        end_points['vote_part_xyz'] = xyz_part
        end_points['vote_part_features'] = features_part

        # --------- Proposal, self-attention and pose estimation ----------

        end_points = self.pnet(xyz_object, features_object, xyz_part,  features_part, end_points)

        return end_points