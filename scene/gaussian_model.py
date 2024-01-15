#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from glob import glob
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import random

def read_depth_image_paths(depth_path):
    if os.path.exists(depth_path):
        left_depth_images = sorted(glob(os.path.join(depth_path,"image_02","*.png")))
        right_depth_images = sorted(glob(os.path.join(depth_path,"image_03","*.png")))
        depth_image_paths = {'left': left_depth_images, 'right': right_depth_images}
        return depth_image_paths

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float64) / 256.
    depth[depth_png == 0] = -1.     #   Set
    return depth


def depth_to_point_cloud(depth_image, intrinsics):
    """
    Convert a depth image to a 3D point cloud using camera intrinsics.

    Parameters:
    - depth_image: 2D numpy array representing the depth values.
    - intrinsics: Camera intrinsic parameters [fx, fy, cx, cy].

    Returns:
    - points_3d: 2D numpy array of shape (num_points, 3) representing 3D coordinates.
    """
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]

    # Mask out missing depth values (-1)
    valid_mask = (depth_image != -1)

    # Get the pixel coordinates only for valid depth values
    u, v = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
    u = u[valid_mask]
    v = v[valid_mask]

    # Convert valid depth values to 3D coordinates
    X = (u - cx) * depth_image[valid_mask] / fx
    Y = (v - cy) * depth_image[valid_mask] / fy
    Z = depth_image[valid_mask]

    # Stack X, Y, Z to get the 3D point cloud
    points_3d = np.stack([X, Y, Z], axis=-1)

    return points_3d


def synchronize_depth_and_gt_depth(depth_image_paths,gt_depth_image_paths, offset):
    _, img_name = os.path.split(gt_depth_image_paths['left'][0])
    img_base = img_name.split('.')[0]
    present_ids = []
    for i,img_name in enumerate(depth_image_paths['left']):
        if img_base in img_name:
            present_ids.append(i)
    
    num_depth_images = len(depth_image_paths['left'])
    synchronized_left_depth_images = depth_image_paths['left'][offset:num_depth_images-offset]
    synchronized_right_depth_images = depth_image_paths['right'][offset:num_depth_images-offset]
    synchronized_depth_image_paths = {'left': synchronized_left_depth_images, 'right':synchronized_right_depth_images}
    return synchronized_depth_image_paths
    
def synchronize_depth_and_scene_info(scene_info, depth_path, gt_depth_path, offset):

    # synchronized_depth_image_paths = synchronize_depth_and_gt_depth(depth_image_paths, gt_depth_image_paths, offset)

    train_depth_image_paths, test_depth_image_paths = {'left': [], 'right': []}, {'left': [], 'right': []}
    train_gt_depth_image_paths, test_gt_depth_image_paths = {'left': [], 'right': []}, {'left': [], 'right': []}

    train_camera_image_names = [cam.image_name for cam in scene_info.train_cameras]
    test_camera_image_names = [cam.image_name for cam in scene_info.test_cameras]

    left_depth_image_paths = sorted(glob(os.path.join(depth_path,'image_02','*.png')))
    right_depth_image_paths = sorted(glob(os.path.join(depth_path,'image_03','*.png')))

    gt_left_depth_image_paths = sorted(glob(os.path.join(gt_depth_path,'image_02','*.png')))
    gt_right_depth_image_paths = sorted(glob(os.path.join(gt_depth_path,'image_03','*.png')))

    start_index, stop_index = 0,0
    indices = []
    for img_path in gt_left_depth_image_paths:
        img_name = os.path.split(img_path)[1].split('.')[0]
        img_id = int(img_name)
        indices.append(img_id)
    start_index, stop_index = min(indices), max(indices)
    

    assert(len(left_depth_image_paths) == len(right_depth_image_paths))
    num_depth_images = len(left_depth_image_paths)
    synchronized_left_depth_images = left_depth_image_paths[offset:num_depth_images-offset]
    synchronized_right_depth_images = right_depth_image_paths[offset:num_depth_images-offset]

    assert(len(synchronized_left_depth_images) == len(gt_left_depth_image_paths) )
    assert(len(synchronized_right_depth_images) == len(gt_right_depth_image_paths) )

    for image_name in train_camera_image_names:
        
        cam_id = image_name.split('_')[0]
        img_base = image_name.split('_')[1]
        img_id = int(image_name.split('_')[1])
        if ( (img_id >=start_index) and (img_id <=stop_index)):

            if cam_id == '02':
                depth_img_path = os.path.join(depth_path,'image_02',image_name+".png")
                gt_depth_img_path = os.path.join(gt_depth_path,'image_02',img_base+".png")
                train_depth_image_paths['left'].append(depth_img_path)
                train_gt_depth_image_paths['left'].append(gt_depth_img_path)
            elif cam_id == '03':
                depth_img_path = os.path.join(depth_path,'image_03',image_name+".png")
                gt_depth_img_path = os.path.join(gt_depth_path,'image_03',img_base+".png")
                train_depth_image_paths['right'].append(depth_img_path)
                train_gt_depth_image_paths['right'].append(gt_depth_img_path)


    for image_name in test_camera_image_names:
        
        cam_id = image_name.split('_')[0]
        img_base = image_name.split('_')[1]
        img_id = int(image_name.split('_')[1])
        if ( (img_id >=start_index) and (img_id <=stop_index)):

            if cam_id == '02':
                depth_img_path = os.path.join(depth_path,'image_02',image_name+".png")
                gt_depth_img_path = os.path.join(gt_depth_path,'image_02',img_base+".png")
                test_depth_image_paths['left'].append(depth_img_path)
                test_gt_depth_image_paths['left'].append(gt_depth_img_path)
            elif cam_id == '03':
                depth_img_path = os.path.join(depth_path,'image_03',image_name+".png")
                gt_depth_img_path = os.path.join(gt_depth_path,'image_03',img_base+".png")
                test_depth_image_paths['right'].append(depth_img_path)
                test_gt_depth_image_paths['right'].append(gt_depth_img_path)

    train_cameras, test_cameras = [], []
    for cam in scene_info.train_cameras:
        cam_id, img_id = cam.image_name.split('_')[0], int(cam.image_name.split('_')[1])
        if (img_id >=start_index) and (img_id <=stop_index):
            train_cameras.append(deepcopy(cam))

    for cam in scene_info.test_cameras:
        cam_id, img_id = cam.image_name.split('_')[0], int(cam.image_name.split('_')[1])
        if (img_id >=start_index) and (img_id <=stop_index):
            test_cameras.append(deepcopy(cam))
        # print(cam_id, img_id)
        # input()
    # scene_info.train_cameras = train_cameras
    # scene_info.test_cameras = test_cameras
    return train_cameras, test_cameras, train_depth_image_paths, test_depth_image_paths, train_gt_depth_image_paths, test_gt_depth_image_paths

def get_depth_scale(depth_image, gt_depth_image):
    """
        Input:
            depth_image: HxW np.array, Depth image predicted from a pre-trained network.
            gt_depth_image: HxW np.array, Ground truth depth image taken from lidar data.
        
    """

    #   1) Find valid depth indices
    valid_indices = gt_depth_image!=-1
    valid_gt_depth_values = gt_depth_image[valid_indices]
    corresponding_depth_values = depth_image[valid_indices]
    scales = valid_gt_depth_values/corresponding_depth_values
    final_scale = np.median(scales)

    # fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    # n_bins=100
    # # We can set the number of bins with the *bins* keyword argument.
    # axs[0].hist(scales, bins=n_bins)
    # axs[1].hist(corresponding_depth_values, bins=n_bins)
    # axs[2].hist(valid_gt_depth_values, bins=n_bins)
    # plt.show()

    return final_scale

def get_subsampled_indices(num_indices, num_samples=100000):
    if num_samples >num_indices:
        num_samples = num_indices
    permutation = random.sample(range(0, num_indices), num_indices)
    return permutation[:num_samples]


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def freeze_params(self):
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._scaling.requires_grad = False
        self._opacity.requires_grad = False
        self._rotation.requires_grad = False

    def unfreeze_params(self):
        self._features_dc.requires_grad = True
        self._features_rest.requires_grad = True
        self._scaling.requires_grad = True
        self._opacity.requires_grad = True
        self._rotation.requires_grad = True

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_depth_map(self, depth_dict, scene_info, spatial_lr_scale: float=1.0):
        '''
            Inputs:
                depth_path: points to the depth images generated from a pre-trained depth estimation network
                gt_depth_path: points to ground truth depth maps.
                K: contains the intrinsics
        '''
        depth_path = depth_dict['depth_path']
        gt_depth_path = depth_dict['gt_depth_path']
        K_left, K_right = depth_dict['K_left'], depth_dict['K_right']
        # print(f"depth_path: {depth_path}")
        # print(f"gt_depth_path: {gt_depth_path}")
        # depth_image_paths = read_depth_image_paths(depth_path)
        # gt_depth_image_paths = read_gt_depth_image_paths(gt_depth_path)

        train_cameras, test_cameras, \
        train_depth_image_paths, test_depth_image_paths, \
        train_gt_depth_image_paths, test_gt_depth_image_paths = synchronize_depth_and_scene_info(scene_info, depth_path, gt_depth_path, offset=5)
                
        depth_image = depth_read(train_depth_image_paths['left'][0])
        gt_depth_image = depth_read(train_gt_depth_image_paths['left'][0])

        scale = get_depth_scale(depth_image, gt_depth_image)
        scaled_depth_image = depth_image*scale
        print(train_gt_depth_image_paths['left'][0])
        
        valid_depth_indices = depth_image!=-1

        # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        # n_bins=100
        # # We can set the number of bins with the *bins* keyword argument.
        # axs[0].imshow(depth_image/np.max(depth_image))
        # axs[1].imshow(scaled_depth_image/np.max(scaled_depth_image))

        point_cloud = depth_to_point_cloud(scaled_depth_image, K_left) #   extract point cloud from depth image
        point_colors = np.array(train_cameras[0].image)[valid_depth_indices]/256. #   extract valid rgb values. 

        sampled_indices = get_subsampled_indices(point_cloud.shape[0])
        point_cloud, point_colors = point_cloud[sampled_indices,:], point_colors[sampled_indices,:]
        pt_cld = BasicPointCloud(points = point_cloud, colors = point_colors, normals = np.zeros_like(point_cloud))
        


        self.create_from_pcd(pt_cld, spatial_lr_scale)

        return train_cameras, test_cameras, train_depth_image_paths, \
            test_depth_image_paths, train_gt_depth_image_paths, test_gt_depth_image_paths

        # synchronized_depth_image_paths = synchronize_depth_and_gt_depth(depth_image_paths,gt_depth_image_paths,offset=5)
        # depth_image = depth_read(synchronized_depth_image_paths['left'][0])
        # gt_depth_image = gt_depth_read(gt_depth_image_paths['left'][0])
        # H, W = depth_image.shape
        # gt_pt3d = depth_to_point_cloud(gt_depth_image, K_left)
        # pt3d = depth_to_point_cloud(depth_image, K_left)
        
        # print(f"pt3d.shape: {pt3d.shape} gt_pt3d.shape: {gt_pt3d.shape}")

        # camera = scene_info.train_cameras[5]
        
        # print(f"HI")
        # import open3d as o3d

        # # Load the point cloud
        # # pcd = o3d.read_point_cloud("path/to/point_cloud.pcd")
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pt3d)
        # o3d.visualization.draw_geometries([pcd])

        # # Visualize the point cloud
        # vis = o3d.visualization.Visualizer()
        # vis.add_geometry(pcd)
        # vis.run()


        

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1