
from functools import reduce
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import trimesh
from trimesh import viewer
from trimesh import sample
import pandas as pd

model_dir = '../models/'
dae_dir = 'visual_meshes/visual.dae'
png_dir = 'visual_meshes/texture_map.png'

csv_dir = '../training_data/objects_v1.csv'
obj_data = pd.read_csv(csv_dir,sep=',')
id2name = list(obj_data['object']) # begins at 0
id2symm = list(obj_data['geometric_symmetry'])


def FPS(ptcloud,count):
    if len(ptcloud) < count:
        return ptcloud
    U = ptcloud
    S = []
    dist_to_S = np.full(len(U),np.inf)
    
    newpt_idx = 0

    for _ in range(count):
        S.append(U[newpt_idx])
        dist_offer = np.sum((U - U[newpt_idx])**2,axis=1)
        dist_to_S = np.min(np.stack((dist_offer,dist_to_S)),axis=0)
        # find new point
        newpt_idx = np.argmax(dist_to_S)
    
    return np.array(S)

def get_canonical_pcd(obj_id,count):
    obj_name = id2name[obj_id]

    dae_f = open(os.path.join(model_dir+obj_name,dae_dir),'rb')
    png_resolver = trimesh.resolvers.FilePathResolver(os.path.join(model_dir+obj_name,png_dir))
    scenemesh = trimesh.exchange.dae.load_collada(dae_f,png_resolver)
    geo = list(scenemesh['geometry'].values())[0]
    mesh = trimesh.Trimesh(geo['vertices'],geo['faces'],vertex_normals=geo['vertex_normals'])
    sample_pcd,_ = sample.sample_surface(mesh, 10*count, face_weight=None, sample_color=False)
    pt_cloud_fps = FPS(sample_pcd,count)
    return pt_cloud_fps

import open3d
def showpcd(pcd):
    points = open3d.utility.Vector3dVector(pcd)
    _pcd = open3d.geometry.PointCloud()
    _pcd.points = points
    open3d.visualization.draw_geometries([_pcd])

from tqdm import tqdm
NUM_OBJECTS = 79
canonical_pcds_filename = './HW2/canonical_pcds.pkl'
if os.path.exists(canonical_pcds_filename):
    with open(canonical_pcds_filename, 'rb') as f:
        canonical_pcds = pickle.load(f)
else:
    canonical_pcds = []
    for id in tqdm(range(NUM_OBJECTS)):
        canonical_pcds.append(get_canonical_pcd(id,5000))
    with open(canonical_pcds_filename, 'wb') as f:
        pickle.dump(canonical_pcds, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

training_data_dir = "../training_data/v2.2"
split_dir = "../training_data/splits/v2"

def get_scene(scenename):
    # scenename refers to the i-j-k prefix
    prefix = os.path.join(training_data_dir, scenename)
    rgb = [prefix + "_color_kinect.png"]
    depth = [prefix + "_depth_kinect.png"]
    label = [prefix + "_label_kinect.png"]
    meta = [prefix + "_meta.pkl"]
    return rgb, depth, label, meta


def lift_seg(depth,label,meta,valid_id):
    # lifting
    intrinsic = meta['intrinsic']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    scene_pcd = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    # segmenting
    pcds = {}
    scales = {}
    for id in valid_id:
        raw_pcd = scene_pcd[np.where(label==id)]
        scales[id] = meta['scales'][id]
        pcds[id] = (raw_pcd) / meta['scales'][id]
        
    return pcds, scales

import copy
def draw_registration_result(source, target, transformation=None, show_normal=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp],point_show_normal=show_normal)

def get_open3d_PointCloud(pcd_nparray):
    points = open3d.utility.Vector3dVector(pcd_nparray.reshape([-1, 3]))
    pcd = open3d.geometry.PointCloud()
    pcd.points = points
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def scene_to_pcds(scenename):
    rgb_files, depth_files, label_files, meta_files = get_scene(scenename)
    rgb = np.array(Image.open(rgb_files[0])) / 255
    depth = np.array(Image.open(depth_files[0])) / 1000
    label = np.array(Image.open(label_files[0]))
    meta = load_pickle(meta_files[0])

    # note: some object may no appear in image (so point cloud is empty) while its id is in meta!
    valid_id = np.unique(label)
    valid_id = np.intersect1d(valid_id,meta['object_ids'])

    poses = np.array([meta['extrinsic']@meta['poses_world'][idx] for idx in valid_id]) # ground truth

    pcds, scales = lift_seg(depth,label,meta,valid_id)

    return valid_id, pcds, scales, poses


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_feature = voxel_size * 5

    pcd_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], open3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))
    return result

def loss(R,R_pred):
    R,R_pred = R,R_pred
    return np.abs(np.arccos(np.clip(0.5*(np.trace(R@R_pred.T)-1),-1,1)) *180 / np.pi)

# handle all symm situations
def rotation(w,theta):
    if theta == 0:
        return np.identity(3)
    # theta:degree
    w = w / np.linalg.norm(w)
    skew = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    theta = theta * np.pi / 180
    return np.identity(3)+skew*np.sin(theta)+skew@skew*(1-np.cos(theta))

rot = {
    'x2' : [rotation([1,0,0],0),rotation([1,0,0],180)],
    'x4' : [rotation([1,0,0],0),rotation([1,0,0],90),rotation([1,0,0],180),rotation([1,0,0],270)],
    'y2' : [rotation([0,1,0],0),rotation([0,1,0],180)],
    'y4' : [rotation([0,1,0],0),rotation([0,1,0],90),rotation([0,1,0],180),rotation([0,1,0],270)],
    'z2' : [rotation([0,0,1],0),rotation([0,0,1],180)],
    'z4' : [rotation([0,0,1],0),rotation([0,0,1],90),rotation([0,0,1],180),rotation([0,0,1],270)],
    'zinf' : [rotation([0,0,1],da) for da in range(0,360,5)],
    'xinf' : [rotation([1,0,0],da) for da in range(0,360,5)],
    'yinf' : [rotation([0,1,0],da) for da in range(0,360,5)],
}

from itertools import permutations,product
def compare(T,T_pred,scale,symm):
    t,t_pred = T[:3,3], T_pred[:3,3]
    # see above
    t_error = min(np.linalg.norm((t-t_pred)*scale),np.linalg.norm((2*t-t_pred)*scale))
    R,R_pred = T[:3,:3], T_pred[:3,:3]

    if symm == 'no':
        return t_error,loss(R,R_pred)

    symm_keys = symm.split('|')

    results = []
    if len(symm_keys)==0:
        results=[np.identity(3)]
    elif len(symm_keys)==1:
        results=rot[symm_keys[0]]
    else:
        for p in permutations(symm_keys):
            factors = [rot[i] for i in p]
            for factor_list in product(*factors):
                results.append(reduce(np.dot, factor_list))

    R_error = min([loss(R,R_pred@R_symm) for R_symm in results])
    return t_error,R_error