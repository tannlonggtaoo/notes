import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations,product
from functools import reduce
import os
import pickle
from PIL import Image

csv_dir = '../training_data/objects_v1.csv'
obj_data = pd.read_csv(csv_dir,sep=',')
id2name = list(obj_data['object']) # begins at 0
id2symm = list(obj_data['geometric_symmetry'])
training_data_dir = "../training_data/v2.2"
split_dir = "../training_data/splits/v2"

def get_split_files(split_name):
    # split_name = 'val' or 'train'
    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip() and int(line.strip()[0]) <= 2]
        rgb_files = [p + "_color_kinect.png" for p in prefix]
        depth_files = [p + "_depth_kinect.png" for p in prefix]
        label_files = [p + "_label_kinect.png" for p in prefix]
        meta_files = [p + "_meta.pkl" for p in prefix]
    return rgb_files, depth_files, label_files, meta_files


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def lift_seg_withrgb(depth,label,meta,valid_id,rgb):
    # lifting
    intrinsic = meta['intrinsic']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    scene_pcd = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    scene_rgbpcd = np.concatenate((rgb,scene_pcd),axis=2)
    # segmenting
    rgbpcds = []
    for id in valid_id:
        raw_pcd = scene_rgbpcd[np.where(label==id)]
        raw_pcd[:,3:] /= meta['scales'][id]
        rgbpcds.append(raw_pcd)
    return rgbpcds

def to_rgbpcds(rgb_file, depth_file, label_file, meta_file):
    rgb = np.array(Image.open(rgb_file)) / 255
    depth = np.array(Image.open(depth_file)) / 1000
    label = np.array(Image.open(label_file))
    meta = load_pickle(meta_file)

    valid_id = np.unique(label)
    valid_id = np.intersect1d(valid_id,meta['object_ids'])
    poses = np.array([meta['extrinsic']@meta['poses_world'][idx] for idx in valid_id]) # ground truth, some transactions are wrong

    rgbpcds = lift_seg_withrgb(depth,label,meta,valid_id,rgb)

    return valid_id, rgbpcds, poses



def pt_select(pcd,cnt):
    if len(pcd)>cnt:
        return pcd[np.random.choice(len(pcd),cnt,replace=False)]
    elif len(pcd)<cnt:
        return pcd[np.random.choice(len(pcd),cnt,replace=True)]
    else: return pcd

def preprocess(split_name,pt_cnt_perpcd):
    rgb_files, depth_files, label_files, meta_files = get_split_files(split_name)
    valid_id_arr,rgbpcds_arr,poses_arr=[],[],[]
    for i in tqdm(range(len(rgb_files))):
        valid_id, rgbpcds, poses = to_rgbpcds(rgb_files[i], depth_files[i], label_files[i], meta_files[i])
        rgbpcds = np.array([pt_select(pcd,pt_cnt_perpcd) for pcd in rgbpcds])
        valid_id_arr.append(valid_id)
        rgbpcds_arr.append(rgbpcds)
        poses_arr.append(poses)
    return np.concatenate(valid_id_arr,axis=0),np.concatenate(rgbpcds_arr,axis=0),np.concatenate(poses_arr,axis=0)

# dataset
class PoseDataset(Dataset):
    def __init__(self,datadir,thumbnail=False) -> None:
        super().__init__()
        metadata = np.load(datadir)
        
        self.valid_id = metadata['valid_id'].reshape(-1,1)
        self.rgbpcd = metadata['rgbpcd']
        self.pose = metadata['pose']

        print(f'metadata from {datadir} loaded...')

        if thumbnail:
            self.valid_id=self.valid_id[:32]
            self.rgbpcd=self.rgbpcd[:32]
            self.pose=self.pose[:32]
        
    def __len__(self) -> int:
        return len(self.valid_id)

    def __getitem__(self, index):
        return torch.Tensor(self.valid_id[index]),torch.Tensor(self.rgbpcd[index]),torch.Tensor(self.pose[index])


# model
def toRot(a,b):
    a = F.normalize(a, dim=-1)
    b = b - a * (a * b).sum(-1, keepdims=True)
    b = F.normalize(b, dim=-1)
    c = torch.cross(a, b, -1)
    return torch.stack([a, b, c], dim=-1)


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Sequential(
            nn.BatchNorm1d(6),

            nn.Conv1d(6,32,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64,512,1),
            nn.BatchNorm1d(512),
        )

        self.fc_stack = nn.Sequential(
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,9),
        )

    def forward(self,x):
        # x.shape is [batchsize,pt_count,6]
        # torch version is too low, so moveaxis is not available
        x = x.transpose(1,2)
        center = x[:,3:6].mean(dim=-1)
        x = self.emb(x)

        x = torch.max(x,2)[0]

        x = self.fc_stack(x)

        R = toRot(x[...,:3],x[...,3:6])
        M = F.pad(R,(0,1,0,1))
        M[:,:3,3] = x[...,6:] + center
        M[:,3,3] = 1

        return M

def check_dim():
    model = PoseNet()
    x = torch.rand(32,512,6) # a little different
    M = model(x)
    print(M.shape)
    print(M[0])



# loss
def rotation(w,theta):
    # theta:degree
    if theta == 0:
        ret = torch.eye(3)
    else:
        w = torch.FloatTensor(w)
        w = F.normalize(w,dim=-1)
        skew = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
        theta = theta * np.pi / 180
        ret = torch.eye(3)+skew*np.sin(theta)+torch.mm(skew,skew)*(1-np.cos(theta))

    ret = ret.to('cuda')
    return ret


def angleloss(R,R_pred):
    return torch.abs(torch.acos(torch.clamp(0.5*(torch.trace(torch.mm(R,R_pred.transpose(0,1)))-1),-1+1e-7,1-1e-7)) *180 / np.pi)


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

def ShapeAgnosticLoss(M_pred,M,valid_id,C=np.pi / 180,ret_exact_error=False):
    R_pred,R = M_pred[:,:3,:3],M[:,:3,:3]
    t_pred,t = M_pred[:,:3,3],M[:,:3,3]
    t_error = torch.norm(t-t_pred,dim=-1,keepdim=True) # approximation

    symms = [id2symm[int(id)] for id in valid_id]

    R_error = torch.zeros_like(valid_id).to('cuda')

    for i,symm in enumerate(symms):
        if symm == 'no':
            R_error[i] = angleloss(R[i],R_pred[i])
            continue
        
        symm_keys = symm.split('|')
        results = []
        if len(symm_keys)==0:
            results=[torch.eye(3)]
        elif len(symm_keys)==1:
            results=rot[symm_keys[0]]
        else:
            for p in permutations(symm_keys):
                factors = [rot[i] for i in p]
                for factor_list in product(*factors):
                    results.append(reduce(torch.mm, factor_list))

        R_error[i] = torch.min(torch.stack([angleloss(R[i],torch.mm(R_pred[i],R_symm.detach())) for R_symm in results]))

    if ret_exact_error:
        return t_error,R_error
    else:
        return torch.mean(t_error+C*R_error)


def train(model,trainloader,optimizer,epoch_limit):
    loss_record = []
    model = model.to('cuda')
    for e in range(epoch_limit):
        loss_sum = 0

        bar = tqdm(enumerate(trainloader),total=len(trainloader),desc=f'Epoch {e}/{epoch_limit} : ')
        for t, (valid_id,x,y) in bar:
            model.train()
            x = x.to('cuda')
            y = y.to('cuda')
            y_pred = model(x)

            loss = ShapeAgnosticLoss(y_pred,y,valid_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss

            bar.set_postfix(loss = loss.item())
            #if t % 10 == 0:
            #    print('Epoch %d, Iteration %d / %d, loss = %.4f' % (e, t, len(trainloader), loss))
          
        torch.cuda.empty_cache()
        loss_record.append(loss_sum/len(trainloader))
        torch.save(model.state_dict(), 'model.pth')
    
    return loss_record
 
# torch.autograd.set_detect_anomaly(True)
# the best way to check where nan value is derived..
# reason: acos have inf grad at 1 & -1


def mainprocess():
    processed_val_dir = '../HW2DATA/processed_val.npz'
    if not os.path.exists(processed_val_dir):
        valid_id_arr,rgbpcds_arr,poses_arr = preprocess('val',1024)
        np.savez(processed_val_dir,valid_id=valid_id_arr,   rgbpcd=rgbpcds_arr,pose=poses_arr)
    else: print('validation data have been preprocessed...')

    processed_train_dir = '../HW2DATA/processed_train.npz'
    if not os.path.exists(processed_train_dir):
        valid_id_arr,rgbpcds_arr,poses_arr = preprocess('train',1024)
        np.savez(processed_train_dir,valid_id=valid_id_arr, rgbpcd=rgbpcds_arr,pose=poses_arr)
    else: print('train data have been preprocessed...')

    
    batchsize = 64
    epochs = 10
    lr = 1e-3
    trainset = PoseDataset(processed_train_dir)
    trainloader = DataLoader(trainset,batchsize)
    model = PoseNet()
    optimizer = Adam(model.parameters(),lr=lr)

    loss_hist = train(model,trainloader,optimizer,epochs)
    print(loss_hist)

if __name__ == '__main__':
    mainprocess()