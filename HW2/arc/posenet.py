from imghdr import tests
from smtpd import DebuggingServer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, lr_scheduler
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
        self.filepath = [os.path.join(datadir,filename) for filename in os.listdir(datadir)]
        if thumbnail:
            self.filepath = self.filepath[:8]
        
    def __len__(self) -> int:
        return len(self.filepath)

    def __getitem__(self, index):
        metadata = np.load(self.filepath[index])
        return torch.Tensor(metadata['valid_id']),torch.Tensor(metadata['rgbpcd']),torch.Tensor(metadata['pose'])


# model
def toRot(a,b):
    a = F.normalize(a, dim=-1)
    b = b - a * (a * b).sum(-1, keepdims=True)
    b = F.normalize(b, dim=-1)
    c = torch.cross(a, b, -1)
    return torch.stack([a, b, c], dim=-1)


class PoseNet(nn.Module):
    def __init__(self,shapenum=79):
        super().__init__()
        self.rgbpcd_emb1 = nn.Sequential(
            nn.Conv1d(6,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.rgbpcd_emb2 = nn.Sequential(
            nn.Conv1d(128,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.rgbpcd_emb3 = nn.Sequential(
            nn.Conv1d(128+256,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.rgbpcd_emb4 = nn.Sequential(
            nn.Conv1d(128+256+1024,2048,1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.shape_emb_n = nn.Sequential(
            nn.Linear(shapenum,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,128),
        )

        self.fc_stack1_n = nn.Sequential(
            nn.Linear(128+256+1024 + 2048 + 128,1024),
            nn.ReLU(),
        )
        self.fc_stack2 = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
        )
        self.fc_stack3 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
        )
        self.fc_stack4 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.fc_stack5 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
        )
        self.fc_stack6 = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
        )
            
        self.fc_out1_n = nn.Linear(32,9)

        self.shapenum = shapenum

    def forward(self,x,id):
        # x.shape is [batchsize,pt_count,6]
        # id.shape is [batchsize,1]
        # torch version is too low, so moveaxis is not available

        # x = x[:,np.random.choice(x.shape[1],self.pt_num),:]
        x = x.transpose(1,2)
        # center = x[:,3:6].mean(dim=-1,keepdim=True)
        # scale = x[:,3:6].max(dim=-1,keepdim=True)[0]-x[:,3:6].min(dim=-1,keepdim=True)[0] + 1e-4
        # x[:,3:6] = (x[:,3:6] - center) / scale
        # x[:,:3] = 2*x[:,:3] - 1 # color

        center = x.mean(dim=-1,keepdim=True)
        scale = x.max(dim=-1,keepdim=True)[0]-x.min(dim=-1,keepdim=True)[0] + 1e-7
        x = (x - center) / scale

        x = self.rgbpcd_emb1(x)
        res1 = self.rgbpcd_emb2(x)
        x = torch.cat((x,res1),-2)

        res2 = self.rgbpcd_emb3(x)
        x = torch.cat((x,res2),-2)
        
        res3 = self.rgbpcd_emb4(x)
        x = torch.cat((x,res3),-2)

        x = torch.max(x,2)[0]

        shape = self.shape_emb_n(F.one_hot(id.long().squeeze(),self.shapenum).float())
        x = torch.cat((x,shape),-1)

        x = self.fc_stack1_n(x)
        x = self.fc_stack2(x)
        x = self.fc_stack3(x)
        x = self.fc_stack4(x)
        x = self.fc_stack5(x)
        x = self.fc_stack6(x)
        x = self.fc_out1_n(x)

        R = toRot(x[...,:3],x[...,3:6])
        M = F.pad(R,(0,1,0,1))
        M[:,:3,3] = scale[:,3:].squeeze()*x[...,6:] + center[:,3:].squeeze()
        M[:,3,3] = 1

        #return M
        return M

def check_dim():
    model = PoseNet()
    x = torch.rand(32,1024,6) # a little different
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


def angleloss(R,R_pred,no_grad=False):
    if no_grad:
        return torch.abs(torch.acos(torch.clamp(0.5*(torch.trace(torch.mm(R,R_pred.transpose(0,1)))-1),-1,1)) *180 / np.pi)
    else:
        return torch.abs(torch.acos(torch.clamp(0.5*(torch.trace(torch.mm(R,R_pred.transpose(0,1)))-1),-1+1e-7,1-1e-7)) *180 / np.pi)


rot_train = {
    'x2' : [rotation([1,0,0],0),rotation([1,0,0],180)],
    'x4' : [rotation([1,0,0],0),rotation([1,0,0],90),rotation([1,0,0],180),rotation([1,0,0],270)],
    'y2' : [rotation([0,1,0],0),rotation([0,1,0],180)],
    'y4' : [rotation([0,1,0],0),rotation([0,1,0],90),rotation([0,1,0],180),rotation([0,1,0],270)],
    'z2' : [rotation([0,0,1],0),rotation([0,0,1],180)],
    'z4' : [rotation([0,0,1],0),rotation([0,0,1],90),rotation([0,0,1],180),rotation([0,0,1],270)],
    'zinf' : [rotation([0,0,1],da) for da in range(0,360,10)],
    'xinf' : [rotation([1,0,0],da) for da in range(0,360,10)],
    'yinf' : [rotation([0,1,0],da) for da in range(0,360,10)],
    }
# inf is 60 in first 10 epochs

rot_test = {
    'x2' : [rotation([1,0,0],0),rotation([1,0,0],180)],
    'x4' : [rotation([1,0,0],0),rotation([1,0,0],90),rotation([1,0,0],180),rotation([1,0,0],270)],
    'y2' : [rotation([0,1,0],0),rotation([0,1,0],180)],
    'y4' : [rotation([0,1,0],0),rotation([0,1,0],90),rotation([0,1,0],180),rotation([0,1,0],270)],
    'z2' : [rotation([0,0,1],0),rotation([0,0,1],180)],
    'z4' : [rotation([0,0,1],0),rotation([0,0,1],90),rotation([0,0,1],180),rotation([0,0,1],270)],
    'zinf' : [rotation([0,0,1],da) for da in range(0,360,3)],
    'xinf' : [rotation([1,0,0],da) for da in range(0,360,3)],
    'yinf' : [rotation([0,1,0],da) for da in range(0,360,3)],
}

def ShapeAgnosticLoss(M_pred,M,valid_id,x,C=0.1,C_ADD=0.1,ret_exact_error=False):
    R_pred,R = M_pred[:,:3,:3],M[:,:3,:3]
    t_pred,t = M_pred[:,:3,3],M[:,:3,3]
    t_error = torch.min(torch.norm(2*t-t_pred,dim=-1,keepdim=True),torch.norm(t-t_pred,dim=-1,keepdim=True)) # approximation

    symms = [id2symm[int(id)] for id in valid_id]

    R_error = torch.zeros_like(valid_id).to('cuda')

    # debugging
    rot = rot_test if ret_exact_error else rot_train

    for i,symm in enumerate(symms):
        if symm == 'no':
            R_error[i] = angleloss(R[i],R_pred[i],ret_exact_error)
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

        R_error[i] = torch.min(torch.stack([angleloss(R[i],torch.mm(R_pred[i],R_symm),ret_exact_error) for R_symm in results]))

    if ret_exact_error:
        return t_error,R_error
    else:
        x = x[:,:,3:]
        x_gt = rigidmotion(x,R,t)
        x_gt2 = x_gt + t.unsqueeze(-1)
        x_pred = rigidmotion(x,R_pred,t_pred)
        ADD = torch.min(((x_gt-x_pred)**2).sum((1,2)),((x_gt2-x_pred)**2).sum((1,2)))

        t_error = F.relu(t_error - 0.01*0.5)
        R_error = F.relu(R_error - 5*0.5)
        return torch.mean(t_error+C*R_error+C_ADD*ADD)

def rigidmotion(x,R,t):
    return torch.matmul(R,x.transpose(1,2)) + t.unsqueeze(-1)

def unpack_DATA(dir,elem_dir):
    metadata = np.load(dir)
    valid_id = metadata['valid_id'].reshape(-1,1)
    rgbpcd = metadata['rgbpcd']
    pose = metadata['pose']
    for i in tqdm(range(len(valid_id)),desc=f'Unpacking {dir} : '):
        np.savez(os.path.join(elem_dir,f'{i}.npz'),valid_id=valid_id[i],rgbpcd=rgbpcd[i],pose=pose[i])

def train(model,trainloader,optimizer,gradclip):
    model.train()
    model = model.to('cuda')
    loss_sum = 0
    for t, (valid_id,x,y) in trainloader:

        valid_id = valid_id.to('cuda')
        x = x.to('cuda')
        y = y.to('cuda')
        y_pred = model(x.detach().clone(),valid_id)
        # x will be changed inplace so detach-copy

        loss = ShapeAgnosticLoss(y_pred,y,valid_id,x)

        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), gradclip)
        optimizer.step()

        loss_sum += loss.item()
        torch.cuda.empty_cache()

    return loss_sum/len(trainloader)
 
# torch.autograd.set_detect_anomaly(True)
# the best way to check where nan value is derived..
# reason: acos have inf grad at 1 & -1

def test(model,loader):
    model = model.to('cuda')
    bar = tqdm(enumerate(loader),total=len(loader),desc='Test : ',leave=False)

    t_correct_count = 0
    R_correct_count = 0
    correct_count = 0
    total_count = 0
    
    # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
    model.eval()
    with torch.no_grad():
        for t, (valid_id,x,y) in bar:
            valid_id = valid_id.to('cuda')
            x = x.to('cuda')
            y = y.to('cuda')
            y_pred = model(x.detach().clone(),valid_id)

            t_error,R_error = ShapeAgnosticLoss(y_pred,y,valid_id,x, ret_exact_error=True)

            t_correct_count += (t_error<0.01).sum()
            R_correct_count += (R_error<5).sum()
            correct_count += ((t_error<0.01)*(R_error<5)).sum()
            total_count += len(t_error)
    
    return t_correct_count.item() / total_count, R_correct_count.item() / total_count, correct_count.item() / total_count


def handle_process(dir,processed_dir,name):
    if not os.path.exists(processed_dir):
        valid_id_arr,rgbpcds_arr,poses_arr = preprocess(name,1024)
        np.savez(processed_dir,valid_id=valid_id_arr,   rgbpcd=rgbpcds_arr,pose=poses_arr)
    if len(os.listdir(dir)) == 0:
        unpack_DATA(processed_dir,dir)
    print(f'{name} data have been preprocessed...')

def reportAcc(name,Tacc,Racc,ALLacc):
    print(f"Accs on {name} set: Tacc = {Tacc:.4f},\
            Racc = {Racc:.4f},\
            ALLacc = {ALLacc:.4f}")

def mainprocess(debug:bool):
    processed_val_dir = '../HW2DATA/processed_val.npz'
    val_dir = '../HW2DATA/val'
    handle_process(val_dir,processed_val_dir,'val')

    processed_train_dir = '../HW2DATA/processed_train.npz'
    train_dir = '../HW2DATA/train'
    handle_process(train_dir,processed_train_dir,'train')

    if debug:
        model_dir = '../HW2DATA/model_debug.pth'
        print('debugging...')
    else:
        model_dir = '../HW2DATA/model.pth'
        print('training...')


    batchsize = 16
    epochs = 40
    lr = 1e-3
    num_workers = 4 if not debug else 0
    gradclip = 1
    start_schedule = 5

    model = PoseNet(79)

    # loading model
    if os.path.exists(model_dir):
        prevdict = torch.load(model_dir)
        curdict = model.state_dict()
        statedict = {k:v for k,v in prevdict.items() if k in curdict.keys()}
        curdict.update(statedict)
        model.load_state_dict(curdict)
        print("Previous model loaded...")
        torch.save(model.state_dict(), model_dir)

    optimizer = Adam(model.parameters(),lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # loaders
    trainset = PoseDataset(train_dir,debug)
    print(f"trainset size = {len(trainset)}")
    trainloader = DataLoader(trainset,batchsize,shuffle=True,num_workers=num_workers)
    

    # create a train subset to check fitting
    if debug:
        trainset_for_test = trainloader.dataset
    else:
        trainset_for_test,_ = random_split(trainloader.dataset,[int(0.01*len(trainloader.dataset)),len(trainloader.dataset)-int(0.01*len(trainloader.dataset))])
    print(f'trainset for test size = {len(trainset_for_test)}')
    trainloader_for_test = DataLoader(trainset_for_test,trainloader.batch_size)

    validset = PoseDataset(val_dir)
    testset, validset = random_split(validset,[int(0.8*len(validset)),len(validset)-int(0.8*len(validset))])
    print(f"testset size = {len(testset)}")
    print(f"validset size = {len(validset)}")
    validloader = DataLoader(validset,batchsize)
    testloader = DataLoader(testset,batchsize)

    for e in range(epochs):
        bar = tqdm(enumerate(trainloader),total=len(trainloader),desc=f'Epoch {e+1}/{epochs} : ')
        loss = train(model,bar,optimizer,gradclip)
        torch.save(model.state_dict(), model_dir)
        torch.cuda.empty_cache()

        T_train_for_test,R_train_for_test,all_train_for_test = test(model,trainloader_for_test)
        T_val,R_val,all_val = test(model,validloader) if not debug else (-1,-1,-1)
        print(f"----loss = {loss:.4f}----")
        reportAcc("train sub",T_train_for_test,R_train_for_test,all_train_for_test)
        if not debug: reportAcc("val",T_val,R_val,all_val)

        if loss < start_schedule: scheduler.step()

    if not debug:
        T_test,R_test,all_test = test(model,testloader)
        reportAcc("test",T_test,R_test,all_test)

# 10 epoch done

if __name__ == '__main__':
    mainprocess(debug=True)