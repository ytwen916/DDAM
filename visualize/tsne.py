from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import scipy.io
import os
from sklearn import manifold, datasets
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net_zjy as embed_net
from utils import *
from loss import OriTripletLoss
from random_erasing import RandomErasing
import numpy as np
from PIL import Image
import torch.utils.data as data
from loss import QuadTripletLoss, CKALoss, ECKALoss, OriTripletLoss

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='DN348', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
# parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
# parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=256, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int, metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')

parser.add_argument('--lkvreuse', default=1, type=int, metavar='t', help='random seed')


parser.add_argument('--num_pos', default=20, type=int, help='num of pos per identity in each modality')
parser.add_argument('--batch-size', default=20, type=int, metavar='B', help='training batch size')

parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--ckaw', default=0.5,type=float, help='weight for cka loss')
parser.add_argument('--triw', default=0.5, type=float, help='weight for triloss')
# parser.add_argument('--margin', default=0, type=float, help='margin for triplet loss')
parser.add_argument('--cross', default=1.0, type=float, help='weight for idloss')
parser.add_argument('--mode', default='v2t', type=str, help='v2t, t2v') #
parser.add_argument('--attpos', default=[0,4,6,0], type=int, help='position of attention layer')
parser.add_argument('--lnratio', default=0.5, type=float, help='ration for ln')
parser.add_argument('--cratio', default=32.0, type=float, help='ratio for down and up')
parser.add_argument('--log_path', default='TSNE/', type=str, help='log save path') #named as std real var

parser.add_argument('--model_path', default='Exp/DN348/ckan/TSNE/', type=str, help='log save path')
parser.add_argument('--save_path', default='Exp/DN348/ckan/TSNE/', type=str, help='log save path')
parser.add_argument('--mat_name', default='Exp/DN348/ckan/', type=str, help='log save path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

if args.lkvreuse>0:
    reuse=True
else:
    reuse=False

dataset = args.dataset
if dataset == 'sysu':
    data_path = './Dataset/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = './Datasets/RegDB/'
    n_class = 206
    test_mode = [2, 1]
elif dataset =='llcm':
    data_path = '/media/jqzhu/e/lhhuang/LLCM/LLCM/'
    n_class = 713
    test_mode = [1, 2] #[2, 1]: VIS to IR; [1, 2]: IR to VIS
elif dataset =='DN348':
    data_path = '/media/jqzhu/941A7DD31A7DB33A/simi/DN/dn348/'
    # trainset = DN348Data(data_path, args.trial, transform=transform_train, img_size=(args.img_w, args.img_h))
    n_class = 200
    test_mode = [2, 1]


checkpoint_path = args.model_path


print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_map = 0  # best test map
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.999, 0.999, 0.999]),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

name = args.mat_name
print(f"==========> tsne_{name}.mat")
result = scipy.io.loadmat(f'{name}')
query_feature = torch.FloatTensor(result['query_fc'])
# query_cam = result['query_cam'][0]
query_label = result['query_label'][0]


gallery_feature = torch.FloatTensor(result['gallery_fc'])
# gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

def load_data1(input_data_path, dn_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]

        file_label = [int(s.split('/')[0]) for s in data_file_list]
        pid_container = set()

        for i in range(len(file_label)):
            # str1 = '%s/%s.jpg'%(self.all_dir,data_mat[i][0])
            # train_paths.append(img_paths[img_paths.index(str1)])
            pid = int(file_label[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid = pid2label[pid]
            file_label[i] = pid

    return file_image, file_label
class DN348Data(data.Dataset):  # new
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None,img_size = (256,256)):
        # Load training images (path) and labels
        train_day_list = data_dir + 'train_test_split/train_list_day.txt'
        train_night_list = data_dir + 'train_test_split/train_list_night.txt'
        train_day_path = data_dir + 'day/'
        train_night_path = data_dir + 'night/'

        day_img_file, train_day_label = load_data1(train_day_list, train_day_path)
        night_img_file, train_night_label = load_data1(train_night_list, train_night_path)

        train_day_image = []
        for i in range(len(day_img_file)):
            img = Image.open(train_day_path + day_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_day_image.append(pix_array)
        train_day_image = np.array(train_day_image)

        train_night_image = []
        for i in range(len(night_img_file)):
            img = Image.open(train_night_path + night_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_night_image.append(pix_array)
            # print(pix_array.shape)
        train_night_image = np.array(train_night_image)

        # BGR to RGB
        self.train_color_image = train_day_image
        self.train_color_label = train_day_label

        # BGR to RGB
        self.train_thermal_image = train_night_image
        self.train_thermal_label = train_night_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_day_label)
class LLCMData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        
        query_feature = torch.FloatTensor(result['query_f'])
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]
        
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]
        

        # BGR to RGB
        self.train_color_image = query_feature  
        self.train_color_label = query_label
        
        # BGR to RGB
        self.train_thermal_image = gallery_feature
        self.train_thermal_label = gallery_label
                
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)



def process_llcm(img_dir, mode = 1):
    if mode == 1:
        input_data_path = os.path.join(data_path, 'idx/test_vis.txt')
    elif mode == 2:
        input_data_path = os.path.join(data_path, 'idx/test_nir.txt')
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        file_cam = [int(s.split('c0')[1][0]) for s in data_file_list]
        
    return file_image, np.array(file_label), np.array(file_cam)
    


end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    # query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    # gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

elif dataset == 'rgbn':
    # training set
    trainset = LLCMData(None, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

elif dataset == 'DN348':
    # training set
    trainset = DN348Data(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

tsne_class = len(np.unique(trainset.train_color_label))

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(tsne_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(tsne_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building modelbyhlh..')
# net = embed_net(n_class, arch=args.arch, reuse=reuse)
# net = embed_net(tsne_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net = embed_net(tsne_class, arch=args.arch,cr=args.cratio,lnr=args.lnratio,attpos=args.attpos)
net = net.to(device)
cudnn.benchmark = True


# define loss function
criterion_id = nn.CrossEntropyLoss()   #label_smoothing=0.1
loader_batch = args.batch_size * args.num_pos
criterion_tri = QuadTripletLoss()
# criterion_tri = OriTripletLoss(args.batch_size, args.margin)
criterion_seman = CKALoss(epsilon=1e-12) #if use linear, sigma not work

criterion_id.to(device)
criterion_tri.to(device)
criterion_seman.to(device)

color= [    # Bright colors  
    (0.8, 0.3, 0.2, 1.0),  # Bright reddish-brown  
    (1.0, 0.0, 0.0, 1.0),  # Bright red  
    (1.0, 0.6, 0.1, 1.0),  # Bright orange  
    (1.0, 0.1, 0.7, 1.0),  # Bright pink  
    (0.9, 0.2, 0.4, 1.0),  # Bright red-pink  
    (0.8, 0.2, 1.0, 1.0),  # Bright purple  
    (0.7, 0.5, 0.3, 1.0),  # Bright brown  
    (0.0, 1.0, 0.0, 1.0),  # Light green  
    (0.4, 1.0, 0.6, 1.0),  # Light mint  
    (0.5, 0.7, 0.9, 1.0),  # Light blue  
    (0.4, 0.8, 0.3, 1.0),  # Light green  
    (0.5, 0.7, 0.4, 1.0),  # Light olive  
    (0.2, 0.6, 0.8, 1.0),  # Light blue  
    (0.7, 0.9, 0.4, 1.0),  # Bright yellow-green  
    (0.7, 0.3, 0.8, 1.0),  # Bright purple  
    (0.3, 0.8, 0.5, 1.0),  # Light teal  
    (0.1, 0.8, 1.0, 1.0),  # Light cyan  
    (1.0, 0.9, 0.3, 1.0),  # Bright yellow  
    (0.8, 0.2, 0.6, 1.0),  # Bright pink-purple  
    (0.9, 0.1, 0.9, 1.0),  # Bright magenta  
    (0.6, 0.1, 0.4, 1.0),  # Bright dark pink  
    (0.9, 0.2, 0.3, 1.0),  # Bright red  
    (0.8, 0.4, 0.1, 1.0),  # Bright orange-brown  
    (0.9, 0.2, 0.3, 1.0),  # Bright red  
    (1.0, 0.9, 0.3, 1.0),  # Bright yellow  
    (0.8, 0.2, 0.6, 1.0),  # Bright pink-purple  
    (0.9, 0.1, 0.9, 1.0),  # Bright magenta  
    (0.8, 0.4, 0.1, 1.0),  # Bright orange-brown  

    # Light colors  
    
    (0.1, 0.1, 1.0, 1.0),  # Light blue  
    (0.3, 0.3, 0.9, 1.0),  # Light blue  
    (0.2, 0.2, 0.5, 1.0),  # Light blue  
    (0.3, 0.4, 0.6, 1.0),  # Light blue  
    (0.4, 0.1, 0.9, 1.0),  # Light purple  
    (0.5, 0.4, 0.2, 1.0),  # Light brown  
    (0.6, 0.7, 0.3, 1.0),  # Light green  
    (0.7, 0.5, 0.6, 1.0),  # Light pink  
    (0.0, 0.5, 0.5, 1.0),  # Light teal  
    (0.2, 0.8, 0.2, 1.0),  # Light green  
    (0.4, 0.9, 0.7, 1.0),  # Light mint  
    (0.6, 0.3, 0.2, 1.0),  # Light brown  
    (0.3, 0.2, 0.4, 1.0),  # Light purple  
    (0.5, 0.6, 0.7, 1.0),  # Light blue  
    (0.6, 0.4, 0.9, 1.0),  # Light purple  
    (0.7, 0.3, 0.1, 1.0),  # Light brown  
    (0.8, 0.5, 0.2, 1.0),  # Light orange  
    (0.0, 0.5, 0.5, 1.0),  # Light teal  
    (0.2, 0.8, 0.2, 1.0),  # Light green  
    (0.4, 0.9, 0.7, 1.0),  # Light mint  
    (0.6, 0.3, 0.2, 1.0),  # Light brown  
    (0.3, 0.2, 0.4, 1.0),  # Light purple  
    (0.5, 0.6, 0.7, 1.0),  # Light blue  
    (0.6, 0.4, 0.9, 1.0),  # Light purple  
    (0.7, 0.3, 0.1, 1.0),  # Light brown  
    (0.8, 0.5, 0.2, 1.0),  # Light orange  

        # (0.2, 0.2, 0.5, 1.0),
        # (0.3, 0.4, 0.6, 1.0),
        # (0.4, 0.1, 0.9, 1.0),
        # (0.5, 0.4, 0.2, 1.0),
        # (0.6, 0.7, 0.3, 1.0),
        # (0.7, 0.5, 0.6, 1.0),
        # (0.8, 0.4, 0.1, 1.0),
        # (0.9, 0.2, 0.3, 1.0),
        # (1.0, 0.9, 0.3, 1.0),
        # (0.0, 0.5, 0.5, 1.0),
        # (0.2, 0.8, 0.2, 1.0),
        # (0.4, 0.9, 0.7, 1.0),
        # (0.6, 0.3, 0.2, 1.0),
        # (0.8, 0.2, 0.6, 1.0),
        # (0.9, 0.1, 0.9, 1.0),
        # (0.3, 0.2, 0.4, 1.0),
        # (0.5, 0.6, 0.7, 1.0),
        # (0.6, 0.4, 0.9, 1.0),
        # (0.7, 0.3, 0.1, 1.0),
        # (0.8, 0.5, 0.2, 1.0),

        ]#R G B

def plot_embedding(X, y, z, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(15, 12), dpi=300)
    cx, cy = [], []

    r = []
    #print(X.shape[0])
    for i in range(X.shape[0]):
        cx.append(X[i, 0])
        cy.append(X[i, 1])
        if z[i] == 1:
            #print(i, y[i])
            # print("==================",i,y[i])
            plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='o')
        else:
            plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='^')


        
def train(epoch, save_path):
    #current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    cc_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    seman_loss = AverageMeter()

    correct = 0
    total = 0
    # switch to train mode
    print('==> STatrt...')

    end = time.time()
    with torch.no_grad():
      for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        labels = torch.cat((label1, label2), 0)
    #   print("=====labels", labels)
        labels = labels.to(torch.int)

        z1 = torch.ones(label1.shape)
        z2 = torch.zeros(label2.shape)
        z = torch.cat((z1, z2), 0)
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        print("================>input1 ", input1)
        print("================>input2 ", input2)

        a = labels.unique()

        for i in range(len(a)):
            for j in range(len(labels)):
                if labels[j] == a[i]:
                    labels[j] = i
            #print(labels)
        from sklearn.decomposition import PCA

        data_time.update(time.time() - end)
        out = torch.cat((input1, input2), 0)
        # 将数据转为numpy格式
        out_np = out.detach().cpu().numpy()
        print("out_np shape:", out_np.shape)

        # 使用PCA进行预处理，减少到50维
        pca = PCA(n_components=50)
        out_pca = pca.fit_transform(out_np)
    #   tsne = manifold.TSNE(n_components=2, init='pca')
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=30, learning_rate=100, n_iter=3000)
        X_tsne = tsne.fit_transform(out_pca)
        plot_embedding(X_tsne, labels, z)
        plt.xticks([])
        plt.yticks([])
        print(f"saving {save_path}")
        plt.savefig(osp.join(save_path, 'tsne_{}.jpg'.format(batch_idx)))



class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        
        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            # print("uni_label: ", uni_label)
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                batch_int = int(batch_idx[i])
                sample_color  = np.random.choice(color_pos[batch_int], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_int], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          

# plt.rcParams.update({'font.size': 14})
# training
print('==> Start Training...')
print('==> Preparing Data Loader...')
# identity sampler
epoch=0
print(trainset.train_color_label)
sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)

trainset.cIndex = sampler.index1  # color index
trainset.tIndex = sampler.index2  # thermal index
print(epoch)
print(trainset.cIndex)
print(trainset.tIndex)

loader_batch = args.batch_size * args.num_pos

trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                              sampler=sampler, num_workers=args.workers, drop_last=True)
# training
save_path = args.save_path
train(epoch, save_path)
