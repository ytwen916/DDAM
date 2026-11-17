import torch
import torch.nn as nn
import torch.nn.functional as F


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)

        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

			
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()

        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org , enhance ):
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)

        return E

class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class L_exp_v2(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_exp_v2, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = torch.FloatTensor([mean_val]).cuda()

    def forward(self, x, x_or):
        x = torch.mean(x,1,keepdim=True)
        x_or = torch.mean(x_or,1,keepdim=True)
        mean = self.pool(x)
        mean_or = self.pool(x_or)
        d = torch.mean(torch.pow(mean-(mean_or+self.mean_val)/2,2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class L_denoise(nn.Module):
    def __init__(self):
        super(L_denoise, self).__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.pool3 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

    def forward(self, x, direction):
        if direction == 'x':
            grad = x[:, :, :, :-1] - x[:, :, :, 1:]
        elif direction == 'y':
            grad = x[:, :, :-1, :] - x[:, :, 1:, :]
        else:
            raise ValueError("Direction should be 'x' or 'y'")
        
        abs_grad = torch.abs(grad)
        max_grad = torch.max(torch.max(abs_grad, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        min_grad = torch.min(torch.min(abs_grad, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        norm1 = (abs_grad - min_grad) / (max_grad - min_grad + 1e-6)

        with torch.no_grad():

            pool_x=self.pool1(x)
            if direction == 'x':
                pool_x = pool_x[:, :, :, :-1] - pool_x[:, :, :, 1:]
            elif direction == 'y':
                pool_x = pool_x[:, :, :-1, :] - pool_x[:, :, 1:, :]
            else:
                raise ValueError("Direction should be 'x' or 'y'")
        
            # pool_x=self.pool1(grad)

            pool_grad=torch.abs(pool_x)

            pool_max_grad = self.pool2(pool_grad)
            pool_min_grad = -self.pool2(-pool_grad)

            pool_max_grad_pool = self.pool3(pool_max_grad)
            pool_min_grad_pool = self.pool3(pool_min_grad)

            weight = (pool_grad - pool_min_grad_pool) / (pool_max_grad_pool-pool_min_grad_pool + 1e-4)

        loss = norm1 * (weight+1e-2)
        return loss

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, margin=0.0,**kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.margin = margin
        # print('COMMON MMDLoss : ')
        # print('margin: ',self.margin)


    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        # source = F.normalize(source, p=2, dim=1)
        # target = F.normalize(target, p=2, dim=1)
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            # TT = F.relu(XX + YY - XY - YX-self.margin)



            TT = XX + YY - XY - YX
            # print(TT)
            # TT = F.softplus(TT/self.margin)
            #
            # idx = TT>self.margin
            # loss = torch.mean(TT[idx])
            # if idx.sum()>0:
            #     loss = torch.mean(TT[idx])
            # else:
            #     loss = torch.zeros(1).cuda()
            # print(TT.shape)
            loss = torch.mean(TT)


            return loss