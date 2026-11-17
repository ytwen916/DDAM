import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)
import warnings
# from model.KAN_Implementations.Efficient_KAN.efficient_kan import Efficient_KANLinear
# from model.KAN_Implementations.Original_KAN.original_kan import KAN
# from model.KAN_Implementations.Fast_KAN.fast_kan import Fast_KANLinear

# class IED(nn.Module):
#
# 	def __init__(self):
# 		super(IED, self).__init__()
#
# 		self.relu = nn.ReLU(inplace=True)
#
#
#
#
#
# 		self.e_conv8 = nn.Conv2d(32,3, 3, 1, 1, bias=True)
#
# 		self.quarter_conv = nn.Conv2d(3, 3, 7, 4)
# 		self.hexa_conv = nn.Conv2d(3, 3, 3, 2)
#
# 		# 定义协同注意力模块
# 		self.co_attention = MultiScaleCoAttention(channels=32)
#
# 		self.channels = 3
# 		kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
# 				[0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
# 				[0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
# 				[0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
# 				[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
# 		kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)		# 固定的5x5高斯核（Gaussian Filter），用于在后面 enhance() 函数中对图像进行卷积平滑
# 		self.kernel = kernel.repeat(self.channels, 1, 1, 1).cuda()
#
#
# 	def forward(self, x):
# 		quarter_scale_x = self.quarter_conv(x)
# 		hexa_scale_x = self.hexa_conv(quarter_scale_x)
#
# 		x1 = self.relu(self.e_conv1(x))
# 		quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x))
# 		hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))
#
# 		x2 = self.relu(self.e_conv2(x1))		# 每经过一层卷积，感受野变大,称它为“语义尺度更高”的特征
# 		quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1))
# 		hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))
#
# 		x3 = self.relu(self.e_conv3(x2))
# 		quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2))
# 		hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))
#
# 		x4 = self.relu(self.e_conv4(x3))
# 		quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3))
# 		hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))
#
# 		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
# 		quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1)))
# 		hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))
#
# 		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
# 		quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1)))
# 		hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))
#
# 		x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1)))
# 		quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1)))
# 		hexa_scale_x7 = self.relu(self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1)))
#
# 		enhanced_features = self.co_attention(x7, quarter_scale_x7, hexa_scale_x7)		# attention_map 是一个像素级的权重图，用来衡量不同空间位置在多尺度特征融合时的重要程度
#
#
# 		# print(enhanced_features.size())
# 		x_r = torch.tanh(self.e_conv8(enhanced_features))
#
# 		# enhance_image,enhance_image_d = self.enhance(x,x_r)		# enhance_image: 逐像素曲线增强结果；enhance_image_d: 加入细节控制的增强结果；
# 		# print(x_r.size())
# 		return x_r
#
class MultiScaleCoAttention(nn.Module):
	def __init__(self, channels):
		super(MultiScaleCoAttention, self).__init__()
		self.channels = channels
		# self.conv_out = nn.Conv2d(channels, channels, 1, bias=False)
		# 可选的预处理层，用于对齐尺度
		# self.align_conv = nn.Conv2d(channels, channels, 1, bias=False)
		# nn.init.zeros_(self.align_conv.weight)
		# self.relu = nn.ReLU(inplace=True)
		# self.bn = nn.BatchNorm2d(channels)


		# self.bn_h = nn.BatchNorm2d(channels)
		# self.bn_m = nn.BatchNorm2d(channels)
		# self.bn_l = nn.BatchNorm2d(channels)


		self.q_proj_h = nn.Conv1d(in_channels=channels,out_channels=channels//2,kernel_size=1,padding=0)
		self.k_proj_h = nn.Conv1d(in_channels=channels,out_channels=channels//2,kernel_size=1,padding=0)
		self.v_proj_h = nn.Conv1d(in_channels=channels,out_channels=channels//2,kernel_size=1,padding=0)

		self.q_proj_l = nn.Conv1d(in_channels=channels,out_channels=channels//2,kernel_size=1,padding=0)
		self.k_proj_l = nn.Conv1d(in_channels=channels,out_channels=channels//2,kernel_size=1,padding=0)
		self.v_proj_l = nn.Conv1d(in_channels=channels,out_channels=channels//2,kernel_size=1,padding=0)

		self.h_up = nn.Sequential(
			nn.Conv1d(channels // 2, channels, 1),
			nn.BatchNorm1d(channels)

		)
		nn.init.constant_(self.h_up[1].weight, 0.0)
		nn.init.constant_(self.h_up[1].bias, 0.0)


		self.l_up = nn.Sequential(
			nn.Conv1d(channels // 2, channels, 1),
			nn.BatchNorm1d(channels)
		)
		nn.init.constant_(self.l_up[1].weight, 0.0)
		nn.init.constant_(self.l_up[1].bias, 0.0)


	def forward(self, high_feat, mid_feat, low_feat):
		"""
		Args:
			high_feat : [B, C, H1, W1] (高尺度)
			mid_feat  : [B, C, H2, W2] (中尺度)
			low_feat  : [B, C, H3, W3] (低尺度)
		Returns:
			fused_out: [B, C, H1, W1] —— 最终融合特征
			attention_map: [B, 1, H1, W1] —— 用于可视化的注意力图
		"""

		B, C, H1, W1 = high_feat.shape
		H2, W2 = mid_feat.shape[2:]
		H3, W3 = low_feat.shape[2:]
		# high_feat = self.relu(self.bn_h(high_feat))
		# mid_feat = self.relu(self.bn_m(mid_feat))
		# low_feat = self.relu(self.bn_l(low_feat))

		# 展平
		high_flat = high_feat.view(B, C, H1 * W1)  # [B, C, A1]
		mid_flat = mid_feat.view(B, C, H2 * W2)  # [B, C, A2]
		low_flat = low_feat.view(B, C, H3 * W3)  # [B, C, A3]

		# 调用融合函数
		fused_flat = self.fuse(mid_flat, low_flat, size_h=(H2, W2), size_l=(H3, W3))
		fused_feat = self.fuse(high_flat, fused_flat, size_h=(H1, W1), size_l=(H2, W2))


		fused_out = fused_feat.view(B, C, H1, W1)

		# fused_out = self.relu(self.bn(self.conv_out(fused_img)))



		return fused_out

	def fuse(self, feat_h, feat_l, size_h, size_l):
		# print(feat_h.size()) #
		feat_h_t = feat_h  #.transpose(1, 2)
		feat_l_t = feat_l  #.transpose(1, 2)

		# 分别线性变换得到 Q、K、V
		q_h = self.q_proj_h(feat_h_t)# [B, C, A_h]
		k_h = self.k_proj_h(feat_h_t)
		v_h = self.v_proj_h(feat_h_t)

		q_l = self.q_proj_l(feat_l_t)
		k_l = self.k_proj_l(feat_l_t)
		v_l = self.v_proj_l(feat_l_t)

		e_h = (k_h * v_h).mean(dim=2, keepdim=True)#.softmax(dim=1)#v_h.size(2)#BxC
		feat_hh = q_h * e_h

		e_l = (k_l * v_l).mean(dim=2, keepdim=True)#.softmax(dim=1)#v_h.size(2)#BxC
		feat_ll = q_l * e_l

		feat_hl = q_h * e_l
		feat_lh = q_l * e_h

		h_att = self.h_up(feat_hh + feat_hl)
		f_h = feat_h + h_att


		l_att = self.l_up(feat_lh + feat_ll)
		f_l = feat_l + l_att


		Hh, Wh = size_h
		Hl, Wl = size_l
		B, C, _ = f_h.shape
		f_h_img = f_h.view(B, C, Hh, Wh)
		f_l_img = f_l.view(B, C, Hl, Wl)

		f_l_img_up = F.interpolate(f_l_img, size=f_h_img.shape[2:], mode='bilinear', align_corners=False)
		fused_img = f_h_img + f_l_img_up
		# fused_img = f_h_img * f_l_img_up

		fused_flat = fused_img.view(B, C, -1)
		return fused_flat


class ScaleAwareFeatureAggregation(nn.Module):  # SAFA模块

	"""
      ScaleAwareFeatureAggregation happens in five steps

      1 : Downgrade the size of two feature maps.
      2 : Converts the feature maps into query and key representaions.
      3 : Concats the downgraded features and divide it into blocks.
      4 : Performs Multi-Head Attention.
      5 : Aggregates and outputs the enhanced representation

    """

	def __init__(self, channels):
		super().__init__()

		self.mult_scale_heads = 8

		self.query_conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, stride=1,padding=3)
		self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1,padding=1)
		# self.ca = CA(channels,channels,4)
		self.key_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1,padding=1)

	def forward(self, x, quarter_scale_x):
		"""
        Args :
            x:  a list of, feature map in (C, H, W) format.
            quarter_scale_x: a list of, quarter scale feature map in (C, H/4, W/4) format.

        Returns :
            aggregated_enhanced_representation : enhanced aggregated feature representations of
                                                 two feature maps from different scales.
                                                 a list of, enhanced features in (C, H, W)

        """

		orig_x = x

		# Key Query Generation
		# x = self.ca(x)
		x = self.query_conv1(x)
		x = self.query_conv2(x)
		quarter_scale_x = self.key_conv(quarter_scale_x)
		batch_size, C, roi_h, roi_w = x.size()

		x = x.view(batch_size, 1, C, roi_h, roi_w)
		quarter_scale_x = quarter_scale_x.view(batch_size, 1, C, roi_h, roi_w)

		x = torch.cat((x, quarter_scale_x), dim=1)
		batch_size, img_n, _, roi_h, roi_w = x.size()

		# Calculating the number of attention blocks
		x_embed = x
		c_embed = x_embed.size(2)

		# Performing multi-head attention
		# (img_n, num_attention_blocks, C / num_attention_blocks, H, W)
		x_embed = x_embed.view(batch_size, img_n, self.mult_scale_heads, -1, roi_h,
							   roi_w)
		# (1, roi_n, num_attention_blocks, C / num_attention_blocks, H, W)
		target_x_embed = x_embed[:, [1]]
		# (batch_size, img_n, num_attention_blocks, 1, H, W)
		ada_weights = torch.sum(
			x_embed * target_x_embed, dim=3, keepdim=True) / (
							  float(c_embed / self.mult_scale_heads) ** 0.5)
		# (batch_size, img_n, num_attention_blocks, C / num_attention_blocks, H, W)
		ada_weights = ada_weights.expand(-1, -1, -1,
										 int(c_embed / self.mult_scale_heads),
										 -1, -1).contiguous()
		ada_weights = ada_weights.view(batch_size, img_n, c_embed, roi_h, roi_w)
		ada_weights = ada_weights.softmax(dim=1)

		# Aggregation and generation of enhanced representation
		x = (x * ada_weights).sum(dim=1)
		# upsample = nn.UpsamplingBilinear2d((orig_/x.size()[-2], /orig_x.size()[-1]))
		aggregated_feature = x #upsample(x)
		aggregated_enhanced_representation = orig_x + aggregated_feature
		return aggregated_enhanced_representation

class FeatEnHancer(nn.Module):
	"""
      Feature extraction for dark objects occurs in three stages
      1 : Extract low light features in three scales
      2 : Scale aware aggregation of low light features
      3 : Construction of enhanced representation of the input image


      Args :
           in_channels : represent the number of channels in the input image (default is 3, considering RGB image)

    """

	def __init__(self, in_channels=3):
		super(FeatEnHancer, self).__init__()

		int_out_channels = 32
		out_channels = 24

		self.relu = nn.ReLU(inplace=True)
		self.e_conv1 = nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)

		# our convolution layers to transform the concatenated feature maps into the required feature shapes
		self.ue_conv8 = nn.Conv2d(out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
		self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)

		# Convolutions for downsampling orignal image into multiple scales
		self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
		self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)

		self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=24).to('cuda')

	def forward(self, x):
		"""
        Args:
            x : a list, batched inputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                * image: Tensor, image in (C, H, W) format.

        Returns:
            enhanced_representation : a list, of enhanced representations
                                      Each representation is a Tensor, in (N, C, H, W) format.
        """

		quarter_scale_x = self.quarter_conv(x)
		hexa_scale_x = self.hexa_conv(quarter_scale_x)

		# Extracting low light featurs at three different scales
		x1 = self.relu(self.e_conv1(x))
		quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x))
		hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))

		x2 = self.relu(self.e_conv2(x1))
		quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1))
		hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))

		x3 = self.relu(self.e_conv3(x2))
		quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2))
		hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))

		x4 = self.relu(self.e_conv4(x3))
		quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3))
		hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
		quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1)))
		hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))

		x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
		quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1)))
		hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))

		x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1)))
		quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1)))
		hexa_scale_x7 = self.relu(self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1)))

		# print(x7.size(),quarter_scale_x7.size(),hexa_scale_x7.size())

		# Applying ScaleAwareFeatureAggregation between X7 and quarter_scale_x7
		x7 = self.feature_aggregation_block(x7, quarter_scale_x7)

		# Upsampling hexa scale to H x W for Skip connection
		x_upsample = nn.UpsamplingBilinear2d((x7.size()[-2], x7.size()[-1]))
		hexa_scale_x7 = x_upsample(hexa_scale_x7)



		x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))

		# Construction of Enhanced representation
		activated_enhanced_feature = torch.tanh(self.ue_conv9(x8))
		feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = torch.split(
			activated_enhanced_feature, 3, dim=1)

		enhanced_representation =  x + feature1 + feature2 + feature3 + feature4 + feature5 + feature6 + feature7 + feature8
		return enhanced_representation

	def feature_aggregation_block(self, x, quarter_scale_x):
		# Does the forward pass and returns the enhanced feature representation by aggregating both input feature maps
		"""
        Args:
            x:  a feature map in (C, H, W) format.
            quarter_scale_x:  quarter scale feature map in (C, H/4, W/4) format.

        Returns:
            aggregated_enhanced_representation : Tensor, in (C, H, W) format.
        """
		aggregated_enhanced_representation = self.scale_aware_aggregation(x, quarter_scale_x)
		return aggregated_enhanced_representation

class zjy_FeatEnHancer(nn.Module):
	"""
      Feature extraction for dark objects occurs in three stages
      1 : Extract low light features in three scales
      2 : Scale aware aggregation of low light features
      3 : Construction of enhanced representation of the input image


      Args :
           in_channels : represent the number of channels in the input image (default is 3, considering RGB image)

    """

	def __init__(self, in_channels=3):
		super(zjy_FeatEnHancer, self).__init__()

		int_out_channels = 32
		out_channels = 24

		self.relu = nn.ReLU(inplace=True)
		self.e_conv1 = nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
		self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)

		# our convolution layers to transform the concatenated feature maps into the required feature shapes
		self.ue_conv8 = nn.Conv2d(out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
		self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)

		# Convolutions for downsampling orignal image into multiple scales
		self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
		self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)

		self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=24).to('cuda')
		self.co_attention = MultiScaleCoAttention(channels=24)

	def forward(self, x):
		"""
        Args:
            x : a list, batched inputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                * image: Tensor, image in (C, H, W) format.

        Returns:
            enhanced_representation : a list, of enhanced representations
                                      Each representation is a Tensor, in (N, C, H, W) format.
        """

		quarter_scale_x = self.quarter_conv(x)
		hexa_scale_x = self.hexa_conv(quarter_scale_x)

		# Extracting low light featurs at three different scales
		x1 = self.relu(self.e_conv1(x))
		quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x))
		hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))

		x2 = self.relu(self.e_conv2(x1))
		quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1))
		hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))

		x3 = self.relu(self.e_conv3(x2))
		quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2))
		hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))

		x4 = self.relu(self.e_conv4(x3))
		quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3))
		hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
		quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1)))
		hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))

		x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
		quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1)))
		hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))

		x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1)))
		quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1)))
		hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))

		x7 = self.co_attention(x7, quarter_scale_x7, hexa_scale_x7)
		# print("x7:", x7.size())

		g = x7

		feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = torch.split(
			x7, 3, dim=1)

		tail = feature1 + feature2 + feature3 + feature4 + feature5 + feature6 + feature7 + feature8

		# x = x + feature1
		# x = x + feature2
		# x = x + feature3
		# x = x + feature4
		# x = x + feature5
		# x = x + feature6
		# x = x + feature7
		# enhanced_representation = x + feature8



		# print("enhanced_representation:", enhanced_representation.size())
		return g, tail

	def feature_aggregation_block(self, x, quarter_scale_x):
		# Does the forward pass and returns the enhanced feature representation by aggregating both input feature maps
		"""
        Args:
            x:  a feature map in (C, H, W) format.
            quarter_scale_x:  quarter scale feature map in (C, H/4, W/4) format.

        Returns:
            aggregated_enhanced_representation : Tensor, in (C, H, W) format.
        """
		aggregated_enhanced_representation = self.scale_aware_aggregation(x, quarter_scale_x)
		return aggregated_enhanced_representation


# class JQZFeatEnHancer(nn.Module):
# 	"""
#       Feature extraction for dark objects occurs in three stages
#       1 : Extract low light features in three scales
#       2 : Scale aware aggregation of low light features
#       3 : Construction of enhanced representation of the input image
#
#
#       Args :
#            in_channels : represent the number of channels in the input image (default is 3, considering RGB image)
#
#     """
#
# 	def __init__(self, in_channels=3):
# 		super(JQZFeatEnHancer, self).__init__()
#
# 		int_out_channels = 32
# 		out_channels = 24
#
# 		# self.relu = nn.ReLU(inplace=True)
#
#
#
#
# 		self.e_conv1 = nn.Sequential(
# 			nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			# nn.ReLU()
# 		)
#
# 		self.e_conv2 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv3 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv4 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),#64 32 128 256
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv5 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv6 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv7 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(out_channels,out_channels,8),
# 			nn.ReLU()
# 		)
#
#
# 		# our convolution layers to transform the concatenated feature maps into the required feature shapes
# 		self.ue_conv8 = nn.Conv2d(out_channels, int_out_channels, 3, 1, 1, bias=True)
# 		self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)
#
# 		# Convolutions for downsampling orignal image into multiple scales
# 		# self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 1,padding=3)
# 		# self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 1,padding=1)
# 		# self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=24).to('cuda')
#
# 		# self.ca = CA(24,24,8)
#
#
# 	def forward(self, x):
# 		"""
#         Args:
#             x : a list, batched inputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 * image: Tensor, image in (C, H, W) format.
#
#         Returns:
#             enhanced_representation : a list, of enhanced representations
#                                       Each representation is a Tensor, in (N, C, H, W) format.
#         """
#
#
#
# 		x1 = self.e_conv1(x)
# 		x2 = self.e_conv2(x1)
# 		x3 = self.e_conv3(x2)
# 		x4 = self.e_conv4(x3)
# 		x5 = self.e_conv5(torch.cat([x3, x4], 1))
# 		x6 = self.e_conv6(torch.cat([x2, x5], 1))
# 		x7 = self.e_conv7(torch.cat([x1, x6], 1))
#
#
# 		x8 = self.ue_conv8(x7)
# 		activated_enhanced_feature = torch.tanh(self.ue_conv9(x8))
# 		feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = torch.split(
# 			activated_enhanced_feature, 3, dim=1)
#
#
#
# 		enhanced_representation =  x + feature1 + feature2 + feature3 + feature4 + feature5 + feature6 + feature7 + feature8
#
#
# 		return enhanced_representation
#
# 	def feature_aggregation_block(self, x, quarter_scale_x):
# 		# Does the forward pass and returns the enhanced feature representation by aggregating both input feature maps
# 		"""
#         Args:
#             x:  a feature map in (C, H, W) format.
#             quarter_scale_x:  quarter scale feature map in (C, H/4, W/4) format.
#
#         Returns:
#             aggregated_enhanced_representation : Tensor, in (C, H, W) format.
#         """
# 		aggregated_enhanced_representation = self.scale_aware_aggregation(x, quarter_scale_x)
# 		return aggregated_enhanced_representation





#
# class JQZFeatEnHancer(nn.Module):
# 	"""
#       Feature extraction for dark objects occurs in three stages
#       1 : Extract low light features in three scales
#       2 : Scale aware aggregation of low light features
#       3 : Construction of enhanced representation of the input image
#
#
#       Args :
#            in_channels : represent the number of channels in the input image (default is 3, considering RGB image)
#
#     """
#
# 	def __init__(self, in_channels=3):
# 		super(JQZFeatEnHancer, self).__init__()
#
# 		int_out_channels = 32
# 		out_channels = 24
#
# 		# self.relu = nn.ReLU(inplace=True)
#
#
#
#
# 		self.e_conv1 = nn.Sequential(
# 			nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			# nn.ReLU()
# 		)
#
# 		self.e_conv2 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv3 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv4 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),#64 32 128 256
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv5 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv6 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv7 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(out_channels,out_channels,8),
# 			nn.ReLU()
# 		)
#
#
# 		# our convolution layers to transform the concatenated feature maps into the required feature shapes
# 		self.ue_conv8 = nn.Conv2d(out_channels, int_out_channels, 3, 1, 1, bias=True)
# 		self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)
#
# 		# Convolutions for downsampling orignal image into multiple scales
# 		# self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 1,padding=3)
# 		# self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 1,padding=1)
# 		# self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=24).to('cuda')
#
# 		# self.ca = CA(24,24,8)
#
#
# 	def forward(self, x):
# 		"""
#         Args:
#             x : a list, batched inputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 * image: Tensor, image in (C, H, W) format.
#
#         Returns:
#             enhanced_representation : a list, of enhanced representations
#                                       Each representation is a Tensor, in (N, C, H, W) format.
#         """
#
#
#
# 		x1 = self.e_conv1(x)
# 		x2 = self.e_conv2(x1)
# 		x3 = self.e_conv3(x2)
# 		x4 = self.e_conv4(x3)
# 		x5 = self.e_conv5(torch.cat([x3, x4], 1))
# 		x6 = self.e_conv6(torch.cat([x2, x5], 1))
# 		x7 = self.e_conv7(torch.cat([x1, x6], 1))
#
#
# 		x8 = self.ue_conv8(x7)
# 		activated_enhanced_feature = torch.tanh(self.ue_conv9(x8))
# 		feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = torch.split(
# 			activated_enhanced_feature, 3, dim=1)
#
#
#
# 		enhanced_representation =  x + feature1 + feature2 + feature3 + feature4 + feature5 + feature6 + feature7 + feature8
#
#
# 		return enhanced_representation
#
# 	def feature_aggregation_block(self, x, quarter_scale_x):
# 		# Does the forward pass and returns the enhanced feature representation by aggregating both input feature maps
# 		"""
#         Args:
#             x:  a feature map in (C, H, W) format.
#             quarter_scale_x:  quarter scale feature map in (C, H/4, W/4) format.
#
#         Returns:
#             aggregated_enhanced_representation : Tensor, in (C, H, W) format.
#         """
# 		aggregated_enhanced_representation = self.scale_aware_aggregation(x, quarter_scale_x)
# 		return aggregated_enhanced_representation
#
# class JQZFeatEnHancer(nn.Module):
# 	"""
#       Feature extraction for dark objects occurs in three stages
#       1 : Extract low light features in three scales
#       2 : Scale aware aggregation of low light features
#       3 : Construction of enhanced representation of the input image
#
#
#       Args :
#            in_channels : represent the number of channels in the input image (default is 3, considering RGB image)
#
#     """
#
# 	def __init__(self, in_channels=3):
# 		super(JQZFeatEnHancer, self).__init__()
#
# 		int_out_channels = 32
# 		out_channels = 24
#
# 		# self.relu = nn.ReLU(inplace=True)
#
#
#
#
# 		self.e_conv1 = nn.Sequential(
# 			nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			# nn.ReLU()
# 		)
#
# 		self.e_conv2 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv3 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv4 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),#64 32 128 256
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv5 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv6 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(int_out_channels,int_out_channels,8),
# 			nn.ReLU()
# 		)
#
# 		self.e_conv7 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True),
# 			# nn.ReLU(),
# 			# CA(out_channels,out_channels,8),
# 			nn.ReLU()
# 		)
#
#
# 		# our convolution layers to transform the concatenated feature maps into the required feature shapes
# 		self.ue_conv8 = nn.Conv2d(out_channels, int_out_channels, 3, 1, 1, bias=True)
# 		self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)
#
# 		# Convolutions for downsampling orignal image into multiple scales
# 		# self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 1,padding=3)
# 		# self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 1,padding=1)
# 		# self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=24).to('cuda')
#
# 		# self.ca = CA(24,24,8)
#
#
# 	def forward(self, x):
# 		"""
#         Args:
#             x : a list, batched inputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 * image: Tensor, image in (C, H, W) format.
#
#         Returns:
#             enhanced_representation : a list, of enhanced representations
#                                       Each representation is a Tensor, in (N, C, H, W) format.
#         """
#
#
#
# 		x1 = self.e_conv1(x)
# 		x2 = self.e_conv2(x1)
# 		x3 = self.e_conv3(x2)
# 		x4 = self.e_conv4(x3)
# 		x5 = self.e_conv5(torch.cat([x3, x4], 1))
# 		x6 = self.e_conv6(torch.cat([x2, x5], 1))
# 		x7 = self.e_conv7(torch.cat([x1, x6], 1))
#
#
# 		x8 = self.ue_conv8(x7)
# 		activated_enhanced_feature = torch.tanh(self.ue_conv9(x8))
# 		feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = torch.split(
# 			activated_enhanced_feature, 3, dim=1)
#
#
#
# 		enhanced_representation =  x + feature1 + feature2 + feature3 + feature4 + feature5 + feature6 + feature7 + feature8
#
#
# 		return enhanced_representation
#
# 	def feature_aggregation_block(self, x, quarter_scale_x):
# 		# Does the forward pass and returns the enhanced feature representation by aggregating both input feature maps
# 		"""
#         Args:
#             x:  a feature map in (C, H, W) format.
#             quarter_scale_x:  quarter scale feature map in (C, H/4, W/4) format.
#
#         Returns:
#             aggregated_enhanced_representation : Tensor, in (C, H, W) format.
#         """
# 		aggregated_enhanced_representation = self.scale_aware_aggregation(x, quarter_scale_x)
# 		return aggregated_enhanced_representation

class  IlluMaxPooling(nn.Module):
	def __init__(self,ksize=9,eps=1e-6,stride=4):
		super().__init__()
		self.pool = nn.MaxPool2d(kernel_size=ksize,padding=(ksize-1)//2,stride=stride)
		self.eps = eps
	def forward(self,x):
		lum = 0.299*x[:,0,:,:]+0.587*x[:,1,:,:]+0.114*x[:,2,:,:]
		mask = (lum/(lum.max()+self.eps)).unsqueeze(1)
		wx = x*mask
		pool = self.pool(wx)
		return pool



class JQZFeatEnHancer(nn.Module):		#52.3version
	"""
      Feature extraction for dark objects occurs in three stages
      1 : Extract low light features in three scales
      2 : Scale aware aggregation of low light features
      3 : Construction of enhanced representation of the input image


      Args :
           in_channels : represent the number of channels in the input image (default is 3, considering RGB image)

    """

	def __init__(self, hiddim= 32):
		super(JQZFeatEnHancer, self).__init__()

		self.fet_rgb1 = FET(hiddim=hiddim)



	def forward(self, x):

		tail = self.fet_rgb1(x)
		return tail

# class FET(nn.Module):
#
#
# 	def __init__(self, hiddim=32):
# 		super(FET, self).__init__()
#
# 		int_out_channels = hiddim
# 		self.e_conv1 = nn.Sequential(
# 			nn.Conv2d(3, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv2 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv3 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv4 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),#64 32 128 256
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv5 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv6 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv7 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
# 		self.tail = nn.Sequential(
# 			nn.Conv2d(int_out_channels, 3, 1, 1, 0, bias=True),
# 		)
# 		nn.init.constant_(self.tail[0].weight,0.0)
# 		nn.init.constant_(self.tail[0].bias,0.0)
#
# 		self.gconv1 = nn.Sequential(
# 			nn.Conv2d(in_channels=hiddim, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True),
# 		)
#
# 		self.gconv2 = nn.Sequential(
# 			nn.Conv2d(in_channels=hiddim, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True),
# 		)
#
# 		self.gconv3 = nn.Sequential(
# 			nn.Conv2d(in_channels=hiddim, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True),
# 		)
#
#
# 	def forward(self, x):
#
# 		x1 = self.e_conv1(x)
# 		x2 = self.e_conv2(x1)
# 		x3 = self.e_conv3(x2)
# 		x4 = self.e_conv4(x3)
# 		x5 = self.e_conv5(torch.cat([x3, x4], 1))
# 		x6 = self.e_conv6(torch.cat([x2, x5], 1))
# 		x7 = self.e_conv7(torch.cat([x1, x6], 1))
#
#
#
#
# 		tail = self.tail(x7)
#
# 		g3 = self.gconv3(x7)
# 		g2 = self.gconv2(x6)
# 		g1 = self.gconv1(x5)
#
# 		# g = torch.cat((g3,g2,g1),dim=1)
#
#
# 		return  g1,g2,g3,tail

# # 0628version
class FET(nn.Module):


	def __init__(self, hiddim=32):
		super(FET, self).__init__()

		int_out_channels = hiddim
		self.conv1 = nn.Sequential(
			nn.Conv2d(4,int_out_channels,kernel_size=1,padding=0,bias=True),
			nn.ReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(int_out_channels,int_out_channels,kernel_size=5,padding=2,bias=True,groups=4),
			nn.ReLU(),
		)
		self.tail = nn.Sequential(
			nn.Conv2d(int_out_channels, 3, 1, 1, 0, bias=True),
		)
		nn.init.constant_(self.tail[0].weight,0.0)
		nn.init.constant_(self.tail[0].bias,0.0)


	def forward(self, x):
		g = x.mean(dim=1,keepdim=True)
		x = torch.cat((x,g),dim=1)
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		tail = self.tail(x2)
		return  tail


# 0703version
# class FET(nn.Module):
#
#
# 	def __init__(self, hiddim=32):
# 		super(FET, self).__init__()
#
# 		int_out_channels = hiddim
# 		self.e_conv1 = nn.Sequential(
# 			nn.Conv2d(3, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv2 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv3 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv4 = nn.Sequential(
# 			nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True),#64 32 128 256
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv5 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv6 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
#
# 		self.e_conv7 = nn.Sequential(
# 			nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True),
# 			nn.ReLU(),
# 		)
# 		self.tail = nn.Sequential(
# 			nn.Conv2d(int_out_channels, 3, 1, 1, 0, bias=True),
# 		)
# 		nn.init.constant_(self.tail[0].weight,0.0)
# 		nn.init.constant_(self.tail[0].bias,0.0)
#
# 		# self.gconv = nn.Sequential(
# 		# 	nn.Conv2d(in_channels=hiddim*2, out_channels=3, kernel_size=1, padding=0, stride=1, bias=True,groups=1),
# 		# )
#
#
#
#
#
# 	def forward(self, x):
#
# 		x1 = self.e_conv1(x)
# 		x2 = self.e_conv2(x1)
# 		x3 = self.e_conv3(x2)
# 		x4 = self.e_conv4(x3)
# 		x5 = self.e_conv5(torch.cat([x3, x4], 1))
# 		x6 = self.e_conv6(torch.cat([x2, x5], 1))
# 		x7 = self.e_conv7(torch.cat([x1, x6], 1))
#
# 		tail = self.tail(x7)
#
# 		# xx = torch.cat((x5,x6),dim=1)
# 		# g = self.gconv(xx)
#
#
#
#
# 		return  tail
