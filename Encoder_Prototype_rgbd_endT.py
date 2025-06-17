import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from baseBlocks import CBR, P2tBackbone
from options import opt

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),  # 卷积块 卷积加批量标准化 激活
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class ConvMlp(nn.Module):
    """ 使用 1x1 卷积保持空间维度的 MLP
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

#rectangular self-calibration attention (ASCA)
class ASCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=2, relu=True):
        super(ASCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather)  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc

        return out
from timm.models.layers import DropPath, to_2tuple

class DKM(nn.Module):
    """ MetaNeXtBlock 块
    参数:
        dim (int): 输入通道数.
        drop_path (float): 随机深度率。默认: 0.0
        ls_init_value (float): 层级比例初始化值。默认: 1e-6.
    """
    def __init__(
            self,
            dim,
            token_mixer=ASCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size,
                                       ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x
class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv2dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        # 或者使用以下代码，根据需要选择
        # self.conv1 = Conv2dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0] = -1
        kernel[:, :, 0, 2] = 1
        kernel[:, :, 2, 0] = 1
        kernel[:, :, 2, 2] = -1

        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        guidance = self.conv1(guidance)

        x_diff = F.conv2d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv2d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out

class CDBE(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(CDBE, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature

        return boundary_enhanced


class Conv2dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dbn, self).__init__(conv, bn)

class LKBF(nn.Module):
    def __init__(self, in_1, in_2):
        super(LKBF, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.dkm = DKM(in_1)
        self.conv_globalinfo = convblock(in_2, in_1, 1, 1, 0)


        self.cdbe_rgb = CDBE(in_channel=in_1, guidance_channels=in_1)
        self.cdbe_depth = CDBE(in_channel=in_1, guidance_channels=in_1)
        self.rgb = SqueezeAndExcitation(in_1, activation=self.activation)
        self.depth = SqueezeAndExcitation(in_1, activation=self.activation)

        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_1, in_1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_fus = convblock(2 * in_1, in_1, 3, 1, 1)
        self.conv_out = convblock(2 * in_1, in_1, 3, 1, 1)

    def forward(self, rgb, depth, global_info):
        rgb_enhanced = self.cdbe_rgb(rgb, depth)  # 用深度图指导RGB图像
        depth_enhanced = self.cdbe_depth(depth, rgb)  # 用RGB图像指导深度图像
        rgb = self.rgb(rgb_enhanced )
        depth = self.depth(depth_enhanced )
        cur_size = rgb.size()[2:]

        att_rgb = self.dkm(rgb)
        att_d = self.dkm(depth)

        xd_in = att_d + att_rgb * att_d
        xr_in = att_rgb

        bgcm_t = xd_in + torch.add(xd_in, torch.mul(xd_in, self.rt_fus(xr_in)))
        bgcm_r = xr_in + torch.add(xr_in, torch.mul(xr_in, self.rt_fus(xd_in)))

        ful_mul = torch.mul(bgcm_r, bgcm_t)
        x_in1 = torch.reshape(bgcm_r, [bgcm_r.shape[0], 1, bgcm_r.shape[1], bgcm_r.shape[2], bgcm_r.shape[3]])
        x_in2 = torch.reshape(bgcm_t, [bgcm_t.shape[0], 1, bgcm_t.shape[1], bgcm_t.shape[2], bgcm_t.shape[3]])
        x_cat = torch.cat((x_in1, x_in2), dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul, ful_max), dim=1)
        bgcm_out = self.conv_fus(ful_out)

        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))
        return self.conv_out(torch.cat((bgcm_out, global_info), 1))


# 解码器部分，接收融合后的特征，并通过逐步融合和上采样操作生成最终的融合特征图。
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sig = nn.Sigmoid()
        self.lkbf1 = LKBF(64, 128)
        self.lkbf2 = LKBF(128, 320)
        self.lkbf3 = LKBF(320, 512)
        self.lkbf4 = LKBF(512, 512)
    def forward(self, rgb_f, d_f):
        f_g = rgb_f[3] + d_f[3]

        f_4 = self.lkbf4(rgb_f[3], d_f[3], f_g)

        f_3 = self.lkbf3(rgb_f[2], d_f[2], f_4)

        f_2 = self.lkbf2(rgb_f[1], d_f[1], f_3)

        f_1 = self.lkbf1(rgb_f[0], d_f[0], f_2)

        return f_1, f_2, f_3, f_4, f_g




# 简单的封装函数，用于创建带有PVT编码器的Transformer模型实例。
def Encoder():
    model = P2tBackbone(opt.load)
    return model
class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)

    fuse = self.bn(x + g)
    return fuse


class MultiScaleGatedAttn(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))

  def forward(self,x,g):
    x_ = x.clone()
    g_ = g.clone()

    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

    multi = self.multi(x, g) # B, C, H, W

    ### Option 2 ###
    multi = self.selection(multi) # B, num_path, H, W

    attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
    #attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_

    x_att = x_att + x_
    g_att = g_att + g_
    ## Bidirectional Interaction

    x_sig = torch.sigmoid(x_att)
    g_att_2 = x_sig * g_att


    g_sig = torch.sigmoid(g_att)
    x_att_2 = g_sig * x_att

    interaction = x_att_2 * g_att_2

    projected = torch.sigmoid(self.bn(self.proj(interaction)))

    weighted = projected * x_

    y = self.conv_block(weighted)

    #y = self.bn_2(weighted + y)
    y = self.bn_2(y)
    return y

# Multi-Scale Gated Integration
class MSFGI(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi_scale_attn = MultiScaleGatedAttn(dim)
        self.conv3 = nn.Conv2d(64, 64, 1)  # Processing the third feature map

    def forward(self, f1, f2, f3):
        # Rescale f1 and f2 if needed
        f1 = F.interpolate(f1, size=f3.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, size=f3.size()[2:], mode='bilinear', align_corners=True)

        # Apply multi-scale attention on f1 and f2
        attn = self.multi_scale_attn(f1, f2)

        # Process f3 separately and combine with attention-weighted features
        f3_processed = self.conv3(f3)
        result = attn * f3_processed
        return result


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


from AEMM import AEMM

# 主模型类，整合了上述所有模块，包含两个编码器（分别处理RGB和深度图像），DHA模块和解码器，用于多模态融合任务。
class Mnet(nn.Module):
    def __init__(self,channel=32):
        super(Mnet, self).__init__()
        self.backbone = Encoder()
        self.decoder = Decoder()
        patch_size = 2
        filters = [16, 32, 64, 128, 256, 512]
        decoder_layer = 1
        self.patch_size = patch_size
        self.filters = filters
        self.aemm1 = AEMM(64, 64)
        self.aemm2 =  AEMM(128, 128)
        self.aemm3 =  AEMM(320, 320)
        self.aemm4 =  AEMM(512, 512)
        self.up_4 = convblock(512, 64, 3, 1, 1)
        self.up_3 = convblock(320, 64, 3, 1, 1)
        self.up_2 = convblock(128, 64, 3, 1, 1)
        self.up_1 = convblock(64, 64, 1, 1, 0)

        # Initialize msgi modules
        self.msfgi4_fusion = MSFGI(64)
        self.msfgi3_fusion = MSFGI(64)
        self.msfgi2_fusion = MSFGI(64)
        self.msfgi1_fusion = MSFGI(64)

        self.ful_gcm_4 = GCM(64, channel)  # Assuming 'channel' is defined elsewhere
        self.ful_gcm_3 = GCM(96, channel)
        self.ful_gcm_2 = GCM(96, channel)
        self.ful_gcm_1 = GCM(96, channel)

        self.upsample_2 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)

        self.score1_fusion = nn.Conv2d(32, 1, 1, 1, 0)
        self.score2_fusion = nn.Conv2d(32, 1, 1, 1, 0)
        self.score3_fusion = nn.Conv2d(32, 1, 1, 1, 0)
        self.score4_fusion = nn.Conv2d(32, 1, 1, 1, 0)


    def forward(self, imgs, depths):
        x_out, y_out = self.backbone(imgs, depths)
        img_1, img_2, img_3, img_4 = x_out
        dep_1, dep_2, dep_3, dep_4 = y_out

        tha1 = self.aemm1(img_1)
        tha2 = self.aemm2(dep_2)
        tha3 = self.aemm3(dep_3)
        tha4 = self.aemm4(dep_4)

        tha4 = F.interpolate(tha4, size=dep_4.shape[2:], mode='bilinear', align_corners=True)
        tha3 = F.interpolate(tha3, size=dep_3.shape[2:], mode='bilinear', align_corners=True)
        tha2 = F.interpolate(tha2, size=dep_2.shape[2:], mode='bilinear', align_corners=True)
        tha1 = F.interpolate(tha1, size=dep_1.shape[2:], mode='bilinear', align_corners=True)

        dep_4 = dep_4 * tha4
        dep_3 = dep_3 * tha3
        dep_2 = dep_2 * tha2
        dep_1 = dep_1 * tha1

        r_f_list = [img_1, img_2, img_3, img_4]
        d_f_list = [dep_1, dep_2, dep_3, dep_4]

        f1, f2, f3, f4, f_g = self.decoder(r_f_list, d_f_list)

        f_4 = self.up_4(F.interpolate(f4, f1.size()[2:], mode='bilinear', align_corners=True))
        f_3 = self.up_3(F.interpolate(f3, f1.size()[2:], mode='bilinear', align_corners=True))
        f_2 = self.up_2(F.interpolate(f2, f1.size()[2:], mode='bilinear', align_corners=True))
        f_1 = self.up_1(f1)

        msfgi4_fusion = self.msfgi4_fusion(f_2, f_3, f_4)
        msfgi3_fusion = self.msfgi3_fusion(f_1, f_2, f_3)
        msfgi2_fusion = self.msfgi2_fusion(f_4, f_3, f_2)
        msfgi1_fusion = self.msfgi1_fusion(f_3, f_2, f_1)

        x_ful_42 = self.ful_gcm_4(msfgi4_fusion)
        x_ful_3_cat = torch.cat([msfgi3_fusion, self.upsample_2(x_ful_42)], dim=1)

        x_ful_32 = self.ful_gcm_3(x_ful_3_cat)

        x_ful_2_cat = torch.cat([msfgi2_fusion, self.upsample_2(x_ful_32)], dim=1)
        x_ful_22 = self.ful_gcm_2(x_ful_2_cat)

        x_ful_1_cat = torch.cat([msfgi1_fusion, self.upsample_2(x_ful_22)], dim=1)
        x_ful_12 = self.ful_gcm_1(x_ful_1_cat)
        out1_fusion = F.interpolate(self.score1_fusion(x_ful_12), size=imgs.size()[2:], mode='bilinear', align_corners=True)
        out2_fusion = F.interpolate(self.score2_fusion(x_ful_22), size=imgs.size()[2:], mode='bilinear', align_corners=True)
        out3_fusion = F.interpolate(self.score3_fusion(x_ful_32), size=imgs.size()[2:], mode='bilinear', align_corners=True)
        out4_fusion = F.interpolate(self.score4_fusion(x_ful_42), size=imgs.size()[2:], mode='bilinear', align_corners=True)
        return out1_fusion, out2_fusion, out3_fusion, out4_fusion

