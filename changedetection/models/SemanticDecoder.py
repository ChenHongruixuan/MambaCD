import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute


class SemanticDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(SemanticDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_4_semantic = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_3_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_2_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_1_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )           

        self.trans_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())


        # Smooth layer
        self.smooth_layer_3_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_0_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, features):
        feat_1, feat_2, feat_3, feat_4 = features

        '''
            Stage I
        '''
        p4 = self.st_block_4_semantic(feat_4)
       
        '''
            Stage II
        '''
        p3 = self.trans_layer_3(feat_3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3_semantic(p3)
        p3 = self.st_block_3_semantic(p3)

        '''
            Stage III
        '''
        p2 = self.trans_layer_2(feat_2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2_semantic(p2)
        p2 = self.st_block_2_semantic(p2)

        '''
            Stage IV
        '''
        p1 = self.trans_layer_1(feat_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1_semantic(p1)
        p1 = self.st_block_1_semantic(p1)
        p1 = self.smooth_layer_0_semantic(p1)
        return p1

   
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
