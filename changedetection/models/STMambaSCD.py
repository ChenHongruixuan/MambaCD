import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from MambaCD.changedetection.models.ChangeDecoder import ChangeDecoder
from MambaCD.changedetection.models.SemanticDecoder import SemanticDecoder

class STMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained,  **kwargs):
        super(STMambaSCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )


        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)


    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_bcd = self.decoder_bcd(pre_features, post_features)
        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)


        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')
        
        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2
