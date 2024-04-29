import sys
sys.path.append('/home/songjian/project/MambaCD')

import argparse
import os
import time

import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MambaCD.changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.STMambaSCD import STMambaSCD

import MambaCD.changedetection.utils_func.lovasz_loss as L
from MambaCD.changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import imageio
import numpy as np
import os
import matplotlib.pyplot as plt


ori_label_value_dict = {
    'background': (0, 0, 0),
    'low_vegetation': (0, 128, 0),
    'nvg_surface': (128, 128, 128),
    'tree': (0, 255, 0),
    'water': (0, 0, 255),
    'Building': (128, 0, 0),
    'Playground': (255, 0, 0)
}

target_label_value_dict = {
    'background': 0,
    'low_vegetation': 1,
    'nvg_surface': 2,
    'tree': 3,
    'water': 4,
    'Building': 5,
    'Playground': 6
}

def map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
    # Reverse the target_label_value_dict to get a mapping from target labels to original labels
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}
    
    # Initialize an empty 3D array for the color-mapped labels
    H, W = labels.shape
    color_mapped_labels = np.zeros((H, W, 3), dtype=np.uint8)
    
    for target_label, ori_label in target_to_ori.items():
        # Find where the label matches the current target label
        mask = labels == target_label
        
        # Map these locations to the corresponding color value
        color_mapped_labels[mask] = ori_label_value_dict[ori_label]
    
    return color_mapped_labels


class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = STMambaSCD(
            output_cd = 2, 
            output_clf = 7,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        self.deep_model = self.deep_model.cuda()

        self.change_map_T1_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'change_map_T1')
        self.change_map_T2_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'change_map_T2')

        if not os.path.exists(self.change_map_T1_saved_path):
            os.makedirs(self.change_map_T1_saved_path)
        if not os.path.exists(self.change_map_T2_saved_path):
            os.makedirs(self.change_map_T2_saved_path)


        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
            print('pretrained weight has been loaded')

        self.deep_model.eval()


    def infer(self):
        torch.cuda.empty_cache()
        dataset = SemanticChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        preds_all = []
        labels_all = []

        # vbar = tqdm(val_data_loader, ncols=50)
        for itera, data in enumerate(val_data_loader):
            acc_meter = AverageMeter()
            
            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, names = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            label_cd = label_cd.cuda().long()
            label_clf_t1 = label_clf_t1.cuda().long()
            label_clf_t2 = label_clf_t2.cuda().long()


            # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
            output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

            label_cd = label_cd.cpu().numpy()
            labels_A = label_clf_t1.cpu().numpy()
            labels_B = label_clf_t2.cpu().numpy()

            change_mask = torch.argmax(output_1, axis=1)

            preds_A = torch.argmax(output_semantic_t1, dim=1)
            preds_B = torch.argmax(output_semantic_t2, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).cpu().numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).cpu().numpy()


            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                preds_all.append(pred_A)
                preds_all.append(pred_B)
                labels_all.append(label_A)
                labels_all.append(label_B)
                acc = (acc_A + acc_B)*0.5
                acc_meter.update(acc)

            change_map_T1 = map_labels_to_colors(np.squeeze(preds_A), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)
            change_map_T2 = map_labels_to_colors(np.squeeze(preds_B), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)
            image_name = names[0][0:-4] + f'.png'

            imageio.imwrite(os.path.join(self.change_map_T1_saved_path, image_name), change_map_T1.astype(np.uint8))
            imageio.imwrite(os.path.join(self.change_map_T2_saved_path, image_name), change_map_T2.astype(np.uint8))

        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 7)
  
        print(f'Inference stage is done, SeK is {Sek}!')
            


def main():
    parser = argparse.ArgumentParser(description="Inference on SECOND dataset")
    parser.add_argument('--cfg', type=str)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SECOND/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SECOND/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='MambaSCD_Tiny')
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    infer = Inference(args)
    infer.infer()


if __name__ == "__main__":
    main()
