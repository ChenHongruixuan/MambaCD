import sys
sys.path.append('/home/songjian/project/MambaCD')

import argparse
import os
import time

import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader, DamageAssessmentDatset
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.STMambaBDA import STMambaBDA

import MambaCD.changedetection.utils_func.lovasz_loss as L

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)

        self.deep_model = STMambaBDA(
            output_building=2, output_damage=5, 
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
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

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

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue
            # labels_clf[labels_clf == 0] = 255
            
            output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()

            ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
            lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
            
            ce_loss_clf = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
            lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
            final_loss = ce_loss_loc + ce_loss_clf + (0.5 * lovasz_loss_loc + 0.75 * lovasz_loss_clf)
            # final_loss = main_loss

            final_loss.backward()

            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, localization loss is {ce_loss_loc + lovasz_loss_loc}, classification loss is {ce_loss_clf + lovasz_loss_clf}')
                if (itera + 1) % 750 == 0:
                    self.deep_model.eval()
                    loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score = self.validation()
                    if oaf1 > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_kc = oaf1
                        best_round = [loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()


                # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                
                output_clf = output_clf[labels_loc > 0]
                labels_clf = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf, output_clf)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_socore()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1
        print(f'lofF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, oaF1 is {oaf1}, '
              f'sub class F1 score is {damage_f1_score}')
        return loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score


def main():
    parser = argparse.ArgumentParser(description="Training on xBD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)

    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/train')
    parser.add_argument('--train_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/test')
    parser.add_argument('--test_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/val_all.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='STMambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
