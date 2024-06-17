import os
import math
import random
import numpy as np
from scipy import stats
from MambaCD.changedetection.utils_func import eval_segm as seg_acc


def read_idtxt(path):
  id_list = []
  #print('start reading')
  f = open(path, 'r')
  curr_str = ''
  while True:
      ch = f.read(1)
      if is_number(ch):
          curr_str+=ch
      else:
          id_list.append(curr_str)
          #print(curr_str)
          curr_str = ''      
      if not ch:
          #print('end reading')
          break
  f.close()
  return id_list

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def seprate_batch(dataset, batch_size):
    """Yields lists by batch"""
    num_batch = len(dataset)//batch_size+1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i+2==num_batch): batch_len = len(dataset)-(num_batch-1)*batch_size
    return(batches)

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new

# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def ImageValStretch2D(img):
    img = img*255
    #maxval = img.max(axis=0).max(axis=0)
    #minval = img.min(axis=0).min(axis=0)
    #img = (img-minval)*255/(maxval-minval)
    return img.astype(int)

def ConfMap(output, pred):
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
      for w_idx in range(w):
        n_idx = int(pred[h_idx, w_idx])
        sum = 0
        for i in range(n):
          val=output[i, h_idx, w_idx]
          if val>0: sum+=val
        conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx]/sum
        if conf[h_idx, w_idx]<0: conf[h_idx, w_idx]=0
    # print(conf)
    return conf

def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum
    
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist

def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa

def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([x for x in range(num_class)])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    
    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return kappa_n0, Fscd, IoU_mean, Sek

def SCDD_eval(pred, label, num_class):
    infer_array = np.array(pred)
    unique_set = set(np.unique(infer_array))
    assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
    label_array = np.array(label)
    assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
    hist = get_hist(infer_array, label_array, num_class)
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        
    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return Fscd, IoU_mean, Sek

def FWIoU(pred, label, bn_mode=False, ignore_zero=False):
    if bn_mode:
        pred = (pred>= 0.5)
        label = (label>= 0.5)
    elif ignore_zero:
        pred = pred-1
        label = label-1
    FWIoU = seg_acc.frequency_weighted_IU(pred, label)
    return FWIoU

def binary_accuracy(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass+1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)

    return (area_intersection, area_union)

def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass+1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)


    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist)