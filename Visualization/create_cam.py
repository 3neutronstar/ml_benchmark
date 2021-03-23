import os
import argparse
from os import path

import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from utils import load_model
from DataSet.data_load import data_loader

def create_cam(model,file_path,file_name,num_result,config):
    result_path=os.path.join(file_path,'grad_data',file_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    _,test_loader= data_loader(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=load_model(model,file_path,file_name).to(device)

    finalconv_name = 'conv3'

    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())

    model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(model.parameters())
    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (128, 128)#img size 128*128
        _, nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
    
    for i, (image_tensor, label) in enumerate(test_loader):
        print(image_tensor.size())
        image_PIL = transforms.ToPILImage()(image_tensor[0])
        image_PIL.save(os.path.join(result_path, 'img%d.png' % (i + 1)))

        image_tensor = image_tensor.to(device)
        logit, _ = model.extract_feature(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx[0].item(), probs[0].item()))
        CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
        img = cv2.imread(os.path.join(result_path, 'img%d.png' % (i + 1)))
        height, width, _ = img.shape
     
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(os.path.join(result_path, 'cam%d.png' % (i + 1)), result)
        if i + 1 == num_result:
            break
        feature_blobs.clear()