import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from grad_cam import GradCAM
import model.model as module_arch
from parse_config import ConfigParser


INPUT_PATH = None
OUTPUT_PATH = None


def preprocess(img):
    mean = 0.1307
    std = 0.3081
    preprocessed_img = img.copy()
    
    preprocessed_img -= mean
    preprocessed_img /= std
        
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    output = torch.tensor(preprocessed_img, requires_grad=True)
    
    return output

def show_cam_on_image(img, mask, save_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_name, np.uint8(255 * cam))
    
    cam = cam[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(np.uint8(255 * cam))

def main(config):
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    img = cv2.imread(INPUT_PATH)
    img = img[:, :, [0]]
    img = np.float32(img) / 255

    img_input = preprocess(img)
    
    # base: 3, deeper: 5, deeperer: 5, bn: 8, dropout: 5, both: 8
    chosen = '3'
    print(chosen)
    grad_cam = GradCAM(model=model, target=chosen)

    mask = grad_cam.get_cam(img_input)
    show_cam_on_image(img, mask, OUTPUT_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-i', '--input', default=None, type=str)
    parser.add_argument('-o', '--output', default=None, type=str)

    args = parser.parse_args()
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output

    config = ConfigParser.from_args(parser)

    main(config)