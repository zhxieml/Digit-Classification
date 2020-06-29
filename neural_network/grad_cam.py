import argparse

import cv2
import numpy as np
import torch


class GradCAM():
    def __init__(self, model, target):
        self._model = model
        self._target = target
        self._features = model.features
        self._maxpool = model.maxpool
        self._classifier = model.classifier

        self._gradients = list()

    def _save_gradient(self, grad):
        self._gradients.append(grad)

    def _extract_features(self, x):
        activations = list()

        for name, module in self._features._modules.items():
            x = module(x)
            
            if name == self._target:
                x.register_hook(self._save_gradient)
                activations += [x]
        
        x = self._maxpool(x)
        x = x.view(x.size(0), -1)
        x = self._classifier(x)
        
        return activations, x

    def get_cam(self, x):
        features, output = self._extract_features(x)
        one_hot = output.max()
            
        self._features.zero_grad()
        self._classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        
        grad_val = self._gradients[-1].data.numpy()
        target = features[-1] 
        target = target.data.numpy()[0, :]
        weights = np.mean(grad_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1:])
        
        for i, w in enumerate(weights): 
            cam += w * target[i, :, :] 
            
        cam = cv2.resize(cam, (28, 28))
        cam = cam - np.min(cam)
        cam = cam  / np.max(cam)
        
        return cam

def main(config):
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    img = cv2.imread('train_example.jpg')
    img = np.float32(img) / 255
    img = img[:, :, [0]]
    print(model)

    img_input = preprocess_image(img)

    # base: 3, deeperer: 5, bn: 8, dropout: 5, both: 8
    grad_cam = GradCAM(model=model, target='1')

    mask = grad_cam.get_cam(img_input)
    show_cam_on_image(img, mask)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)