import math

import torch
import torchvision

from sigver.preprocessing.normalize import preprocess_signature
# Functions to load the CNN model
from sigver.featurelearning.models import SigNet
# Functions for plotting:
import matplotlib.pyplot as plt
import cv2
import numpy as np


plt.rcParams['image.cmap'] = 'Greys'


class SigVerfiy:
    def __init__(self, weight_path=""):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device: {}'.format(device))

        # Load the model
        state_dict, _, _ = torch.load(weight_path, map_location='cpu')
        self.model = SigNet().to(device).eval()
        self.model.load_state_dict(state_dict)

        self.resize_shape = (800, 800)          # raw img --> resize img
        self.canvas_size = (840, 1360)           # resize img --> padding img
        self.input_size = (150, 220)            # padding img --> center nomalized img
        self.threshold = 80
        self.max_distance = 70

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(img)
        # plt.show()
        return img

    def predict(self, sig1_path, sig2_path, visualize=True):
        sig1, sig2 = self.load_image(sig1_path), self.load_image(sig2_path)
        preprocessed_sig1 = torch.tensor(
            [preprocess_signature(sig1, canvas_size=self.canvas_size, resize_size=self.resize_shape)])
        preprocessed_sig2 = torch.tensor(
            [preprocess_signature(sig2, canvas_size=self.canvas_size, resize_size=self.resize_shape)])

        print(preprocessed_sig1.shape)


        sig1_input = preprocessed_sig1.view(-1, 1, self.input_size[0], self.input_size[1]).float().div(255)
        sig2_input = preprocessed_sig2.view(-1, 1, self.input_size[0], self.input_size[1]).float().div(255)
        with torch.no_grad():
            user1_feature = self.model(sig1_input)
            user2_feature = self.model(sig2_input)

        distance = torch.norm(user1_feature-user2_feature).item()
        confidence = round((self.max_distance - distance) / self.max_distance * 100,2)
        if confidence > self.threshold:
            label = "MATCH"
            print("Match | Similar score: ", distance)
        else:
            label = "NOT MATCH"
            print("Not match | Similar score: ", distance)

        concatenated = torch.cat((preprocessed_sig1[0], preprocessed_sig2[0]), 0)
        print(concatenated.shape)
        if visualize:
            f, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].imshow(preprocessed_sig1[0])
            ax[1].imshow(preprocessed_sig2[0])
            plt.show()
            self.imshow(torchvision.utils.make_grid(concatenated),
                   'Label: {} | Confidence: {:.2f}%'.format(label, confidence))
        else:
            img = torchvision.utils.make_grid(concatenated)
            npimg = img.numpy()
            # npimg = 255 - npimg
            # text =
            #     plt.text(75, 8, text, style='italic', fontweight='bold',
            #              bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
            result = np.transpose(npimg, (1, 2, 0))
            return result, label, confidence
        print("Confidence: ", confidence)


    def imshow(self, img, text=None, should_save=False):
        print(img.shape)
        npimg = img.numpy()
        # npimg = 255 - npimg
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic', fontweight='bold',
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='Greys_r')
        plt.show()


if __name__ == "__main__":
    sig1_path ="data/hao.png"
    sig2_path ="data/hao_fake.png"

    verifier = SigVerfiy(weight_path='models/signet_f_lambda_0.95.pth')
    verifier.predict(sig1_path, sig2_path, visualize=True)

    # verifier.test_preprocess("data/real_1.png")
