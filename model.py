from Pillow import Image

from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

classes_ja = ["ソーダ流紋岩","安山岩", "粗粒玄武岩"]
classes_en = ["rhyolite_rock","andesite_rock", "xuanwu_rock"]
n_class = len(classes_ja)
img_size=(100,180,3)#(100,180,3)

#画像認識のモデル
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,8,3)
        self.conv2 = nn.Conv2d(8,16,3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3)
        self.conv4 = nn.Conv2d(32,64,3)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(193600, 256)#(64*4*4,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256,3)#(256,10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        #x = F.relu(self.conv3(x))
        #x = F.relu(self.bn2(self.conv4(x)))
        #x = self.pool(x)
        x = x.view(x.size(0), -1)#(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()

#訓練済みパラメータの読み込みと設定
net.load_state_dict(torch.load("model_cnn_rock_3_Frequency_2.pth"))#,map_location=torch.device("cpu"))#"/content/drive/MyDrive/nose/cnn_pytorch//model_cnn.pth"
net.to(torch.device("cpu"))

def predict(img):
    #モデルへの入力
    img = img.resize((224,224))#サイズを変換img_size, img_size(100,180)
    #img = img.convert("L")#モノクロに変換
    img = img.convert("RGB")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#((0.0),(1.0))
    ])
    img = transform(img)
    x = img.reshape(1,3,224,224)#(1, 1, img_size, img_size)(1, 1, 100, 180)

    #予測
    net.eval()
    y = net(x)

    #結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y).cpu(), dim=0) #確率で表す
    #y_prob = torch.nn.functional.softmax(torch.squeeze(y))#確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending = True)#降順にソート
    return [(classes_ja[idx], classes_en[idx],prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
