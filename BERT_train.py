import random
import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim
import torchtext
import sys

from dataloader import get_chABSA_DataLoaders_and_TEXT
from bert import BertTokenizer
from bert import get_config, BertModel,BertForchABSA, set_learned_params

train_dl, val_dl, TEXT, dataloaders_dict= get_chABSA_DataLoaders_and_TEXT(max_length=256, batch_size=32)

# モデル設定のJOSNファイルをオブジェクト変数として読み込みます
config = get_config(file_path="./weights/bert_config.json")

# BERTモデルを作成します
net_bert = BertModel(config)

# BERTモデルに学習済みパラメータセットします
net_bert = set_learned_params(
    net_bert, weights_path="./weights/pytorch_model.bin")

# モデル構築
net = BertForchABSA(net_bert)

# 訓練モードに設定
net.train()

print('ネットワーク設定完了')



# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

# 1. まず全部を、勾配計算Falseにしてしまう
for name, param in net.named_parameters():
    param.requires_grad = False

# 2. 最後のBertLayerモジュールを勾配計算ありに変更
for name, param in net.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

# 3. 識別器を勾配計算ありに変更
for name, param in net.cls.named_parameters():
    param.requires_grad = True

    # 最適化手法の設定

# BERTの元の部分はファインチューニング
optimizer = optim.Adam([
    {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': net.cls.parameters(), 'lr': 5e-5}
], betas=(0.9, 0.999))

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算

# 学習・検証を実施
from train import train_model

# 学習・検証を実行する。
# エポック数
num_epochs = 50
net_trained = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)



"""
# 学習したネットワークパラメータを保存します
save_path = './weights/bert_fine_tuning_chABSA_test.pth'
dict(torch.save(net_trained.state_dict(), save_path))
"""