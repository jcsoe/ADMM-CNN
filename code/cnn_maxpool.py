import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import sys
import time
import common_cnn as coc
import common as co
from tensorflow.python.keras.utils.np_utils import to_categorical
# import ADMM
#import torch.nn.functional as F
# 只有这个不能只更新V1
# W更新也有问题？nopool加上w就有问题

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from MLP.pytorch_version.dlADMM import common
# from MLP.pytorch_version.dlADMM.input_data import mnist, fashion_mnist
from MLP.pytorch_version.dlADMM.input_data import mnist1, fashion_mnist, load_dada, fashion_mnist1,cifar10
from MLP.pytorch_version.dlADMM import common as wang

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

channel = 7
# initialize the neural network
# 卷积-全连接-全连接
def Net(images, label):
    print("image", images.shape)
    seed_num = 13

    torch.random.manual_seed(seed=seed_num)
    w2 = torch.normal(size=[channel, 1, 5, 5], mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b2 = torch.normal(size=[channel, 1], mean=0, std=0.1, device=device)

    conv2d2 = nn.Conv2d(1, channel, 5)
    conv2d2.weight.data = w2
    b21 = b2
    conv2d2.bias.data = b21.reshape([channel])
    z2 = conv2d2(images)
    d2 = f.relu(z2)
    pool = nn.MaxPool2d(2, 2)
    a2_tu= pool(z2)

    # conv2d2 = coc.Convolution(w2, b2)
    # z2 = conv2d2.forward(images)
    print("z2:", z2.shape)
    # d2 = co.relu(z2)
    print("d2", d2.shape)
    # pool2d2 = coc.Pooling(2, 2, stride=2)
    # a2_tu, mask = pool2d2.forward(d2)
    print("a2_tu", a2_tu.shape)

    a2 = a2_tu.view(a2_tu.shape[0], -1)
    print("a2_flatten", a2.shape)
    a2_zhi = torch.transpose(a2, 0, 1)  # 这的转置是为了方便下面相乘,所以变成了（- ，batch_size）
    print("a2_trans", a2_zhi.shape)

    torch.random.manual_seed(seed=seed_num)
    w3 = torch.normal(size=(500, a2.shape[1]), mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b3 = torch.normal(size=(500, 1), mean=0, std=0.1, device=device)
    z3 = torch.matmul(w3, a2_zhi) + b3
    a3 = co.relu(z3)
    print("a3", a3.shape)

    torch.random.manual_seed(seed=seed_num)
    w5 = torch.normal(size=(10, 500), mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b5 = torch.normal(size=(10, 1), mean=0, std=0.1, device=device)
    imask = torch.eq(label, 0)
    z5 = torch.where(imask, -torch.ones_like(label), torch.ones_like(label))

    return w2, b2, z2, d2, a2_tu, a2_zhi, w3, b3, z3, a3, w5, b5, z5

def NN_OUT(images,W0, b0,W1,b1,W2,b2,act_type):
    # print('PRED', images.shape)
    #conv2d2 = coc.Convolution(W0, b0)
    #D0 = co.act_fun(conv2d2.forward(images), act_type)
    # print(D0.shape)
    #pool2d2 = coc.Pooling(2, 2, stride=2)
    #V0, mask = pool2d2.forward(D0)
    # print(V0.shape)
    b0 = b0.reshape([channel])
    conv_layer = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    conv_layer.weight.data = W0
    conv_layer.bias.data = b0
    z2 = conv_layer(images)
    d2 = f.relu(z2)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    V0 = maxpool(d2)


    a2 = V0.view(V0.shape[0], -1)
    a2_zhi = torch.transpose(a2, 0, 1)  # 这的转置是为了方便下面相乘,所以变成了（- ，batch_size）
    # print(a2_zhi.shape)

    a3 = co.act_fun(torch.matmul(W1, a2_zhi) + b1, act_type)

    V01 = torch.matmul(W2, a3) + b2
    # print(V01.shape)

    return V01

def Net1(images, label):
    seed_num = 5

    torch.random.manual_seed(seed=seed_num)
    w2 = torch.normal(size=[5, 1, 5, 5], mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b2 = torch.normal(size=[5], mean=0, std=0.1, device=device)
    conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=7)
    conv_layer.weight.data = w2
    conv_layer.bias.data = b2
    z2 = conv_layer(images)
    d2 = f.relu(z2)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    a2_tu = maxpool(d2)

    a2 = a2_tu.view(a2_tu.shape[0], -1)
    print("a2_flatten", a2.shape)

    torch.random.manual_seed(seed=seed_num)
    w3 = torch.normal(size=(1000, a2.shape[1]), mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b3 = torch.normal(size=[1000], mean=0, std=0.1, device=device)
    linear_layer = nn.Linear(in_features=a2.shape[1], out_features=1000, bias=True)
    linear_layer.weight.data = w3
    linear_layer.bias.data = b3
    z3 = linear_layer(a2)
    a3 = f.relu(z3)
    print(a3.shape)

    torch.random.manual_seed(seed=seed_num)
    w5 = torch.normal(size=(10, 1000), mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b5 = torch.normal(size=[10], mean=0, std=0.1, device=device)
    linear_layer5 = nn.Linear(in_features=1000, out_features=10, bias=True)
    linear_layer5.weight.data = w5
    linear_layer5.bias.data = b5
    z5 = linear_layer5(a3)

    return w2, b2, z2, d2, a2_tu, a2, w3, b3, z3, a3, w5, b5, z5

def NN_OUT1(images,W0, b0,W1,b1,W2,b2,act_type):
    conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)
    conv_layer.weight.data = W0
    conv_layer.bias.data = b0
    z2 = conv_layer(images)
    d2 = f.relu(z2)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    a2_tu = maxpool(d2)

    a2 = a2_tu.view(a2_tu.shape[0], -1)
    print("NN_a2_flatten", a2.shape)

    linear_layer = nn.Linear(in_features=a2.shape[1], out_features=1000, bias=True)
    linear_layer.weight.data = W1
    linear_layer.bias.data = b1
    z3 = linear_layer(a2)
    a3 = f.relu(z3)
    print("NN_a3",a3.shape)

    linear_layer5 = nn.Linear(in_features=1000, out_features=10, bias=True)
    linear_layer5.weight.data = W2
    linear_layer5.bias.data = b2
    z5 = linear_layer5(a3)
    print("NN_z5", z5.shape)

    return z5

image_num = 5000
mnist = mnist1()
# mnist = fashion_mnist1()
# mnist = cifar10()
x_train = mnist.train_loader
x_test = mnist.test_loader

dataiter = iter(x_train)
images, label = next(dataiter)
images = images[:image_num]
label = torch.tensor(to_categorical(label, num_classes=10).reshape(60000, 10)[:image_num], device=device)
label = torch.transpose(label, 0, 1)
print(images.shape)
print('image_piex', images[3000][0][16][16])
print(label.shape)
print('piex', images[244][0][15][15])

dataiter1 = iter(x_test)
images1, label1 = next(dataiter1)
# images1 = images1[:image_num]
label1 = torch.tensor(to_categorical(label1, num_classes=10).reshape(10000, 10), device=device)
label1 = torch.transpose(label1, 0, 1)
print('image_shape', images1.shape)
print('label_shape', label1.shape)

num_of_neurons = 1000
w2, b2, z2, d2, a2_tu, a2_zhi,w3, b3, z3, a3, w5, b5, z5 = Net(images, label)
# w2, b2, z2, d2, a2_tu, a2, w3, b3, z3, a3, w5, b5, z5 = Net1(images, label)
print('!')

L = 2 # 隐藏层个数
loss_type = 1
act_type = 2
# 初始化各个参数
L = L+1  # 此时 L=2
W0 = [0, w2, w3, w5]
b0 = [0, b2, b3, b5]
V0 = [0,0,0,0]
U0 = [0,0,0,0]
U0_ = [0,0,0,0]
D0 = [0,0,0,0]
W =  [0,0,0,0]
b =  [0,0,0,0]
V =  [0,0,0,0]
U =  [0,0,0,0]
U_ = [0,0,0,0]
D =  [0,0,0,0]

#bbbb = b0[1].reshape([6])
#conv = f.conv2d(images, W0[1], bbbb, stride=1, padding=0) # 函数形式的卷积
#act = f.relu(conv)
#print(act.shape)
#print('d_ori', act[66][4][10:14,10:14])
#print('d_ori', act[66][4][0:4,0:4])
NumEpoch = 20
conv2d2 = coc.Convolution(W0[1], b0[1])
D0[1] = co.act_fun(conv2d2.forward(images), act_type)
#print(D0[1].shape)
#print('d_ori', D0[1][66][4][0:4,0:4])
#print('d_ori', D0[1][66][4][10:14,10:14])
U0[1] = torch.zeros(D0[1].shape)
pool2d2 = coc.Pooling(2, 2, stride=2)
V0[1], mask = pool2d2.forward(D0[1])
v1_shape = V0[1].shape
U0_[1] = torch.zeros(V0[1].shape)

a2 = V0[1].view(V0[1].shape[0], -1)
print("a2_flatten", a2.shape)
a2_zhi = torch.transpose(a2, 0, 1)  # 这的转置是为了方便下面相乘,所以变成了（- ，batch_size）
print("a2_trans", a2_zhi.shape)

V0[2] = co.act_fun(torch.matmul(W0[2], a2_zhi)+b0[2], act_type)
U0[2] = torch.zeros(V0[2].shape)

V0[3] = torch.matmul(W0[3], V0[2])+b0[3]
U0[3] = torch.zeros(V0[3].shape)

'''
D0[1] = d2
U0[1] = torch.zeros(D0[1].shape)
V0[1] = a2_tu
v1_shape = V0[1].shape
U0_[1] = torch.zeros(V0[1].shape)

a2 = a2
print("a2_flatten", a2.shape)
#a2_zhi = torch.transpose(a2, 0, 1)  # 这的转置是为了方便下面相乘,所以变成了（- ，batch_size）
#print("a2_trans", a2_zhi.shape)

V0[2] = a3
U0[2] = torch.zeros(V0[2].shape)

V0[3] = z5
U0[3] = torch.zeros(V0[3].shape)
'''

beta = 1e-6*torch.ones(L+2, 1) # L=2, beta = [0,0,0,0]
beta[3] = 1e-5
beta[2] = 1e-4
betad = 1e-3
beta[1] = 1e-2
print(beta.shape)
print('beta',beta)
lambda1 = 1e-5
trainerr = torch.zeros(NumEpoch, 1)
loss = torch.zeros(NumEpoch, 1)
ntr = label.shape[1]
ntr1 = label1.shape[1]
acc2 = []
acc3 = []
test_acc = []

# beta1 = beta2 = 1e-7
# D0[1] = torch.tensor(D0[1],requires_grad=True)
# W0[1] = torch.tensor(W0[1],requires_grad=True)
# b0[1] = torch.tensor(b0[1],requires_grad=True)
# images = torch.tensor(images,requires_grad=True)
# D[1] = torch.tensor(D[1],requires_grad=True)
# D0[1] = torch.tensor(D0[1],requires_grad=True)
# accnote3 = 1
# accnote2 = 1
# accnote1 = 1
# a2_zhi 就相当于直的V0[1]
for i in range(NumEpoch):
    #print('image_piex1', images[3000][0][16][16])
    #print('u_ori', U0[1][66][4][10:14, 10:14])
    #print('u_ori', U0[3])
    print('i', i)
    #print('w1', W0[1][1:2])
    #print('b1', b0[1].T)
    #print('D0', D0[1][3000][2][10:14, 10:14])
    #print('w2', W0[2][800, 228:240])
    #print('b2', b0[2][228:240].T)
    print('w3', W0[3][:, 300])
    print('b3', b0[3].T)
    # 计算准确率

    ypred_train = NN_OUT(images, W0[1], b0[1], W0[2], b0[2], W0[3], b0[3], act_type)
    actual = torch.argmax(label, dim=0)
    pred = torch.argmax(ypred_train, dim=0)
    acc = torch.eq(pred, actual)
    acc = (torch.sum(acc)) / ntr
    acc2.append(float(acc))
    print('train_acc', acc)
    ypred_test = NN_OUT(images1, W0[1], b0[1], W0[2], b0[2], W0[3], b0[3], act_type)
    actual1 = torch.argmax(label1, dim=0)
    pred1 = torch.argmax(ypred_test, dim=0)
    acc1 = torch.eq(pred1, actual1)
    acc1 = (torch.sum(acc1)) / ntr1
    acc3.append(float(acc1))
    print('test_acc', acc1)

    '''
    ypred_train = NN_OUT1(images, W0[1], b0[1], W0[2], b0[2], W0[3], b0[3], act_type)
    actual = torch.argmax(label, dim=0)
    pred = torch.argmax(ypred_train, dim=1)
    acc = torch.eq(pred, actual)
    acc = (torch.sum(acc)) / ntr
    acc2.append(float(acc))
    print('train_acc', acc)
    '''
    # V[3] = co.Vout_admm_wang(label, W0[3], b0[3], V0[2], U0[3], beta[3], V0[3])
    # W[3], b[3] = co.w_out_nolambada(V0[2], V0[3], U0[3], beta[3], b0[3])# pinverse耗资过大，计算缓慢
    # b[3] = co.b_out_wang(V0[2], W0[3], V0[3], b0[3], U0[3], beta[3])
    # W[3] = co.w_out_wang(V0[2], b[3], V0[3], W0[3], U0[3], beta[3], 1)
    W[3], b[3] = co.Wout_admm(V0[2], V0[3], U0[3], beta[3], lambda1)
    W[2], b[2] = co.Whindden_admm(W0[2],b0[2], a2_zhi, V0[2], U0[2], beta[2], lambda1, act_type)
    # W[2], b[2] = W0[2], b0[2]
    # D[1] = co.d_admm_cnn2(D0[1], W0[1], b0[1], images, V0[1], U0[1], U0_[1], betad, lambda1, act_type)
    #更新W，不使用库函数卷积
    W[1], b[1] = co.Whindden_admm_cnn1(W0[1], b0[1], images, D0[1], U0[1], beta[1], lambda1, act_type)
    # W[1], b[1] = W0[1], b0[1]
    # 更新W，使用库函数卷积
    #W[1] = co.Whindden_admm_cnn(W0[1], b0[1], images, D0[1], U0[1], beta[1], lambda1, act_type)
    # b[1] = co.bhindden_admm_cnn(W[1], b0[1], images, D0[1], U0[1], beta[1], lambda1, act_type)
    #b[1] = b0[1]
    # print("差", torch.sum(W[1]-w2))
    # W[1] = W0[1]
    #
    # D[1] = co.d_admm_cnn_ori(D0[1], W[1], b[1], images, V0[1], U0[1], U0_[1], betad, lambda1, act_type)
    # D[1] = co.d_admm_cnn2(D0[1], W[1], b[1], images, V0[1], U0[1], U0_[1], betad, lambda1, act_type)
    # D[1] = co.d_admm_cnn1(D0[1], W0[1], b0[1], images, V0[1], U0[1], U0_[1], beta[1], beta[2], act_type)
    # 在反向更新的时候将D直接更新为V的上采样,实验结果证明不可以
    # D[1] = co.up_sample_d(D0[1], V0[1])
    # D[1] = co.d_admm_cnn3(D0[1], W0[1], b0[1], images, V0[1], U0[1], U0_[1], beta[1], beta[2], act_type)
    # D[1] = co.d_admm_cnn4(D0[1], W0[1], b0[1], images, V0[1], U0[1], U0_[1], beta[1], beta[2], act_type)

    # W[1], b[1] = W0[1], b0[1]
    # D[1] = D0[1]
    # conv2d2 = coc.Convolution(W[1], b[1])
    # D[1] = co.act_fun(conv2d2.forward(images), act_type)
    D[1] = co.d_admm_cnn2(D0[1], W[1], b[1], images, V0[1], U0[1], U0_[1], beta[1], lambda1, act_type, channel)
    # D[1] = co.d_admm_cnn3(D0[1], W[1], b[1], images, V0[1], U0[1], U0_[1], beta[1], beta[2], act_type)
    # D[1] = co.d_admm_cnn(W0[1], b0[1], images, V0[1], U0[1], U0_[1], beta[1], beta[2], act_type)  # 直接使用了四维数据，不用再展开
    # D[1] = co.d_admm_cnn4(D0[1], W[1], b[1], images, V0[1], U0[1], U0_[1], beta[1], beta[2], act_type)
    # a2_zhi_ = a2_zhi
    # V[1] = V0[1]
    a2_zhi_ = co.V_admm_LA_maxpool(W[2], b[2], a2_zhi, V0[2], U0_[1], U0[2], D[1], betad, beta[2], act_type)
    V[1] = torch.reshape(torch.transpose(a2_zhi_, 0, 1), v1_shape)
    V[2] = co.V2ndend_admm(W[2], b[2], W[3], b[3], a2_zhi_, V0[3], U0[2], U0[3], beta[2], beta[3], act_type)
    # V[2] = V0[2]
    V[3] = co.Vout_admm(label, W[3], b[3], V[2], U0[3], beta[3])
    # 下一行用交叉熵损失函数
    #V[3] = co.Vout_admm_wang(label, W[3], b[3], V[2], U0[3], beta[3], V0[3])

    # conv2d21 = coc.Convolution(W[1], b[1])
    baaaa = b[1].reshape([channel])
    conv2d21 = f.conv2d(images, W[1], baaaa)
    act1 = f.relu(conv2d21)
    # U[1] = U0[1] + beta[1] * (co.act_fun(conv2d21.forward(images), act_type) - D[1])
    U[1] = U0[1] + beta[1] * (act1 - D[1])
    # print('U1', U[1][3:4])
    # pool2d3 = coc.Pooling(2, 2, stride=2)
    # TEP, mask = pool2d3.forward(D[1])
    TEP = f.max_pool2d(D[1],2, 2)
    U_[1] = U0_[1] + betad * (TEP-V[1])
    U[2] = U0[2] + beta[2] * (f.relu(torch.matmul(W[2], a2_zhi_) + b[2]) - V[2])
    U[3] = U0[3] + beta[3] * (torch.matmul(W[3], V[2]) + b[3] - V[3])
    # U[1] = U0[1]
    # U_[1] = U0_[1]
    # U[2] = U0[2]

    # 只需要将U置0即可
    U0_[1] = U_[1]
    D0[1] = D[1]
    a2_zhi = a2_zhi_
    for i in range(1, 4):
        W0[i] = W[i]
        b0[i] = b[i]
        V0[i] = V[i]
        U0[i] = U[i]
    
    

    '''
        # wang
    V[3] = co.Vout_admm_wang(label, W0[3], b0[3], V0[2], U0[3], beta[3], V0[3])
    b[3] = wang.update_b(V0[2], W0[3], V[3], b0[3], U0[3], beta[3])
    W[3] = wang.update_W(V0[2], b[3], V[3], W0[3], U0[3], beta[3], 2)
    W[3] = wang.update_W(V0[2], b[3], V[3], W[3], U0[3], beta[3], 2)
    b[3] = wang.update_b(V0[2], W[3], V[3], b[3], U0[3], beta[3])
    V[3] = co.Vout_admm_wang(label, W[3], b[3], V0[2], U0[3], beta[3], V0[3])

    U[3] = U0[3] + beta[3] * (torch.matmul(W[3], V0[2]) + b[3] - V[3])

    W0[3] = W[3]
    b0[3] = b[3]
    V0[3] = V[3]
    U0[3] = U[3]
    '''
    '''
    # 直接求解且无正则化,这样仍然会三轮直接收敛，10000张直接acc=1，看起来是直接求解的问题
    W[3], b[3] = co.w_out_nolambada(V0[2], V0[3], U0[3], beta[3], b[3])
    #V[3] = co.Vout_admm(label, W[3], b[3], V0[2], U0[3], beta[3])
    V[3] = co.Vout_admm_wang(label, W[3], b[3], V0[2], U0[3], beta[3], V0[3])
    U[3] = U0[3] + beta[3] * (torch.matmul(W[3], V0[2]) + b[3] - V[3])

    W0[3] = W[3]
    b0[3] = b[3]
    V0[3] = V[3]
    U0[3] = U[3]
    '''

    '''
    # 无正则化且用二次逼近
    beta[3] = 1e-4
    W[3] = wang.update_W(V0[2], b0[3], V0[3], W0[3], U0[3], beta[3], 1)
    b[3] = wang.update_b(V0[2], W[3], V0[3], b0[3], U0[3], beta[3])
    V[3] = wang.update_zl(V0[2], W[3], b[3], label, V0[3], U0[3], beta[3])

    U[3] = U0[3] - beta[3] * (torch.matmul(W[3], V0[2]) + b[3] - V[3])

    W0[3] = W[3]
    b0[3] = b[3]
    V0[3] = V[3]
    U0[3] = U[3]
    '''

    '''
    # 只更新最后一层
    # 直接求解而且有正则化
    W[3], b[3] = co.Wout_admm(V0[2], V0[3], U0[3], beta[3], lambda1)
    V[3] = co.Vout_admm(label, W[3], b[3], V0[2], U0[3], beta[3])
    U[3] = U0[3] + beta[3] * (torch.matmul(W[3], V0[2]) + b[3] - V[3])

    W0[3] = W[3]
    b0[3] = b[3]
    V0[3] = V[3]
    U0[3] = U[3]
    '''




