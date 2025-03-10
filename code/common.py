import torch
import numpy as np
# import common_cnn as cn
from MLP.pytorch_version.another import common_cnn as cn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn

def cross_entropy_with_softmax(label, zl):
    prob = softmax(zl)
    imask =torch.eq(prob,0.0)
    prob = torch.where(imask,torch.tensor(1e-10,device=device),prob)
    loss = cross_entropy(label, prob)
    return loss
def softmax(x):
    exp =torch.exp(x)
    imask =torch.eq(exp,float("inf"))
    exp = torch.where(imask,torch.exp(torch.tensor(88.6, device=device)),exp)
    return exp/(torch.sum(exp,dim=0)+1e-10)
def cross_entropy(label, prob):
    loss = -torch.sum(label * torch.log(prob))
    return loss
#return the  relu function
def relu(x):
    return torch.maximum(x, torch.tensor(0,device=device))

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    col = np.array(col)
    N, C, H, W = input_shape  # padding之前的图像大小
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 1, 2, 4, 5)

    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    num = np.zeros_like(img)
    for y in range(out_h):
        y_min = y * stride
        for x in range(out_w):
            x_min = x * stride
            # 要注意这里是 += 而非 = ，原因就是上面的那段话
            img[:, :, y_min:y_min + filter_h, x_min:x_min + filter_w] += col[:, :, y, x, :, :]
            num[:, :, y_min:y_min + filter_h, x_min:x_min + filter_w] += np.ones_like(col)[:, :, y, x, :, :]

        if stride == 2:
            num[:, :, :, -1] = 1
            num[:, :, -1, :] = 1

    ooo = img[:, :, pad:H + pad, pad:W + pad] / num[:, :, pad:H + pad, pad:W + pad]
    ooo = torch.tensor(ooo, dtype=torch.float32)
    return ooo


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # input_data = np.array(input_data)
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 向下取整
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = input_data
    # img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))
    col = torch.zeros((N, C, out_h, out_w, filter_h, filter_w))
    # x_min 和 y_min 用于界定一个滤波器作用的方块区域
    for y in range(out_h):
        y_min = y * stride
        for x in range(out_w):
            x_min = x * stride
            col[:, :, y, x, :, :] = img[:, :, y_min:y_min + filter_h, x_min:x_min + filter_w]
    col = col.permute(0, 2, 3, 1, 4, 5).reshape(N * out_h * out_w, -1)
    col = torch.tensor(col, dtype=torch.float32)
    return col


def act_fun(u,act_type):
    if act_type == 1:
        f = 1/(1+torch.exp(-u))
    else:
        f = relu(u)
    return f
def act_fun_Grad(v, act_type):
    if act_type == 1:
        z = act_fun(v, act_type)
        f = z * (1 - z)
    else:
        f = torch.where(v > 0, 1.0, 0.0)
    return f

def V2ndend_admm(W1,b1,W2,b2,V0,V2prev,U1prev,U2prev,beta1,beta2,act_type):
    temp = (U1prev + beta1 * act_fun(torch.matmul(W1, V0) + b1, act_type) -
            torch.matmul(torch.transpose(W2,0,1), (U2prev-beta2*(V2prev-b2))))
    I = torch.eye(W2.shape[1])
    temp1 = (beta1 * I + beta2 * torch.matmul(torch.transpose(W2,0,1), W2))
    temp1 = torch.inverse(temp1)
    V = torch.matmul(temp1, temp)
    return V

# 针对卷积层到全连接层之间的转化，只需要把第一个激活变成池化即可，也就是C的后面把一部分
def Vhindden_admm(W1,b1,W2,b2,V0,V1prev,V2prev,U1prev,U2prev,beta1,beta2,act_type):
    B = V2prev-U2prev/beta2
    C = act_fun(torch.matmul(W1, V0)+b1, act_type) + U1prev/beta1

    temp = torch.matmul(W2, V1prev)
    I = torch.eye(W2.shape[1])
    mu = torch.max(torch.abs(B))
    hidden = temp + b2
    G = (act_fun(hidden, act_type) - B) * (act_fun_Grad(hidden, act_type))
    temp1 = beta1*C + beta2*torch.matmul(torch.transpose(W2, 0, 1), (mu * temp / 2 - G))
    temp2 = beta1 * I + beta2 * mu * torch.matmul(torch.transpose(W2, 0, 1), W2)/2
    temp2 = torch.inverse(temp2)
    V = torch.matmul(temp2,temp1)
    return V

def Vout_admm(Y,W1,b1,V0,U1prev,beta):
    # n = int(b1.shape[0])
    # b1_ = torch.reshape(b1, (n, 1))
    V = (Y+U1prev + beta*(torch.matmul(W1, V0) + b1))/(1+beta)
    return V

def Vout_admm_wang(label, W, b, a_last, u, rho, zl_old):
    fzl = 10e10
    MAX_ITER = 500
    zl = zl_old
    lamda = 1
    zeta = zl
    eta = 4
    TOLERANCE = 1e-3
    for i in range(MAX_ITER):
        fzl_old = fzl
        #fzl = cross_entropy_with_softmax(label, zl) + rho / 2 * torch.sum(
        #    (zl - torch.matmul(W, a_last) - b + u / rho) * (zl - torch.matmul(W, a_last) - b + u / rho))
        fzl = cross_entropy_with_softmax(label, zl) + rho / 2 * torch.sum(
            (-zl + torch.matmul(W, a_last) + b + u / rho) * (-zl + torch.matmul(W, a_last) + b + u / rho))
        if abs(fzl - fzl_old) < TOLERANCE:
            break
        lamda_old = lamda
        lamda = (1 + np.sqrt(1 + 4 * lamda * lamda)) / 2
        gamma = (1 - lamda_old) / lamda
        gradients2 = (softmax(zl) - label)
        zeta_old = zeta
        zeta = (rho * (torch.matmul(W, a_last) + b - u / rho) + (zl - eta * gradients2) / eta) / (rho + 1 / eta)
        zl = (1 - gamma) * zeta + gamma * zeta_old
    return zl

def Whindden_admm(W1prev,b1prev,V0prev,V1prev,U1prev,beta,lambda1,act_type):
    #print('147', W1prev.shape)
    #print(b1prev.shape)
    W1prev = torch.cat([W1prev, b1prev], 1)
    V0prev = torch.cat([V0prev,torch.ones(1,V0prev.shape[1])], 0)
    B = V1prev-U1prev/beta

    I = torch.eye(V0prev.shape[0])
    h = torch.max(torch.abs(B))
    h = 2
    temp = torch.matmul(W1prev, V0prev)
    G = (act_fun(temp, act_type) - B) * (act_fun_Grad(temp, act_type))
    temp1 = torch.matmul(beta*(h * temp / 2 - G), torch.transpose(V0prev,0,1))
    temp2 = lambda1*I + torch.matmul(beta*h*V0prev, torch.transpose(V0prev,0,1)) / 2
    temp2 = torch.inverse(temp2)
    W_total = torch.matmul(temp1, temp2)
    W = W_total[:, 0:W_total.shape[1]-1]
    b = W_total[:, -1]  # 为什么b可以这样得到？
    b = torch.reshape(b, (b.shape[0], 1))
    return W, b

def Wout_admm(V0prev,V1prev,U1prev,beta,lambda1):
    m, n = V0prev.shape
    V0_total = torch.cat([V0prev, torch.ones(1, n)], 0)
    I = torch.eye(m+1)

    A = torch.matmul((beta*V1prev-U1prev), torch.transpose(V0_total,0,1))
    B = lambda1*I+torch.matmul(beta*V0_total, torch.transpose(V0_total,0,1))
    B = torch.inverse(B)
    W_total = torch.matmul(A, B)
    W = W_total[:, 0:W_total.shape[1]-1]
    b = W_total[:, -1]
    b = torch.reshape(b, (b.shape[0], 1))

    return W, b

# 没有正则化且直接求解
def w_out_nolambada(V0prev,V1prev,U1prev,beta, b):
    temp1 = V1prev - U1prev/beta - b
    temp2 = torch.pinverse(V0prev)
    w_new = torch.matmul(temp1, temp2)
    b_new = V1prev - U1prev/beta - torch.matmul(w_new, V0prev)
    b_new = torch.mean(b_new, dim=1)
    b_new = b_new.reshape([10, 1])
    print('b_shape', b_new.shape)
    return w_new,b_new
def eq(a, W_next, b_next, z_next, u_next,rho):
    temp = - z_next + torch.matmul(W_next, a) + b_next + u_next / rho
    res = rho / 2 * torch.sum(temp * temp)
    return res
def eq_w(a, W_next, b_next, z_next, u_next,rho):
    temp = torch.matmul(W_next, a) + b_next - z_next + u_next/rho
    temp2 = torch.transpose(a,0,1)
    res = rho * torch.matmul(temp, temp2)
    return res
def P(W_new, theta, a_last, W, b, z, u,rho):
    temp = W_new - W
    res = eq(a_last, W, b, z, u,rho) + torch.sum(eq_w(a_last, W, b, z, u,rho) * temp) + torch.sum(theta * temp * temp) / 2
    return res
def w_out_wang(a_last, b, z, W_old, u,rho, alpha=1):
    gradients = eq_w(a_last, W_old, b, z, u, rho)
    gamma = 1.5
    zeta = W_old - gradients / alpha
    while (eq(a_last, zeta, b, z, u,rho) > P(zeta, alpha, a_last, W_old, b, z, u,rho)):
        alpha = alpha * gamma
        zeta = W_old - gradients / alpha  # Learning rate decreases to 0, leading to infinity loop here.
    print('alpha', alpha)
    W = zeta
    return W
def eq_b(a, W_next, b_next, z_next,u_next, rho):
    res = torch.reshape(torch.mean(rho * (torch.matmul(W_next, a) + b_next - z_next + u_next/rho), dim=1),shape=(-1, 1))
    return res
def b_out_wang(a_last, W, z, b_old, u,rho):
    gradients = eq_b(a_last, W, b_old, z, u,rho)
    res = b_old - gradients / rho
    return res
# 勿删
def NN_output(X,W,b,act_type):
    L = len(W)  # 一共有多少层,这里其实多加了一个1,即对于三层网络，L=4
    a = []
    z = []
    for i in range(L):
        a.append(0) # a = z = [0,0,0,0]
        z.append(0)
    a[1] = X
    for j in range(1, L-1):
        z[j] = torch.matmul(W[j], a[j]) + b[j]
        a[j+1] = act_fun(z[j], act_type)
    # b_ = torch.reshape(b[L-1],(b[L-1].shape[0],1))
    out = torch.matmul(W[L-1], a[L-1]) + b[L-1]
    return out


'''
卷积
'''
# 针对卷积层到全连接层之间的转化，只需要把第一个激活变成池化即可，也就是C的后面把一部分
# 这里要把mean(d)拉直，而且要直接使用a2_zhi,更新a2_zhi之后在返回来更新img。拉直后记得转置。
# 这里U2pre也要拉直
# 这里卷积直接接输出似乎还不太一样，建议后面两层全连接
def Vhindden_admm_LA(W2,b2, V1prev,V2prev,U1prev,U2prev,d,beta1,beta2,act_type):
    B = V2prev-U2prev/beta2
    meanpol = cn.meanPooling(2,2)
    tep,_ = meanpol.forward(d)
    tep = tep.view(tep.shape[0], -1)
    tep = torch.transpose(tep, 0, 1)  # 这的转置是为了方便下面相乘,所以变成了（- ，batch_size）
    U1prev = U1prev.view(U1prev.shape[0], -1)
    U1prev = torch.transpose(U1prev, 0, 1)
    # C = act_fun(torch.matmul(W1, V0)+b1, act_type) + U1prev/beta1
    C = tep + U1prev / beta1
    # 这里改成池化即可

    # print(type(W2))
    # print(type(V1prev))
    temp = torch.matmul(W2, V1prev)
    I = torch.eye(W2.shape[1])
    mu = torch.max(torch.abs(B))
    # mu = 1
    hidden = temp + b2
    G = (act_fun(hidden, act_type) - B) * (act_fun_Grad(hidden, act_type))
    temp1 = beta1*C + beta2*torch.matmul(torch.transpose(W2, 0, 1), (mu * temp / 2 - G))
    temp2 = beta1 * I + beta2 * mu * torch.matmul(torch.transpose(W2,0,1), W2)/2
    temp2 = torch.inverse(temp2)
    V = torch.matmul(temp2,temp1)

    return V

def V_admm_LA_maxpool(W2,b2, V1prev,V2prev,U1prev,U2prev,d,beta1,beta2,act_type):
    B = V2prev-U2prev/beta2
    # maxpol = cn.Pooling(2, 2, 2)
    # tep, _ = maxpol.forward(d)
    tep = F.max_pool2d(d, 2, 2)
    #print(tep.shape)
    # print(U1prev.shape)
    tep = tep.view(tep.shape[0], -1)
    tep = torch.transpose(tep, 0, 1)  # 这的转置是为了方便下面相乘,所以变成了（- ，batch_size）
    U1prev = U1prev.view(U1prev.shape[0], -1)
    U1prev = torch.transpose(U1prev, 0, 1)
    # C = act_fun(torch.matmul(W1, V0)+b1, act_type) + U1prev/beta1
    C = tep + U1prev / beta1
    # 这里改成池化即可

    # print(type(W2))
    # print(type(V1prev))
    temp = torch.matmul(W2, V1prev)
    I = torch.eye(W2.shape[1])
    mu = torch.max(torch.abs(B))

    hidden = temp + b2
    G = (act_fun(hidden, act_type) - B) * (act_fun_Grad(hidden, act_type))
    temp1 = beta1*C + beta2*torch.matmul(torch.transpose(W2, 0, 1), (mu * temp / 2 - G))
    temp2 = beta1 * I + beta2 * mu * torch.matmul(torch.transpose(W2,0,1), W2)/2
    temp2 = torch.inverse(temp2)
    V = torch.matmul(temp2,temp1)

    return V

# 考虑没有池化的卷积
# 在这里将img形式拉直并转化维度
# 返回拉直形式
# # Vhindden_admm(W1,b1,W2,b2,V0,V1prev,V2prev,U1prev,U2prev,beta1,beta2,act_type):
def V_admm_LA_nopool(W1,b1,W2,b2,V0,V1prev,V2prev,U1prev,U2prev,beta1,beta2,act_type):
    B = V2prev - U2prev / beta2

    conv = cn.Convolution(W1,b1)
    conv1 = conv.forward(V0)
    C = act_fun(conv1, act_type) + U1prev/beta1 # 文中蓝色
    # V1prev = V1prev.reshape(V1prev.shape[0], -1)
    # V1prev = torch.transpose(V1prev, 0, 1)
    C = C.reshape(C.shape[0], -1)
    C = torch.transpose(C, 0, 1)

    temp = torch.matmul(W2, V1prev)
    I = torch.eye(W2.shape[1])
    mu = torch.max(torch.abs(B))
    hidden = temp + b2
    G = (act_fun(hidden, act_type) - B) * (act_fun_Grad(hidden, act_type)) # 文中黄色部分
    temp1 = beta1*C + beta2*torch.matmul(torch.transpose(W2, 0, 1), (mu * temp / 2 - G))
    temp2 = beta1 * I + beta2 * mu * torch.matmul(torch.transpose(W2,0,1),W2)/2
    temp2 = torch.inverse(temp2)
    V = torch.matmul(temp2,temp1)
    return V

# 针对平均池化做的上采样
def up_sample(d_out, pad=0, stride=2, pool_h=2, pool_w=2):
    # 这里如果是奇数还有问题，待会完善
    N, C, H, W = d_out.shape
    H = 2*H
    W = 2*W
    input_shape = (N, C, H, W)

    out_h = (H + 2 * pad - pool_h) // stride + 1
    out_w = (W + 2 * pad - pool_w) // stride + 1

    dout = d_out.reshape(N * C * out_h * out_w, 1)
    dout = dout/4
    dcol = dout.repeat_interleave(4,dim=1)
    dcol = dcol.reshape(N, C, out_h * out_w, pool_h * pool_w).permute(0, 2, 1, 3).reshape(N * out_h * out_w, C * pool_h * pool_w)

    dx = col2im(dcol, input_shape, pool_h, pool_w, stride, pad)

    return dcol, dx


# 还是采用上采样的形式
# 这里需要对U2prev/beta2 - V1prev进行上采样变的得到新的值
# 这里可以直接使用四维数据不用展开
def d_admm_cnn(W1,b1,V0,V1prev,U1prev,U2prev,beta1,beta2,act_type):
    temp = U2prev/beta2 - V1prev
    _, temp1 = up_sample(temp)
    conv = cn.Convolution(W1,b1)
    temp2 = 16 * beta1 * (act_fun(conv.forward(V0), act_type) + U1prev / beta1) - 4 * beta2 * temp1
    # temp2 = 16*beta1*(act_fun(torch.matmul(W1, V0)+b1, act_type) + U1prev/beta1) - 4*beta2(temp1)
    temp3 = 16*beta1 + beta2
    d = temp2/temp3
    return d

# 将均值池化用卷积表达出来
def d_admm_cnn1(d,W1,b1,V0,V1prev,U1prev,U2prev,beta1,beta2,act_type,pad=0,filter_h=2,filter_w=2,stride=2):
    N, C, W, H = d.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 向下取整
    out_w = ((W + 2 * pad - filter_w) // stride + 1)

    w_pol = 0.25 * torch.ones(size=(1, 4))
    conv = cn.Convolution(W1, b1)
    temp1 = act_fun(conv.forward(V0), act_type)
    temp1 = cn.d2col(temp1)
    U1prev = cn.d2col(U1prev)
    V1prev = V1prev.reshape(1,N*C*out_h*out_w)  # 这个不是im2col
    U2prev = U2prev.reshape(1,N*C*out_h*out_w)

    # print((beta1*temp1+U1prev).shape)
    # print(torch.matmul(torch.transpose(w_pol,0,1), (U2prev-beta2*V1prev)).shape)
    temp1 = (beta1*temp1+U1prev) - torch.matmul(torch.transpose(w_pol, 0, 1), (U2prev-beta2*V1prev))
    WI = torch.matmul(torch.transpose(w_pol, 0, 1), w_pol)
    I = torch.eye(WI.shape[1])
    temp2 = (beta1*I+beta2*WI)
    temp2 = torch.inverse(temp2)
    out = torch.matmul(temp2, temp1)
    out = cn.col2d(out, d.shape)

    return out

def up_sample_d(d_in, d_out, pad=0, stride=2, pool_h=2, pool_w=2):
    pool = cn.Pooling(2, 2, 2)
    _, mask = pool.forward(d_in)
    N, C, H, W = d_in.shape
    input_shape = (N, C, H, W)

    out_h = (H + 2 * pad - pool_h) // stride + 1
    out_w = (W + 2 * pad - pool_w) // stride + 1

    dout = d_out.reshape(N * C * out_h * out_w, 1)

    d_in = im2col(d_in, pool_h, pool_w, stride, pad)
    d_in = d_in.reshape(N, out_h * out_w, C, pool_h * pool_w).permute(0, 2, 1, 3).reshape(N * C * out_h * out_w, pool_h * pool_w)
    d_in[torch.arange(mask.shape[0]), mask] = dout[torch.arange(mask.shape[0]), 0]

    d_in = d_in.reshape(N, C, out_h * out_w, pool_h * pool_w).permute(0, 2, 1, 3).reshape(N * out_h * out_w,
                                                                                          C * pool_h * pool_w)
    dx = col2im(d_in, input_shape, pool_h, pool_w, stride, pad)

    return dx

def up_sample1(d_in, d_out, pad=0, stride=2, pool_h=2, pool_w=2):
    pool = cn.Pooling(2, 2, 2)
    _, mask = pool.forward(d_in)
    input_shape = d_in.shape
    N, C, H, W = d_in.shape
    out_h = (H + 2 * pad - pool_h) // stride + 1
    out_w = (W + 2 * pad - pool_w) // stride + 1

    dout = d_out.reshape(N * C * out_h * out_w)

    dcol = torch.zeros((N * C * out_h * out_w, pool_h * pool_w))
    dcol[torch.arange(mask.shape[0]), mask] = dout
    dcol = dcol.reshape(N, C, out_h * out_w, pool_h * pool_w).permute(0, 2, 1, 3).reshape(N * out_h * out_w, C * pool_h * pool_w)

    dx = col2im(dcol, input_shape, pool_h, pool_w, stride, pad)
    return  dx

def down_sample(d,mask,pad=0,filter_h=2,filter_w=2,stride=2):
    N, C, H, W = d.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = im2col(d, filter_h, filter_w, stride, pad)
    # col = col.reshape(N, out_h * out_w, C, self.pool_h * self.pool_w).transpose(0, 2, 1, 3).reshape(N * C * out_h * out_w, self.pool_h * self.pool_w)
    col = col.reshape(N, out_h * out_w, C, filter_h * filter_w).permute(0, 2, 1, 3).reshape(
        N * C * out_h * out_w, filter_h * filter_w)
    out = col[torch.arange(mask.shape[0]), mask]
    out = out.reshape(N, C, out_h, out_w)

    return out

def f(d,a,b,beta1,beta2):
    temp1 = beta1*torch.sum((a-d)*(a-d))/2
    temp2 = beta2*torch.sum((F.max_pool2d(d, kernel_size=2) + b)*(F.max_pool2d(d, kernel_size=2) + b))/2
    return temp1+temp2

# 想办法解决一下最大池化
# 但是对于d来说，更新完V之后进行上采样，得到新的d
#
def d_admm_cnn_ori(d,W1,b1,V0,V1prev,U1prev,U2prev,beta1,beta2,act_type):
    #conv = cn.Convolution(W1, b1)
    #a = act_fun(conv.forward(V0), act_type)
    #a = a + U1prev / beta1
    #b = -V1prev + U2prev / beta2
    print('d_ori', d[66][4][10:14, 10:14])

    pol = cn.Pooling(2, 2, 2)
    pol, mask = pol.forward(d)
    conv = cn.Convolution(W1, b1)
    temp1 = act_fun(conv.forward(V0), act_type)
    temp1 = down_sample(temp1, mask)
    print('temp1', temp1[66][4][5:7, 5:7])
    U1prev = down_sample(U1prev, mask)
    out = ((beta1*temp1+U1prev)+(beta2*V1prev-U2prev))/(beta1+beta2)
    print('out_pool', out[66][4][5:7, 5:7])
    out = up_sample_d(d, out)
    print('out', out[66][4][10:14, 10:14])
    return out
def d_admm_cnn2(d,W1,b1,V0,V1prev,U1prev,U2prev,beta1,beta2,act_type, channel):
    '''
    pol = cn.Pooling(2, 2, 2)
    pol, mask = pol.forward(d)
    a = down_sample(a, mask)
    out = (beta1*a + beta2*b)/(beta1+beta2)
    '''
    # print("Optimized d:", f(out, a, b, beta1, beta2))
    #print('d_ori', d[66][4][10:14, 10:14])
    pool = nn.MaxPool2d(2, 2, return_indices=True)
    d_pool, indices = pool(d)
    batch_size, channels, pooled_height, pooled_width = d_pool.shape
    indices_flat = indices.view(batch_size, channels, -1)

    b = b1.reshape([channel])
    conv = F.conv2d(V0, W1, b, stride=1, padding=0) # 函数形式的卷积
    act = F.relu(conv)
    act_flat = act.view(batch_size, channels, -1)
    act_pooled_flat = torch.gather(act_flat, 2, indices_flat)
    act_pooled = act_pooled_flat.view_as(d_pool)
    #print('act_pool', act_pooled[66][4][5:7, 5:7])

    U1prev_flat = U1prev.view(batch_size, channels, -1)
    U1prev_pooled_flat = torch.gather(U1prev_flat, 2, indices_flat)  # 根据索引进行gather操作
    U1prev_pooled = U1prev_pooled_flat.view_as(d_pool)
    #print(U1prev_pooled[66][4][0:2, 0:2])

    out_pool = ((beta1 * act_pooled + U1prev_pooled) + (beta2 * V1prev - U2prev)) / (beta1 + beta2)
    #print('out_pool', out_pool[66][4][5:7, 5:7])
    # indices = indices.view(-1)
    #d_pool.backward(out_pool)
    #grad_input = d.grad
    unpool1 = nn.MaxUnpool2d(2, 2)
    grad_input = unpool1(out_pool, indices, output_size=d.size())
    unpool = nn.MaxUnpool2d(2, 2)
    d_unpool = unpool(d_pool, indices, output_size=d.size())
    out = d - d_unpool + grad_input
    #print('out', out[66][4][10:14, 10:14])

    #out_pool_flat = out_pool.view(-1)
    #d_filled = d.clone()
    #print('cgaaaaaaa', d_filled[66][4][10:14, 10:14])
    #d_filled.contiguous().view(-1).scatter_(0, absolute_indices, out_pool_flat)
    #print('cgaaaaaaa', d_filled[66][4][10:14,10:14])

    #print('d_filled', d.shape)
    #d_filled = d.contiguous().view(-1).scatter(0, indices, out_pool_flat)
    #d_filled = d_filled.view_as(d)
    return out

# 最大池化
def d_admm_cnn3(d,W1,b1,V0,V1prev,U1prev,U2prev,beta1,beta2,act_type):
    conv = cn.Convolution(W1, b1)
    a = act_fun(conv.forward(V0), act_type)
    a = a + U1prev / beta1
    b = -V1prev + U2prev / beta2
    print('d_ori', f(d, a, b, beta1, beta2))

    V1prev = up_sample_d(d, V1prev)
    U2prev = up_sample1(d, U2prev)

    conv = cn.Convolution(W1, b1)
    temp1 = act_fun(conv.forward(V0), act_type)

    out = ((beta1*temp1+U1prev)+(beta2*V1prev-U2prev))/(beta1+beta2)
    print("Optimized d:", f(out, a, b, beta1, beta2))

    return out

def d_admm_cnn4(d,W1,b1,V0,V1prev,U1prev,U2prev,beta1,beta2,act_type):
    conv = cn.Convolution(W1, b1)
    a = act_fun(conv.forward(V0), act_type)
    a = a + U1prev/beta1
    b = -V1prev + U2prev/beta2

    learning_rate = 1
    # optimizer = torch.optim.SGD([d], lr=learning_rate)
    optimizer = torch.optim.Adam([d], lr=learning_rate, betas=(0.9, 0.99))
    print('d_ori', f(d, a, b, beta1, beta2))
    loss1 = 1000
    d_note = d.clone().detach()
    d_note1 = d.clone().detach()
    # 迭代优化
    num_iterations = 5
    for i in range(num_iterations):
        optimizer.zero_grad()  # 清零梯度
        loss = f(d, a, b, beta1, beta2)  # 计算目标函数
        # gradients = torch.autograd.grad(loss, d, retain_graph=True)[0]
        loss.backward(retain_graph=True)  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        print('LOSS', i, loss)
        print(loss1)
        if loss1 <= loss:
            break
        d_note1 = d_note.clone().detach()
        d_note = d.clone().detach()
        loss1 = loss
        #print(i, loss)
        #loss2 = loss1
        #loss1 = loss
        #if loss2 <= loss1:
            #print('break')
            #pass
        #nothing = f(d, a, b, beta1, beta2)
        #print(nothing)
    print("Optimized d:", f(d_note1, a, b, beta1, beta2))

    return d_note1

def Whindden_admm_cnn1(W1prev,b1prev,V0prev,d,U1prev,beta,lambda1,act_type,pad=0,stride=1):
    #conv = nn.Conv2d(1, 5, 5)
    #conv.weight.data = W1prev
    #b = b1prev.reshape([5])
    #conv.bias.data = b
    #Z = conv(V0prev)
    #Z_act = F.relu(Z)
    #myloss = My_loss()
    #loss = myloss(Z_act, d, beta, U1prev)
    # 在直接计算之前，需要把W，V，d转化为col的形式，不然不能计算
    # b的话是应该为[W1prev.shape[0],1]
    # b1prev = b1prev.reshape([W1prev.shape[0], 1])
    # d应该注意不是im2col,U1prev也是
    # 没有pool直接把d换成v即可
    FN, C, FH, FW = W1prev.shape
    N, C, H, W = V0prev.shape
    out_h = (H + 2 * pad - FH) // stride + 1
    out_w = (W + 2 * pad - FW) // stride + 1

    w_shape = W1prev.shape
    W1prev = W1prev.reshape(FN, -1)
    V0prev = torch.transpose(im2col(V0prev,5,5),0,1)
    d = d.permute(1, 0, 2, 3).reshape(FN, N*out_h*out_w)
    U1prev = U1prev.permute(1, 0, 2, 3).reshape(FN, N * out_h * out_w)

    W1prev = torch.cat([W1prev, b1prev], 1)
    V0prev = torch.cat([V0prev, torch.ones(1,V0prev.shape[1])], 0)
    B = d-U1prev/beta

    I = torch.eye(V0prev.shape[0])
    h = torch.max(torch.abs(B))
    print('h', h)
    # h=1
    temp = torch.matmul(W1prev, V0prev)
    G = (act_fun(temp, act_type) - B) * (act_fun_Grad(temp, act_type))
    temp1 = torch.matmul(beta*(h * temp / 2 - G), torch.transpose(V0prev,0,1))
    temp2 = lambda1*I + torch.matmul(beta*h*V0prev, torch.transpose(V0prev,0,1)) / 2
    temp2 = torch.inverse(temp2)
    W_total = torch.matmul(temp1, temp2)

    W = W_total[:, 0:W_total.shape[1]-1]
    b = W_total[:, -1]  # 为什么b可以这样得到？,就相当于把b加在了W后边，a后面加了一层1，那么相乘相加直接得到的就是直接加b。

    W = torch.reshape(W, w_shape)

    b = b.reshape(FN, 1)

    return W, b

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, beta, U, W, lambda1):
        a = (x-y+U/beta)*(x-y+U/beta)
        loss = torch.sum(a)
        loss = (beta/2)*loss
        print('loss', loss)

        #a = (x-y+U/beta)*(x-y+U/beta)
        #temp1 = (beta/2)*torch.sum(a)
        #temp2 = (lambda1/2)*torch.sum(W*W)
        #loss = temp1 + temp2
        #print('loss', loss)
        return loss

def Whindden_admm_cnn(W1prev,b1prev,V0prev,d,U1prev,beta,lambda1,act_type=0,pad=0,stride=1):
    conv = nn.Conv2d(1, 6, 5)
    conv.weight.data = W1prev
    b = b1prev.reshape([6])
    conv.bias.data = b
    Z = conv(V0prev)
    Z_act = F.relu(Z)

    myloss = My_loss()
    loss = myloss(Z_act, d, beta, U1prev, W1prev,lambda1)
    loss.backward()
    grad = conv.weight.grad
    # B = d - U1prev / beta
    # h = torch.max(torch.abs(B))
    h = 2
    # print('h', h)
    w_new = W1prev - grad/(h/2)
    # loss_new = myloss(Z_act, d, beta, U1prev, w_new,lambda1)
    #rint('loss_new', loss_new)
    #while loss < loss_new:
        #h=2*h
        #w_new = W1prev - grad / (h / 2)
        #loss_new = myloss(Z_act, d, beta, U1prev, w_new,lambda1)
    #print('h', h)
    return w_new

def bhindden_admm_cnn(W1prev,b1prev,V0prev,d,U1prev,beta,lambda1,act_type=0,pad=0,stride=1):
    conv = nn.Conv2d(1, 6, 5)
    conv.weight.data = W1prev
    b = b1prev.reshape([6])
    conv.bias.data = b
    Z = conv(V0prev)
    Z_act = F.relu(Z)

    myloss = My_loss()
    loss = myloss(Z_act, d, beta, U1prev,W1prev,lambda1)
    loss.backward()
    grad = conv.bias.grad
    print('b_grad', grad)
    h = 2
    b_new = b - grad/(h/2)
    print('b_new', b_new)

    return b_new


def V_admm_CNN(d1, d2, W1, b1, V0, U1, U2, beta1, beta2, act_type, pad=0, stride=1):
    # 首先修改尺寸
    FN, CH, FH, FW = W1.shape
    W1 = W1.reshape(FN, -1)

    N, CH, H, W = V0.shape
    Nn, CHh, Hh, Ww = d2.shape
    out_h = (H + 2 * pad - FH) // stride + 1
    out_w = (W + 2 * pad - FW) // stride + 1
    v0_shape = V0.shape
    V0 = torch.transpose(im2col(V0, 5, 5), 0, 1)
    pool = nn.MaxPool2d(2, 2)
    d1 = pool(d1)
    # d1 = torch.transpose(im2col(d1, 5, 5), 0, 1)
    # U1 = torch.transpose(im2col(U1, 5, 5), 0, 1)

    # d2 = d2.permute(1, 0, 2, 3).reshape(FN, N * out_h * out_w)
    # U2 = U2.permute(1, 0, 2, 3).reshape(FN, N * out_h * out_w)

    B = d2 - U2 / beta2
    B = B.permute(1, 0, 2, 3).reshape(FN, N * out_h * out_w)
    C = U1 / beta1 + d1
    C = torch.transpose(im2col(C, 5, 5), 0, 1)

    temp = torch.matmul(W1, V0)
    I = torch.eye(W1.shape[1])
    mu = torch.max(torch.abs(B))
    G = (act_fun(temp + b1, act_type) - B) * (act_fun_Grad(temp + b1, act_type))
    temp1 = beta1 * C + beta2 * torch.matmul(torch.transpose(W1, 0, 1), (mu * temp / 2 - G))
    temp2 = beta1 * I + beta2 * mu * torch.matmul(torch.transpose(W1, 0, 1), W1) / 2
    temp2 = torch.inverse(temp2)
    V = torch.matmul(temp2, temp1)

    V = col2im(torch.transpose(V,0,1), v0_shape,5,5)

    return V


if __name__ == '__main__':
    a = torch.normal(size=(2,2,2,2), mean=0, std=0.5)
    b = torch.normal(size=(2,2,4,4), mean=0, std=0.5)
    c = torch.normal(size=(2,2,4,4), mean=0, std=0.5)
    # c = up_sample1(b,a)
    pol = cn.Pooling(2, 2, 2)
    pol, mask = pol.forward(b)
    d = down_sample(c, mask)
    e = up_sample_d(b, a)
    print(b)
    print(c)
    print(d)

    print(a)
    print(e)





    # A = torch.tensor([1,0,-1])

    #print(act_fun_Grad(A,2))
    #M = torch.normal(size=(10,1), mean=0, std=0.5)
    #print(M)
    # = M.repeat(1,3)
    #print(N)

    #b = torch.normal(size=(1, 2, 3, 3), mean=0, std=0.5)
    #print(b)
    #b_ = torch.normal(size=(1, 2, 6, 6), mean=0, std=0.5)
    #c,d = up_sample(b)
    #print(c)
    #print(d)