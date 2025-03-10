import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.db = None
        self.dW = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        '''
         col = im2col(x, FH, FW, self.stride, self.pad)
        # col_W = self.W.reshape(FN, -1).T
        col_W = torch.transpose(self.W.reshape(FN, -1), 0, 1)
        # print(col.dtype)
        # print(col_W.dtype)
        '''

        col = torch.transpose(im2col(x, FH, FW, self.stride, self.pad),0,1)
        col_W = self.W.reshape(FN, -1)
        # out = (np.dot(col, col_W) + self.b).reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        # out = (torch.matmul(col_W, col) + self.b).reshape(N, out_h, out_w, FN).permute(0, 3, 1, 2)

        out = (torch.matmul(col_W, col) + self.b).reshape(FN, N, out_h, out_w).permute(1,0,2,3)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = dout.sum(axis=0)
        # self.dW = np.dot(self.col.T, dout)
        self.dW = torch.matmul(self.col.T, dout)
        self.dW = self.dW.T.reshape(FN, C, FH, FW)

        # dcol = np.dot(dout, self.col_W.T)
        dcol = torch.matmul(dout, self.col_W.T)
        dcol = np.array(dcol)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.mask = None

    def forward(self, x):
        N, C, H, W = x.shape

        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # col = col.reshape(N, out_h * out_w, C, self.pool_h * self.pool_w).transpose(0, 2, 1, 3).reshape(N * C * out_h * out_w, self.pool_h * self.pool_w)
        col = col.reshape(N, out_h * out_w, C, self.pool_h * self.pool_w).permute(0, 2, 1, 3).reshape(
            N * C * out_h * out_w, self.pool_h * self.pool_w)

        # mask = np.argmax(col, axis=1)
        # out = col[np.arange(mask.size), mask]
        mask = torch.argmax(col, dim=1)  # dim给定的定义是：the demention to reduce.也就是把dim这个维度的，变成这个维度的最大值的index。
        out = col[torch.arange(mask.shape[0]), mask]
        out = out.reshape(N, C, out_h, out_w)

        self.mask = mask
        self.input_shape = x.shape

        return out, mask

    def backward(self, dout):
        N, C, H, W = self.input_shape

        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        dout = dout.reshape(N * C * out_h * out_w)
        # dcol = np.zeros((N * C * out_h * out_w, self.pool_h * self.pool_w))
        # dcol[np.arange(self.mask.size), self.mask] = dout
        dcol = torch.zeros((N * C * out_h * out_w, self.pool_h * self.pool_w))
        dcol[torch.arange(self.mask.size), self.mask] = dout

        dcol = dcol.reshape(N, C, out_h * out_w, self.pool_h * self.pool_w).transpose(0, 2, 1, 3).reshape(
            N * out_h * out_w, C * self.pool_h * self.pool_w)
        dcol = np.array(dcol)
        dx = col2im(dcol, self.input_shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

class meanPooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.mask = None

    def forward(self, x):
        N, C, H, W = x.shape

        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # col = col.reshape(N, out_h * out_w, C, self.pool_h * self.pool_w).transpose(0, 2, 1, 3).reshape(N * C * out_h * out_w, self.pool_h * self.pool_w)
        col = col.reshape(N, out_h * out_w, C, self.pool_h * self.pool_w).permute(0, 2, 1, 3).reshape(
            N * C * out_h * out_w, self.pool_h * self.pool_w)

        # mask = np.argmax(col, axis=1)
        # out = col[np.arange(mask.size), mask]
        mask = torch.argmax(col, dim=1)  # dim给定的定义是：the demention to reduce.也就是把dim这个维度的，变成这个维度的最大值的index。
        out = col.mean(dim=1)
        #print(col)
        #print(out)
        # out = col[torch.arange(mask.shape[0]), mask]
        out = out.reshape(N, C, out_h, out_w)

        self.mask = mask
        self.input_shape = x.shape

        return out, mask

    def backward(self, dout):
        N, C, H, W = self.input_shape

        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        dout = dout.reshape(N * C * out_h * out_w)
        # dcol = np.zeros((N * C * out_h * out_w, self.pool_h * self.pool_w))
        # dcol[np.arange(self.mask.size), self.mask] = dout
        dcol = torch.zeros((N * C * out_h * out_w, self.pool_h * self.pool_w))
        dcol[torch.arange(self.mask.size), self.mask] = dout

        dcol = dcol.reshape(N, C, out_h * out_w, self.pool_h * self.pool_w).transpose(0, 2, 1, 3).reshape(
            N * out_h * out_w, C * self.pool_h * self.pool_w)
        dcol = np.array(dcol)
        dx = col2im(dcol, self.input_shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

def d2col(img, pad=0, filter_h=2, filter_w=2, stride=2):
    N, C, W, H = img.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 向下取整
    out_w = ((W + 2 * pad - filter_w) // stride + 1)

    # img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))
    col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))
    # x_min 和 y_min 用于界定一个滤波器作用的方块区域
    for y in range(out_h):
        y_min = y * stride
        for x in range(out_w):
            x_min = x * stride
            col[:, :, y, x, :, :] = img[:, :, y_min:y_min + filter_h, x_min:x_min + filter_w]
    col = col.transpose(4, 5, 0, 1, 2, 3).reshape(filter_h * filter_h, -1)
    col = torch.tensor(col, dtype=torch.float32)
    return col

def col2d(col,input_shape,pad=0, filter_h=2, filter_w=2, stride=2):
    col = np.array(col)
    N, C, H, W = input_shape  # padding之前的图像大小
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(filter_h, filter_w, N, C, out_h, out_w,).transpose(2, 3, 4, 5, 0, 1)

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


def meanpool_cnn(img, pad=0, filter_h=2, filter_w=2, stride=2):
    N, C, W, H = img.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 向下取整
    out_w = ((W + 2 * pad - filter_w) // stride + 1)

    col =d2col(img)

    w_col=0.25 * torch.ones(size=(1, 4))
    out = torch.matmul(w_col,col)
    print(out.shape)
    out = out.reshape((N, C, out_h, out_w))

    return out

def maxpool(img, pad=0, filter_h=2, filter_w=2, stride=2):
    N, C, W, H = img.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 向下取整
    out_w = ((W + 2 * pad - filter_w) // stride + 1)

    # col = d2col(img)

    col = im2col(img, filter_h, filter_w, stride, pad)
    # col = col.reshape(N, out_h * out_w, C, self.pool_h * self.pool_w).transpose(0, 2, 1, 3).reshape(N * C * out_h * out_w, self.pool_h * self.pool_w)
    col = col.reshape(N, out_h * out_w, C, filter_h * filter_w).permute(0, 2, 1, 3).reshape(
        N * C * out_h * out_w, filter_h * filter_w)
    col_w = np.zeros_like(col)
    # print(col)
    mask = torch.argmax(col, dim=1)  # dim给定的定义是：the demention to reduce.也就是把dim这个维度的，变成这个维度的最大值的index。
    out = col[torch.arange(mask.shape[0]), mask]
    out = out.reshape(N, C, out_h, out_w)
    col_w[torch.arange(mask.shape[0]), mask] = 1
    col_w = torch.tensor(col_w)
    #col_w = (col_w.reshape(N, C, out_h * out_w, filter_h * filter_w))
    #col_w = torch.tensor(col_w).permute(0, 2, 1, 3)
    #col_w = col2im(col_w,(N, C, W, H),filter_h, filter_w, stride, pad)

    return out, col_w


if __name__ =='__main__':
    a = torch.normal(size=(2,2,4,4),mean=0,std=0.5)
    print(a)
    pol, colw = maxpool(a)
    print(pol)
    print(colw)

    temp1 = d2col(a)
    print(temp1)
    c = torch.matmul(colw,temp1)
    print(c)
    # c = c.reshape(2, 2 * 2, 2, 2 * 2)

    # c = col2im(c, pol.shape,2,2,2)
    # c = c.reshape(pol.shape)
    # print(c)

    # print(colw)
    #pol = meanpool_cnn(a)
    #(pol)
    #pol1 = meanPooling(2,2)
    #pol2,_ = pol1.forward(a)
    #print(pol2)
    #b = d2col(a)
    #d = col2d(b,a.shape)
    #print(b)
    #print(d)

    # print(a.reshape(2,32))

