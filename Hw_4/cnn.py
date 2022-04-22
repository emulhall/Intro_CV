import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    indices=np.arange(im_train.shape[1])
    mini_batch_x=[]
    mini_batch_y=[]

    while(len(indices)>0):
        #Randomly select a batch
        chosen=np.random.choice(indices, size=batch_size, replace=False)

        #Update mini_batch_x with those chosen
        mini_batch_x.append(im_train[:,chosen])

        #Get the mini batch from label_train
        cool_mini_batch_y=label_train[:,chosen]

        #Convert mini_batch_y to one hot
        mini_batch_y.append(np.transpose(np.eye(10)[cool_mini_batch_y.flatten()]))

        #Update indices to remove chosen indices
        f=np.isin(indices, chosen, invert=True)
        indices=indices[f]

    return np.array(mini_batch_x), np.array(mini_batch_y)


def fc(x, w, b):
    y=w@x+b
    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dw=dl_dy.T@x.T
    dl_dw=dl_dw.flatten()
    dl_dw=np.reshape(dl_dw, (1,-1))

    dl_db=dl_dy

    dl_dx=dl_dy@w

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l=np.linalg.norm(y-y_tilde)**2

    #dl_dy=(y_tilde-y)
    dl_dy=np.reshape(y_tilde-y, (1, -1))
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    #Pass X through softmax to get final output
    y_tilde=np.exp(x)/np.sum(np.exp(x))
    l=-(y*np.log(y_tilde)).sum()

    dl_dy=np.reshape(y_tilde-y, (1, -1))

    return l, dl_dy

def relu(x):
    y=x*(x>0)
    return y


def relu_backward(dl_dy, x, y):
    dl_dx=dl_dy*(x>0)
    return dl_dx


def conv(x, w_conv, b_conv):
    #Pad image to maintain size
    x_pad=np.pad(x, ((1,1), (1,1), (0,0)))
    y=np.zeros((14,14,w_conv.shape[3]))

    for u in range(x.shape[1]):
        for v in range(x.shape[0]):
            temp=np.tile(np.reshape(x_pad[v:v+3, u:u+3,:], (3,3,1,1)), (3))
            y[u,v,:]=np.reshape(np.sum(w_conv*temp, axis=(0,1)), (1,1,3))
    y+=b_conv
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    dl_db=np.sum(dl_dy, axis=(0,1))
    x_pad=np.pad(x, ((1,1), (1,1), (0,0)))

    dl_dw=np.zeros((3,3,3))
    for k in range(dl_dy.shape[0]):
        for l in range(dl_dy.shape[1]):
            dl_dw=dl_dw+dl_dy[k,l,:]*x_pad[k:k+3,l:l+3,:]

    np.reshape(dl_dw, (3,3,1,3))
    return dl_dw, dl_db

def pool2x2(x):
    y=np.zeros((math.ceil(x.shape[0]/2), math.ceil(x.shape[1]/2),x.shape[2]))

    for u in range(0,y.shape[1]):
        for v in range(0,y.shape[0]):
            y[v,u,:]=np.amax(x[2*v:2*(v+1),2*u:2*(u+1)],axis=(0,1))
    return y

def pool2x2_backward(dl_dy, x, y):
    dl_dx=np.zeros(x.shape)
    for u in range(0,y.shape[1]):
        for v in range(0,y.shape[0]):
            indices=np.argwhere(x[2*v:2*(v+1),2*u:2*(u+1)]==y[v,u,:])
            for idx in indices:
                dl_dx[idx[0],idx[1],idx[2]]=dl_dy[u,v,idx[2]]
    return dl_dx


def flattening(x):
    y=x.flatten(order='F')
    y=np.reshape(y, (-1,1),order='F')
    return y


def flattening_backward(dl_dy, x, y):
    dl_dx=np.reshape(dl_dy, x.shape, order='F')
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    np.random.seed(1) 
    #Set learning rate
    lr=0.1

    #Set the decay rate
    dr=0.9

    #Initialize weights with a Gaussian noise
    w=np.random.normal(loc=0, scale=1, size=(10, 196))

    #Initialize bias
    b=np.zeros((10,1))

    k=0

    nIters=1000
    for n in range(nIters):
        #At every 1000th iteration decay learning rate
        if(n%1000==0 and n!=0):
            lr=dr*lr

        dL_dw=np.zeros(w.shape)
        dL_db=np.zeros(b.shape)
        for i in range(mini_batch_x[k].shape[1]):
            x=np.reshape(mini_batch_x[k,:,i], (-1,1))
            y=np.reshape(mini_batch_y[k,:,i], (-1,1))
            #Make a prediction
            y_tilde=fc(x, w, b)

            #Compute loss
            l, dl_dy=loss_euclidean(np.reshape(y_tilde, (-1)), np.reshape(mini_batch_y[k,:,i], (-1)))

            #Gradiant back prop
            dl_dx, dl_dw, dl_db=fc_backward(dl_dy, x, w, b, y_tilde)

            #Update gradients
            dL_dw=dL_dw+np.reshape(dl_dw, (dL_dw.shape))
            dL_db=dL_db+np.reshape(dl_db,(dL_db.shape))

        #k++ (Set k = 0 if k is greater than the number of mini-batches.)
        k+=1
        if(k>=len(mini_batch_x)):
            k=0


        #Update weights and bias
        w=w-(lr/mini_batch_x[k].shape[1])*dL_dw
        b=b-(lr/mini_batch_x[k].shape[1])*dL_db

    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    np.random.seed(0) 
    #Set learning rate
    lr=0.2

    #Set the decay rate
    dr=0.8

    #Initialize weights with a Gaussian noise
    w=np.random.normal(loc=0, scale=1, size=(10, 196))

    #Initialize bias
    b=np.zeros((10,1))

    k=0

    nIters=5000
    for n in range(nIters):
        #At every 1000th iteration decay learning rate
        if(n%1000==0 and n!=0):
            lr=dr*lr

        dL_dw=np.zeros(w.shape)
        dL_db=np.zeros(b.shape)

        for i in range(mini_batch_x[k].shape[1]):
            x=np.reshape(mini_batch_x[k,:,i], (-1,1))
            y=np.reshape(mini_batch_y[k,:,i], (-1,1))

            #Compute loss
            y_tilde=fc(x, w, b)
            l, dl_dy=loss_cross_entropy_softmax(y_tilde,y)

            #Gradiant back prop
            dl_dx, dl_dw, dl_db=fc_backward(dl_dy, x, w, b, y_tilde)

            #Update gradients
            dL_dw=dL_dw+np.reshape(dl_dw, (dL_dw.shape))
            dL_db=dL_db+np.reshape(dl_db,(dL_db.shape))

        #k++ (Set k = 0 if k is greater than the number of mini-batches.)
        k+=1
        if(k>=len(mini_batch_x)):
            k=0

        #Update weights and bias
        w=w-(lr/mini_batch_x[k].shape[1])*dL_dw
        b=b-(lr/mini_batch_x[k].shape[1])*dL_db

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    np.random.seed(1)
    #Set learning rate
    lr=0.3

    #Set the decay rate
    dr=0.9

    #Initialize weights with a Gaussian noise
    w1=np.random.normal(loc=0, scale=1, size=(30, 196))
    w2=np.random.normal(loc=0, scale=1, size=(10, 30))

    #Initialize bias
    b1=np.zeros((30,1))
    b2=np.zeros((10,1))

    k=0

    nIters=5000
    for n in range(nIters):
        #At every 1000th iteration decay learning rate
        if(n%1000==0 and n!=0):
            lr=dr*lr

        dL_dw1=np.zeros(w1.shape)
        dL_dw2=np.zeros(w2.shape)
        dL_db1=np.zeros(b1.shape)
        dL_db2=np.zeros(b2.shape)

        for i in range(mini_batch_x[k].shape[1]):
            x=np.reshape(mini_batch_x[k,:,i], (-1,1))
            y=np.reshape(mini_batch_y[k,:,i], (-1,1))

            #Make prediction
            x1=fc(x,w1,b1)
            x1_relu=relu(x1)
            x2=fc(x1_relu,w2,b2)

            #Compute loss
            l, dl_dy=loss_cross_entropy_softmax(x2,y)

            #Gradiant back prop
            dl_dx2, dl_dw2, dl_db2=fc_backward(dl_dy, x1_relu, w2, b2, x2)
            dl_dx1=relu_backward(dl_dx2, x1.T, x1_relu)
            dl_dx0, dl_dw1, dl_db1=fc_backward(dl_dx1, x, w1, b1, x1)

            #Update gradients
            dL_dw2=dL_dw2+np.reshape(dl_dw2, (dL_dw2.shape))
            dL_db2=dL_db2+np.reshape(dl_db2,(dL_db2.shape))
            dL_dw1=dL_dw1+np.reshape(dl_dw1, (dL_dw1.shape))
            dL_db1=dL_db1+np.reshape(dl_db1,(dL_db1.shape))

        #k++ (Set k = 0 if k is greater than the number of mini-batches.)
        k+=1
        if(k>=len(mini_batch_x)):
            k=0

        #Update weights and bias
        w1=w1-(lr/mini_batch_x[k].shape[1])*dL_dw1
        b1=b1-(lr/mini_batch_x[k].shape[1])*dL_db1
        w2=w2-(lr/mini_batch_x[k].shape[1])*dL_dw2
        b2=b2-(lr/mini_batch_x[k].shape[1])*dL_db2

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    np.random.seed(2)
    #Set learning rate
    lr=0.9

    #Set the decay rate
    dr=0.9

    #Initialize weights with a Gaussian noise
    w_conv=np.random.normal(loc=0.0, scale=1.0, size=(3,3,1,3))
    w_fc=np.random.normal(loc=0.0, scale=1.0, size=(10,147))

    #Initialize bias
    b_conv=np.zeros((3))
    b_fc=np.zeros((10,1))

    k=0

    nIters=20000
    for n in range(nIters):
        #At every 1000th iteration decay learning rate
        if(n%1000==0 and n!=0):
            lr=dr*lr

        dL_dw_conv=np.zeros(w_conv.shape)
        dL_dw_fc=np.zeros(w_fc.shape)
        dL_db_conv=np.zeros(b_conv.shape)
        dL_db_fc=np.zeros(b_fc.shape)

        for i in range(mini_batch_x[k].shape[1]):
            x=np.reshape(mini_batch_x[k,:,i], (14,14,1), order='F')
            y=np.reshape(mini_batch_y[k,:,i], (-1,1), order='F')

            #Make prediction
            #Convolutional layer
            conv_out=conv(x, w_conv, b_conv)

            #Relu activation
            relu_out=relu(conv_out)

            #pooling layer
            pool_out=pool2x2(relu_out)

            #Flatten
            flat_out=flattening(pool_out)

            #FC
            fc_out=fc(flat_out, w_fc, b_fc)

            #Compute loss
            l, dl_dy=loss_cross_entropy_softmax(fc_out,y)

            #Gradiant back prop
            #Backprop fc
            dl_dx_flat, dl_dw_fc, dl_db_fc=fc_backward(dl_dy,flat_out, w_fc, b_fc, fc_out)

            #Backprop flattening
            dl_dx_pool=flattening_backward(dl_dx_flat,pool_out, flat_out)

            #Backprop pooling
            dl_dx_relu=pool2x2_backward(dl_dx_pool,relu_out, pool_out)

            #Backprop relu
            dl_dx_conv=relu_backward(dl_dx_relu, conv_out, relu_out)

            #Backprop conv
            dl_dw_conv, dl_db_conv=conv_backward(dl_dx_conv, x, w_conv, b_conv, conv_out)

            #Update gradients
            dL_dw_conv=dL_dw_conv+np.reshape(dl_dw_conv, (dL_dw_conv.shape))
            dL_db_conv=dL_db_conv+np.reshape(dl_db_conv,(dL_db_conv.shape))
            dL_dw_fc=dL_dw_fc+np.reshape(dl_dw_fc, (dL_dw_fc.shape))
            dL_db_fc=dL_db_fc+np.reshape(dl_db_fc,(dL_db_fc.shape))

        #k++ (Set k = 0 if k is greater than the number of mini-batches.)
        k+=1
        if(k>=len(mini_batch_x)):
            k=0


        #Update weights and bias
        w_conv=w_conv-(lr/mini_batch_x[k].shape[1])*dL_dw_conv
        b_conv=b_conv-(lr/mini_batch_x[k].shape[1])*dL_db_conv
        w_fc=w_fc-(lr/mini_batch_x[k].shape[1])*dL_dw_fc
        b_fc=b_fc-(lr/mini_batch_x[k].shape[1])*dL_db_fc



    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



