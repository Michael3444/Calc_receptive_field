# -*- coding:utf-8 -*-
# author: lipengkun
# time: 2019/1/22



# [filter size, stride, padding]
# Assume the two dimensions are the same
# Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
# Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

from __future__ import division
import math


def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    # actual padding
    actualP = (n_out - 1) * s - n_in + k
    # right padding
    pR = math.ceil(actualP / 2)
    # left padding
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print layer_name + ":"
    print "\t n features: %s \n \t jump: %s \n \t receptive size: %s \n \t start: %s " % (
    layer[0], layer[1], layer[2], layer[3])


def residual_unit(stride, name, bottle_neck):
    if bottle_neck:
        return [[1,1,0],[3, stride, 1],[1,1,0]], [name + '_conv1', name + '_conv2', name + '_conv3']
    else:
        return [[3, stride, 1], [3, 1, 1]], [name + '_conv1', name + '_conv2']


def resnet(num_layer):
    assert num_layer in [18, 34, 50, 101, 152, 200, 269], '{} layers have not been implemented'.format(num_layer)
    if num_layer == 18:
        units = [2, 2, 2, 2]
    elif num_layer == 34:
        units = [3, 4, 6, 3]
    elif num_layer == 50:
        units = [3, 4, 6, 3]
    elif num_layer == 101:
        units = [3, 4, 23, 3]
    elif num_layer == 152:
        units = [3, 8, 36, 3]
    elif num_layer == 200:
        units = [3, 24, 36, 3]
    elif num_layer == 269:
        units = [3, 30, 48, 8]
    if num_layer >= 50:
        bottle_neck = True
    else:
        bottle_neck = False
    convnet = []
    layer_names = []
    num_stages = 4

    convnet.append([7,2,3])
    layer_names.append('conv1')
    convnet.append([3,2,1])
    layer_names.append('pool1')
    for i in range(num_stages):
        nets, names = residual_unit(stride=1 if i==0 else 2, name='stage{}_unit{}'.format(i+1, 1), bottle_neck=bottle_neck)
        convnet.extend(nets)
        layer_names.extend(names)
        for j in range(units[i] - 1):
            nets, names = residual_unit(stride=1, name='stage{}_unit{}'.format(i+1, j+2), bottle_neck=bottle_neck)
            convnet.extend(nets)
            layer_names.extend(names)
    return convnet, layer_names


def vgg(num_layer):
    assert num_layer in [16, 19], 'layer {} not implemented'.format(num_layer)
    if num_layer == 16:
        units = [2,5,9,13]
    elif num_layer == 19:
        units = [2,5,10,15]
    nets = [[3,1,1]] * (num_layer - 3)
    names = ['conv' + str(i+1) for i in range(num_layer - 3)]
    pools = ['pool' + str(i+1) for i in range(4)]
    for i in range(len(units)):
        nets.insert(units[i], [2,2,0])
        names.insert(units[i], pools[i])
    nets.append([3,1,1])
    names.append('rpn')
    return nets, names


if __name__ == '__main__':
    layerInfos = []
    # convnet, layer_names = resnet(50)
    convnet, layer_names = vgg(16)
    imsize = 224
    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5

    print "---------------Net summary-------------"

    currentLayer = [imsize, 1, 1, 0]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])

    print "---------------------------------------"

    layer_name = raw_input("Layer name where the feature in: ")
    layer_idx = layer_names.index(layer_name)
    idx_x = int(raw_input("index of the feature in x dimension (from 0):"))
    idx_y = int(raw_input("index of the feature in y dimension (from 0):"))

    n = layerInfos[layer_idx][0]
    j = layerInfos[layer_idx][1]
    r = layerInfos[layer_idx][2]
    start = layerInfos[layer_idx][3]
    assert idx_x < n
    assert idx_y < n

    print "receptive field: (%s, %s)" % (r, r)
    print "center: (%s, %s)" % (start + idx_x * j, start + idx_y * j)