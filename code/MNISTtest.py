#_*_coding:utf-8_*_
"""
    提取神经网络保存的参数，并测试数据
"""
import BPANN
import MNISTtrain

net = BPANN.load( 'netParam.txt' )

test = MNISTtrain.getDataFromFile( './test/t10k-images.idx3-ubyte', './test/t10k-labels.idx1-ubyte', 10000 )

# 测试结果
result = net.getAccuracy( test )
print result
    