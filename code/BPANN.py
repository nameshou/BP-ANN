#_*_coding:utf-8_*_
"""
    net3.py,BP神经网络的主文件，主要向外暴露的函数如下：
        Network类下的: 
                SGD(····)
                predict(···)
                save(···)
        
        非Network类下的函数有:
                load(···)
        具体参数在下面的每一个函数之前都有介绍。
"""


import json
import random
import sys

import numpy 

class BPANN(object):

    def __init__(self, sizes ):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [numpy.random.randn(y, x) / numpy.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.accuracy = []


    
    def predict(self, a):
        """
            预测某一个数据的值，假如该网络的输入层神经元的个数是100个的话，则a为100行，1列的二维数组。
                inumpyuts :
                    a : 用来测试的数据
                returns:
                    预测的结果
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            evaluation_data=None ):
        """
            SGD : （ stochastic gradient descent ）随机梯度下降的缩写，其实这个个函数就是训练数据的函数。
            inumpyuts: 
                training_data ，是一个包含多个输入与输出元组的列表，形如[(x1,y1), (x2,y2)···]
                epochs: 迭代次数
                mini_batch_size : 随机抽取样本的个数，对于一个有超大规模的输入集，我们将这些数据集分成多份来进行迭代更新权值以及阈值，该参数表示将当前训练集分割成多少份。
                eta : 学习速率
                lmbda : 正则化参数
                evaluation_data : 用来验证当前网络的识别效率的数据集，如果有的话，在每一次迭代完成的时候都会显示当前的权值以及阈值下该神经网络的识别效率。
                
        """
                
        if evaluation_data is not None: 
            n_data = len(evaluation_data)
        n = len(training_data)
        
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch( mini_batch, eta )
                
            print "Epoch %s training complete" % j
            # 打印一下各个数据的预测百分比
            if evaluation_data is not None:
                accuracy = self.getAccuracy( evaluation_data )
                print "The accuracy of test data is %s / %s " % ( accuracy, n_data )
                self.accuracy.append( accuracy )

    def update_mini_batch(self, mini_batch, eta ):
        """
            更新权值与阈值的函数
            inumpyuts : 
                mini_batch : 在SGD函数中将训练集分割成若干份之后的list集合
                eta : 学习速率
                lmbda : 正则化参数
                n : 整个训练集的个数

        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            # 下面这一行代码比较关键，它是实现反向传播求出各个神经层的权值与阈值的偏差的关键函数
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        #更新权值与阈值
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
            通过反向传播算法来求解在某哥输入数据下的各个神经层的偏差，进而求出它们各自的权值与阈值的偏差，并返回。
            inumpyuts :
                x : 某一个输入的数据
                y ： 期望输出的结果
            
            returns:
                返回求解得到的各个神经层的权值与阈值的偏差的列表
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        
        activation = x
        
        # 保存每一个神经层的激活值
        activations = [x] 
        
        # 保存每一个神经层的z值，z = wx + b
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # 计算最后一层的偏差，进而才能计算前两层的偏差
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        
        # 从倒数第二层开始计算各个神经层的偏差
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)


    def getAccuracy(self, data ):
        """
            获取某一个测试数据集的测试精度
            inumpyuts：
                data : 测试的数据集，与training_data是相类似的
            
            returns:
                测试集的正确率
        """
        results = [( numpy.argmax( self.predict(x) ), \
                    numpy.argmax( y ) )
                    for (x, y) in data]
                    
        return sum(int(x == y) for (x, y) in results)


    def save(self, filename):
        """
            将神经网络的神经层的信息 、 权值、阈值三个参数保存到文件中。
            inumpyut :
                filename : 存储数据的文件
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        


def load(filename):
    """
        读取由save存储的数据，并初始化神经网络
        inumpyut :
            filename : 存储数据的文件名
        returns :
            net : 初始化之后的神经网络
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = BPANN( data["sizes"] )
    net.weights = [numpy.array(w) for w in data["weights"]]
    net.biases = [numpy.array(b) for b in data["biases"]]
    
    return net



###  以下的函数为本文件的辅助函数
def vectorized_result(j):
    """
        生成10行1列的向量，并将下标为j的元素的值设置为1
        inumpyut : 
            j : 将哪一个元素设置为1的下标
    """
    array = numpy.zeros((10, 1))
    array[j] = 1.0
    return numpy.array( array ,dtype=numpy.float32 )

def sigmoid(z):
    """
        激活函数
    """
    return 1.0 / ( 1.0 + numpy.exp(-z) )


def sigmoid_prime( z ):
    """
        sigmoid函数的求导形式
    """
    return sigmoid( z )*( 1 - sigmoid(z) )
    