#_*_coding:utf-8_*_
"""
    用net3.py的BP神经网络来测试MNIST的数据集
""" 

import struct
import numpy
import cv2

import BPANN

def getDataFromFile(  imagePath, labelPath, samples = 1000, func=None ):
    """
        从MNIST数据集中获取数据，因为在新的MNIST数据集中，数据存放的方式是将每一个图片铺平之后序列存储到文件中的，所以我们可以反过来将其读取出来。
        inputs:
            imagePath : 保存图片的文件
            labelPath : 保存图像标签信息的文件
            samples : 需要取得多少的图片，默认为1000张
            func : 在获取每一张图片之前是否需要处理一下，比如细化，或者我们想获取的是图像的其他信息，可以由这个函数来处理
    """
    binaryImageData = open( imagePath, 'rb' )
    imageBuffer = binaryImageData.read()
    binaryImageData.close()
    
    binaryLabel = open( labelPath , 'rb' )
    labelBuffer = binaryLabel.read()
    binaryLabel.close()
    
    trainData = []
    resultData = []
    #使用大端法读取四个整型数---每个整型数4个字节，分别为魔数、图片数量、图片像素行列数，
    #index为位置指针。(魔数，图片类型的数)
    #不过我们不需要这些东西
    index = 0
    #magic, numImages, numRows, numColums=struct.unpack_from('>iiii', imageBuffer, index )
    
    index += struct.calcsize('>iiii' )
    
    for x in range(samples):
        img = numpy.array( struct.unpack_from( '>784B', imageBuffer, index ) )\
                    .reshape( 28, 28).astype( numpy.uint8 )
        #二值化图像
        img[numpy.where( img[:, :] > 0)] = 1
        
        if func is None:
            img = img.reshape( 784, -1 )
        else:
            img = func( img )

        trainData.append( img )
        index += struct.calcsize( '>784B' )
    
    #获得对应的标签
    index = struct.calcsize('>ii' )
    for x in range(samples):
        r = numpy.zeros( 10 ).reshape( 10, -1 )
        result = numpy.array( struct.unpack_from( '>B', labelBuffer, index ) )[0]
        r[result] = 1;
        resultData.append( r )
        index += struct.calcsize( '>B' )
        
    return zip( trainData, resultData )

def imshow( img, nameOfWindow='nameOfWindow' ):
    cv2.imshow( nameOfWindow, img )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
将灰度图像转换成彩色图像，方便查看效果
"""
def cvt2BGR( img ):
    edges = cv2.cvtColor( img.copy(), cv2.COLOR_GRAY2BGR )
    return edges
    
if __name__ == '__main__':
    train = getDataFromFile( './samples/train-images.idx3-ubyte', './samples/train-labels.idx1-ubyte', 60000 )

    """
    for i in range( 10 ):
        img = train[i][0].reshape( 28, 28 ).astype( numpy.uint8 ) * 255 
        number = numpy.argmax( train[i][1] )
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cvt2BGR( img )
        cv2.putText( img, str( number ), ( 0, 20 ), font, 1, ( 255, 0, 0 ), 1 )
        imshow( img )
    """
    # 初始化神经网络
    net = BPANN.BPANN( [784, 30, 10 ] )
    
    # 训练，参数比较随意，所以结果不是最佳的
    #30表示迭代次数
    #6表示我们将训练的数据分成若干份，每一份包含6个数据
    #0.2表示学习速率
    net.SGD( train[:50000], 10, 6, 0.2, train[50000:] )
    
    # 保存网络的参数
    net.save( 'netParam.txt' )