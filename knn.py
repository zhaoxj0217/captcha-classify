from PIL import Image
import numpy as np
import os
import captchaUtil as capt

#knn分类算法 此处使用欧式距离公式
#inX :待分类向量
#dataSet：训练集合
#labels：训练集合对应的分类结果
#k值
def classify(inX, dataSet, labels, k):
    # 距离计算 start,
    dataSetSize = dataSet.shape[0]# 计算有多少个训练样本
    tmp = np.tile(inX, (dataSetSize, 1))# 将待分类的输入向量进行 行反向复制dataSetSize次，列方向复制1一次,即与训练样本大小一致
    diffMat = tmp - dataSet # 数组相减
    sqDiffMat = diffMat ** 2 #2次方
    sqDistances = sqDiffMat.sum(axis=1)# 对于二维数组axis=1表示按行相加 , axis=0表示按列相加
    distances = sqDistances ** 0.5 # 平方根
    # 距离计算 end
    sortedDistIndicies = distances.argsort() # 将输入与训练集的距离排序 argsort函数返回的是数组值从小到大的索引值
    classCount = {}
    for i in range(k): # 统计距离最近的K个lable值
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]#返回k个中数量最多的label

# 识别验证码
def getCaptcha(dataSet,lables,image):
    w, h = image.size
    splitCol = capt.getSplit(image)#切割验证码一个个字符识别
    if len(splitCol) != 4:
        print("切割错误！", text)
        return ""
    chars = []
    for i in range(4):
        cropImg = image.crop((splitCol[i][0], 0, splitCol[i][1], h))
        cropImg = capt.resizeImge(cropImg, int(w /4), h)  # 切割的图片尺寸不一，统一尺寸
        charvec = capt.img2vector(cropImg)
        char =  classify(charvec, dataSet, lables, 3)#KNN分类算法识别该图片是什么字符
        chars.append(str(char))
    return "".join(chars)


if __name__ == "__main__":

    # 训练图片的预处理 --分割图片 --start
    # im_paths = filter(lambda fn: os.path.splitext(fn)[1].lower() == '.jpg',
    #                   os.listdir("./image/train"))
    #
    # for im_path in im_paths:
    #     try:
    #         imagepath = "./image/train/" + im_path
    #         image = Image.open(imagepath)
    #         imgry = image.convert('L')  # 转化为灰度图
    #         # threshold =  adaptiveThreshold(imgry)
    #         table = capt.get_bin_table(230)
    #         out = imgry.point(table, '1')
    #         text = os.path.basename(im_path).split('_')[0]
    #         capt.splitByPixel(out, text)
    #     except:
    #         print(im_path, "error")
    #         pass
    # 训练图片的预处理 --分割图片 --end

    # 图片识别  --start
    #创建训练集合
    dataSet,lables = capt.createDataSet()#训练数据加载比较慢

    #测试knn分类结果
    im_paths = filter(lambda fn: os.path.splitext(fn)[1].lower() == '.jpg',
                       os.listdir("./image/test"))
    total = 0
    right = 0
    for im_path in im_paths:
        total+=1
        imagepath = "./image/test/" + im_path
        #测试图片预处理 start
        image = Image.open(imagepath)
        imgry = image.convert('L')  # 转化为灰度图
        table = capt.get_bin_table(230)
        out = imgry.point(table, '1')
        # 测试图片预处理 end
        text = os.path.basename(im_path).split('_')[0]
        # 识别验证码
        knntext = getCaptcha(dataSet,lables,out)
        if text == knntext:
            right+=1
            print("knn right")
        else:
            print(text,"knn error:",knntext)
    print("right:total is:",right,total)
    # 图片识别  --end

