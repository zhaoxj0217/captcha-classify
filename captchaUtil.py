#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
from PIL import Image
import captchaUtil as capt

def getCaptcha(image,model):
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
        text_x  = np.tile(charvec, (1, 1))
        char =  model.predict(text_x)#SVM分类算法识别该图片是什么字符
        chars.append(str(char[0]))
    return "".join(chars)

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    model = SVC(kernel='rbf', probability=True,decision_function_shape='ovo')
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params() #获取最佳参数
    for para, val in best_parameters.items():
        print
        para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True,decision_function_shape='ovo')
    model.fit(train_x, train_y)
    return model



if __name__ == '__main__':
    train_x, train_y  = capt.createDataSet()
    num_train, num_feat = train_x.shape
    start_time = time.time()
    model = svm_cross_validation(train_x, train_y)
    print ('training took %fs!' % (time.time() - start_time))
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
        svmtext = getCaptcha(out,model)
        if text == svmtext:
            right += 1
            print("svm right")
        else:
            print(text, "svm error:", svmtext)
    print("right:total is:", right, total)



