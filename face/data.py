#使用workbook方法，创建一个新的工作簿
import cv2
import xlwt
import numpy as np
import joblib
import cv2  # 图像处理的库OpenCv
def data(self):
    line_brow_x = []
    line_brow_y = []
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    #添加一个sheet，名字为mysheet
    sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
    #遍历每张图片提取图片中人脸的特征值
    path="D:\desktop\-Dlib--2.0\数据集\S005_001_00000009.png"
    print(path)
    im_rd = cv2.imread(path)
    im_rd = cv2.resize(im_rd ,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    n = 0
    k = cv2.waitKey(1)
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
    # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
    faces = self.detector(img_gray, 0)
    # 待会要显示在屏幕上的字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 如果检测到人脸
    if len(faces) != 0:
        # 对每个人脸都标出68个特征点
        # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
        for k, d in enumerate(faces):
            # 用红色矩形框出人脸
            cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
            # print(d.top())
            # 计算人脸热别框边长
            self.face_width = d.right() - d.left()
            # 使用预测器得到68点数据的坐标
            shape = self.predictor(im_rd, d)
            # 圆圈显示每个特征点
            for i in range(17, 27):
                cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (214, 238, 247), -1, 8)
            for i in range(36, 68):
                cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (214, 238, 247), -1, 8)
                # cv2.putText(im_rd, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            for i in range(18, 22):  # 眉毛连线
                cv2.line(im_rd, (shape.part(i).x, shape.part(i).y), (shape.part(i - 1).x, shape.part(i - 1).y),
                         (214, 238, 247))
                cv2.line(im_rd, (shape.part(i + 5).x, shape.part(i + 5).y),
                         (shape.part(i + 4).x, shape.part(i + 4).y), (214, 238, 247))
            for i in range(36, 42):  # 眼睛连线
                if i == 36:
                    cv2.line(im_rd, (shape.part(36).x, shape.part(36).y), (shape.part(41).x, shape.part(41).y),
                             (214, 238, 247))
                    cv2.line(im_rd, (shape.part(42).x, shape.part(42).y), (shape.part(47).x, shape.part(47).y),
                             (214, 238, 247))
                    continue
                cv2.line(im_rd, (shape.part(i).x, shape.part(i).y), (shape.part(i - 1).x, shape.part(i - 1).y),
                         (214, 238, 247))
                cv2.line(im_rd, (shape.part(i + 6).x, shape.part(i + 6).y),
                         (shape.part(i + 5).x, shape.part(i + 5).y), (214, 238, 247))
            for i in range(49, 60):  # 嘴巴外圈连线
                if i == 59:
                    cv2.line(im_rd, (shape.part(59).x, shape.part(59).y), (shape.part(48).x, shape.part(48).y),
                             (214, 238, 247))
                cv2.line(im_rd, (shape.part(i).x, shape.part(i).y), (shape.part(i - 1).x, shape.part(i - 1).y),
                         (214, 238, 247))
            # 分析任意n点的位置关系来作为表情识别的依据
            for i in range(61, 68):  # 嘴巴内圈连线
                if i == 67:
                    cv2.line(im_rd, (shape.part(67).x, shape.part(67).y), (shape.part(60).x, shape.part(60).y),
                             (214, 238, 247))
                cv2.line(im_rd, (shape.part(i).x, shape.part(i).y), (shape.part(i - 1).x, shape.part(i - 1).y),
                         (214, 238, 247))
            mouth_width = (shape.part(54).x - shape.part(48).x) / self.face_width  # 嘴巴咧开程度
            mouth_higth = (shape.part(66).y - shape.part(62).y) / self.face_width  # 嘴巴张开程度
            a = round(mouth_width, 10)
            b = round(mouth_higth, 10)
            # print("嘴巴宽度与识别框宽度之比：", mouth_width)
            # print("嘴巴高度与识别框高度之比：", mouth_higth)
            # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
            brow_sum = 0  # 高度之和
            frown_sum = 0  # 两边眉毛距离之和
            for j in range(17, 21):
                brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                frown_sum += shape.part(j + 5).x - shape.part(j).x
                line_brow_x.append(shape.part(j).x)
                line_brow_y.append(shape.part(j).y)
            # self.brow_k, self.brow_d = self.fit_slr(line_brow_x, line_brow_y)  # 计算眉毛的倾斜程度
            tempx = np.array(line_brow_x)
            # print(tempx)
            tempy = np.array(line_brow_y)
            # print(tempy)
            #    if (len(line_brow_y) >= 20 or len(line_brow_x) >= 20):
            #        line_brow_x = []
            #        line_brow_y = []
            z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
            # print(z1)
            self.brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的
            brow_hight = (brow_sum / 10) / self.face_width  # 眉毛高度占比
            brow_width = (frown_sum / 5) / self.face_width  # 眉毛距离占比
            f = self.brow_k
            c = round(brow_hight, 10)
            dd = round(brow_width, 310)
            # 眼睛睁开程度
            eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                       shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
            eye_hight = (eye_sum / 4) / self.face_width
            e = round(eye_hight, 10)
            # print("眼睛睁开距离与识别框高度之比：",round(eye_open/self.face_width,3))
            # 分情况讨论
            # 张嘴，可能是开心或者惊讶

            #人脸特征之眼高值的读入
            print(a,b,c,d,dd,e,f)
            n = n + 1
            #人脸特征值之眉毛倾斜程度的读入
            #sheet.write(m,n,self.brow_k)
            #保存xls工作簿
            book.save()
