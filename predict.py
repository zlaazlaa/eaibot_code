#!/usr/bin/env python3
# coding=utf-8
# @Time    : 2022/5/1 9:35
# @Author  : 武汉理工大学 - 智慧物流挑战赛小车
# @File    : predict.py
# @Description : 总测试的推理程序

import os
import sys
import io
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import time
import numpy as np

from PIL import Image, ImageDraw, ImageFont

import ocr_rec_predict as predict_rec
import ocr_det_predict as predict_det
from object_predict import Detector

# 所有省份的标签值
LABELS = ['北京', '天津', '辽宁', '甘肃', '福建', '山东', '湖北', '四川', '江苏', '河北', '云南',
          '山西', '广东', '台湾', '广西', '宁夏', '香港', '上海', '重庆', '吉林', '青海', '浙江',
          '安徽', '湖南', '陕西', '江西', '河南', '贵州', '海南', '西藏', '新疆', '澳门', '内蒙古', '黑龙江']

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def doMsg(msg):
    rospy.loginfo("I heard:%s", msg.data)

def callback(data):
    imgdata = CvBridge().imgmsg_to_cv2(data, "rgb8")
    cv2.imwrite("/home/eaibot/cam_test.jpg", imgdata)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/usb_cam/image_correct", Image, callback)


class TextSystem(object):
    """总的检测类, 完成目标检测, 文本检测, 文本识别

    Args:
        args: 参数配置
        det_model_dir: 文本检测模型的路径
        rec_model_dir: 文本识别模型的路径
        picodet_model_dir: 目标检测模型的路径
        turn_mode: 旋转模式 0:不旋转  1:逆时针旋转  2:顺时针旋转
        threshold: 目标检测结果的置信度阈值
    """

    def __init__(self, det_model_dir, rec_model_dir, picodet_model_dir, threshold=0.8):
        self.text_detector = predict_det.TextDetector(det_model_dir)  # 创建文本模型检测器
        self.text_recognizer = predict_rec.TextRecognizer(rec_model_dir)  # 创建文本识别检测器
        self.detector = Detector(picodet_model_dir, threshold)  # 创建目标检测模型检测器
        self.drop_score = 0.5  # 识别置信度阈值

    def __call__(self, img, original_image, ratio):
        """
        对输入图片进行预测推理

        :param img: 输入的缩小图片
        :param original_image: 原始图像
        :param ratio: 原始图像和缩小的图片的比率
        :return: filter_rec_res: 过滤后的识别结果
                 bbox: 目标检测的bbox[x_min, y_min, x_max, y_max]
                 dt_boxes: 文字识别的预测框
        """
        result = self.detector.predict_image([img])  # 对图片进行目标检测, 返回N个box  box:[cls, [x_min, y_min, x_max, y_max], conf]
        bbox = result[0][1]  # 得到bbox
        original_bbox = [int(ratio * i) for i in bbox]  # 对bbox的每个值乘以比率得到原始图像的bbox值

        # 裁剪出邮件图片
        crop_img = original_image[original_bbox[1]:original_bbox[3], original_bbox[0]:original_bbox[2], :]
        # 将裁剪的图片送入文本识别网络
        dt_boxes = self.text_detector(crop_img)
        # 对dt_boxes从上到下, 从左到右排序
        dt_boxes = sorted_boxes(dt_boxes)
        ori_im = crop_img

        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])  # 复制dt_box
            img_crop = get_rotate_crop_image(ori_im, tmp_box)  # 根据dt_box裁剪出对应的文本识别框
            img_crop_list.append(img_crop)  # 在裁剪图片列表中添加裁剪的图片

        rec_res = self.text_recognizer(img_crop_list)  # 对裁剪的图片列表进行文本识别推理

        filter_boxes, filter_rec_res = [], []
        # 遍历检测框dt_boxes和对应的识别结果, 通过阈值drop_score过滤掉置信度低的识别结果
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:  # 根据阈值进行过滤
                filter_boxes.append(box)  # 添加符合要求的dt_box
                filter_rec_res.append(rec_result)  # 添加符合要求的识别结果

        return filter_rec_res, bbox, dt_boxes


def sorted_boxes(dt_boxes):
    """
    将dt_box从上到下从左到右排序

    :param dt_boxes: dt_boxes文本检测框的四个角点坐标
    :return: 排序后的dt_boxes
    """
    num_boxes = dt_boxes.shape[0]  # [N 4 2]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[3][1]))

    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    """
    根据dt_boxes裁剪出对应的文本识别框

    :param img: 输入图片
    :param points: dt_box四个角点坐标
    :return: 裁剪得得图片
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                  borderMode=cv2.BORDER_REPLICATE,
                                  flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_text(rec_res):
    """
    文本识别筛选函数, 遍历模型推理的文本识别结果,
    遇到符合要求的直接返回

    :param rec_res: 模型推理的文本识别结果
    :return: 最终的识别筛选结果, 若没有则返回None
    """
    for res in rec_res:
        name = res[0]
        if len(name) >= 2:
            if name.encode("utf-8") in LABELS:
                return name
            name = name[:2]
            if name in LABELS:
                return name
            elif name == '贵阳':
                return '贵州'
            elif name == '黑龙':
                return '黑龙江'
            elif name == '内蒙':
                return '内蒙古'
            elif name == '订宁' or name == '丁宁':
                return '辽宁'

    return None


def draw_show(image, txt, bbox, font_path):
    """ 可视化结果, 按照要求显示原图片和处理后的图片

    :param image: 输入原图片
    :param txt: 识别的省份名称
    :param bbox: 目标检测出的检测框
    :param font_path: 使用字体的路径
    :return: None
    """
    height, width = image.shape[:2]
    if txt is None:
        print("---------------")
        txt = "None"
    if len(bbox) == 0:
        bbox = [0, 0, 0, 0]

    font_size = 50
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")  # 创建字体

    pil_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将OpenCV的BGR格式转换为PIL的RGB格式
    pil_img = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pil_img)
    draw.text((0, 0), txt, (255, 0, 0), font=font)  # 在图片上标注省份
    cv2img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式

    cv2.rectangle(cv2img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 1)  # 标注邮件目标位置
    center_xy = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))  # 计算邮件中心坐标
    cv2.putText(cv2img, "[%d,%d]" % (center_xy[0], center_xy[1]), (center_xy[0] - 40, center_xy[1] + 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, lineType=cv2.LINE_AA)  # 标注邮件中心位置坐标

    draw_widow = 'draw%dx%d' % (width, height)  # 以图片的高和宽命名处理后的图像窗口
    cv2.namedWindow('original-image')  # 命名原始图像显示窗口
    cv2.namedWindow(draw_widow)  # 命名处理后的图像显示窗口
    cv2.imshow("original-image", image)  # 显示原始图像
    cv2.imshow(draw_widow, cv2img)  # 显示处理后的图像
    cv2.waitKey(1)  # 等待1ms


def main(image_dir, det_model_dir, rec_model_dir, picodet_model_dir):
    """
    主函数, 完成图片的目标检测, 文字检测, 文字识别, 图片展示, 结果保存

    :param image_dir: 需要检测的图片的路径
    :param det_model_dir: 文字检测模型的路径
    :param rec_model_dir: 文字识别模型的路径
    :param picodet_model_dir: 目标检测模型的路径
    :return: None
    """

    # 获得需要检测图片名称列表
    image_file_list = os.listdir(image_dir)
    # 从图片文件夹中选取1张
    image_choice_list = random.sample(image_file_list, 1)
    print("predict images (%d) : " % len(image_choice_list))
    print(image_choice_list)
    # 创建推理类
    text_sys = TextSystem(det_model_dir, rec_model_dir, picodet_model_dir)
    # 字体路径
    font_path = 'font/simsun.ttc'
    # 在当前路径下创建result.txt结果文件并打开以后续写入
    f = io.open('result.txt', 'w', encoding='utf-8')
    # 开始预测
    # 记录开始时间
    total_time = 0.
    # 遍历图片列表开始推理识别
    #    for image_file in image_choice_list:
    # 记录每张图片推测的开始时间
    img_start_time = time.time()
    # 图片的相对路径
    img_path = './images/1.jpg'
    # 读取图片
    original_image = cv2.imread(img_path)
    # 记录原图像的高和宽
    original_height, original_width = original_image.shape[:2]
    # 计算图像的高宽之比
    ratio = original_height * 1. / original_width
    # 缩小图片, 利于目标检测
    img = cv2.resize(original_image, (320, int(320 * ratio)), interpolation=cv2.INTER_LINEAR)
    # 预测推理, 进行以此目标检测, 文本检测, 文本识别, 返回文字识别结果(未筛选)、邮件检测框、文字检测框
    rec_res, bbox, dt_boxes = text_sys(img, original_image, original_width / 320)
    # 对文本识别得到的结果进行筛选, 获得最终的文字识别结果
    rec = get_text(rec_res)
    print("rec: ", rec)
    # 计算目标中心坐标(x, y)
    center_xy = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
    # 每张图片的识别结果
    text = u"%s,(%d,%d);" % (rec, center_xy[0], center_xy[1])
    # 写入result.txt
    f.write(text)
    # 显示结果函数, 显示原图和处理后的图片
    draw_show(img, rec, bbox, font_path)
    # 输出一张图片的推测时间
    end_time = time.time() - img_start_time
    total_time += end_time
    print("time : %.3f s" % end_time)
    print("--------total_time : %.3f s" % total_time)

if __name__ == "__main__":
    # 预测图片的路径
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = 'images'

    # 文字检测模型的路径
    ocr_det_model_dir = 'weights/ocr_det_model'

    # 文字识别模型的路径
    ocr_rec_model_dir = 'weights/ocr_rec_model'

    # 目标检测模型的路径
    obj_model_dir = 'weights/obj_det_model'

    # 主函数入口
    main(image_dir, ocr_det_model_dir, ocr_rec_model_dir, obj_model_dir)
