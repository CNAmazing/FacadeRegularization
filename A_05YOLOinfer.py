import os
import cv2
import numpy as np
import onnxruntime as ort
from matplotlib import pyplot as plt
from A01_BIP import BIP
from A03_SLOD2_WIN import SLOD2_WIN
from A06 import ruleAlignment,ruleAlignment_Energy,ruleAlignment_EnergySingle
from Atools import YOLO11,draw_detections,plt_show_image,pltShow
import random


def cal_iou(bbox1, bbox2):
    inter_area = 0
    union_area=0
    for box1 in bbox1:
        for box2 in bbox2:
            # 计算交集区域
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])
            current_inter_area = (max(0, x2 - x1) * max(0, y2 - y1))
            inter_area +=current_inter_area
            
            # 计算并集区域
            if current_inter_area > 0:
                union_area +=( box1[2] * box1[3] + box2[2] * box2[3] - current_inter_area)
            
            # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def get_jpg_paths(folder_name):
    """
    获取指定文件夹下images子目录中的所有.jpg图片路径及不带后缀的文件名
    
    参数:
        folder_name (str): 目标文件夹名称（如"x"）
        
    返回:
        tuple: (完整路径列表, 不带后缀的文件名列表)，如(
                ["x/images/pic1.jpg", "x/images/pic2.jpg"], 
                ["pic1", "pic2"]
               )
    """
    full_paths = []
    basenames = []
    
    try:
        images_dir = folder_name
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
            
        for f in os.listdir(images_dir):
            if f.lower().endswith(".jpg"):
                file_path = os.path.join(images_dir, f)
                if os.path.isfile(file_path):
                    full_paths.append(file_path)
                    # 去掉.jpg后缀获取纯文件名
                    basenames.append(os.path.splitext(f)[0])
                    
        return full_paths, basenames
        
    except Exception as e:
        print(f"错误: {e}")
        return [], []
def  singleImageInference(image_path):
    # image_path = r"E:\WorkSpace\FacadeRegularization\dataset\all\00162.jpg"
    image=  cv2.imread(image_path)
    facade_detection = YOLO11(  onnx_model="checkpoint\YOLO_window.onnx",
                                input_image=image_path,
                                confidence_thres=0.5,
                                iou_thres=0.25,
                )
    """
    result={
            'boxes':result_boxes,
            'scores':result_scores,
            'class_ids':result_class_ids
            }
    """
    result,output_image = facade_detection.main()
    boxes = result['boxes']
    class_ids = result['class_ids']
    bbox=[ ]
    for i, box in enumerate(boxes):
        if class_ids[i] == 0:
            # 只处理类别为0的窗口
            bbox.append(box)
            
    # bbox = result['boxes']
    bbox_BIP= BIP(bbox)
    image_BIP=image.copy()
    draw_detections(image_BIP, bbox_BIP)
    
    bbox_SWIN,_= SLOD2_WIN(image,bbox)
    image_bbox_SWIN=image.copy()
    draw_detections(image_bbox_SWIN, bbox_SWIN)
    
    bbox_r= ruleAlignment(bbox)
    image_r=image.copy()
    draw_detections(image_r, bbox_r)

    bbox_rE= ruleAlignment_Energy(bbox)
    image_rE=image.copy()
    draw_detections(image_rE, bbox_rE)

    iou_BIP=cal_iou(bbox, bbox_BIP)
    iou_SWIN=cal_iou(bbox, bbox_SWIN)
    iou_r=cal_iou(bbox, bbox_r)
    iou_rE=cal_iou(bbox, bbox_rE)
    print(f"iou_BIP: {iou_BIP:.4f}")
    print(f"iou_SWIN: {iou_SWIN:.4f}")
    print(f"iou_r: {iou_r:.4f}")
    print(f"iou_rE: {iou_rE:.4f}")
    pltShow(output_image, image_BIP, image_bbox_SWIN,image_r,image_rE)
    
    return iou_BIP,iou_SWIN,iou_r,iou_rE
def main():
    # folder= r"E:\WorkSpace\FacadeRegularization\dataset\all"
    # jpg_paths, basenames = get_jpg_paths(folder)
    # # random.shuffle(jpg_paths)
    # BIP_iouScore=[]
    # SWIN_iouScore=[]
    # r_iouScore=[]
    # rE_iouScore=[]
    # for jpg_path in jpg_paths:
    #     print(f"Processing {jpg_path}...")
    #     iou_BIP,iou_SWIN,iou_r,iou_rE=singleImageInference(jpg_path)
     
    #     BIP_iouScore.append(iou_BIP)
    #     SWIN_iouScore.append(iou_SWIN)
    #     r_iouScore.append(iou_r)
    #     rE_iouScore.append(iou_rE)
    # BIP_iouScore=np.array(BIP_iouScore)
    # SWIN_iouScore=np.array(SWIN_iouScore)
    # r_iouScore=np.array(r_iouScore)
    # rE_iouScore=np.array(rE_iouScore)
    # BIP_iouScore_mean = np.mean(BIP_iouScore)
    # SWIN_iouScore_mean = np.mean(SWIN_iouScore)
    # r_iouScore_mean = np.mean(r_iouScore)
    # rE_iouScore_mean = np.mean(rE_iouScore)
    # print(f"BIP_iouScore_mean: {BIP_iouScore_mean:.4f}")
    # print(f"SWIN_iouScore_mean: {SWIN_iouScore_mean:.4f}")
    # print(f"r_iouScore_mean: {r_iouScore_mean:.4f}")
    # print(f"rE_iouScore_mean: {rE_iouScore_mean:.4f}")
    '''
    单图像推理
    '''
    image_path = r"E:\WorkSpace\FacadeRegularization\dataset\poly3.jpg"
    singleImageInference(image_path)
if __name__ == "__main__":
   main()