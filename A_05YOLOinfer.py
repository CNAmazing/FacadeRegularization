import os
import cv2
import numpy as np
import onnxruntime as ort
from matplotlib import pyplot as plt
from A01_BIP import BIP
from A03_SLOD2_WIN import SLOD2_WIN
from A06 import ruleAlignment,ruleAlignment_Energy,ruleAlignment_EnergySingle
from Atools import YOLO11,draw_detections,plt_show_image,pltShow,draw_detections_by_wireframe,cvSave,YOLO11_glass,pre_cluster,boxes_classification_HW_Get_idx
from A08_grow import cluster_completion,cluster_completion_HW
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
    door=[]
    for i, box in enumerate(boxes):
        if class_ids[i] == 0:
            # 只处理类别为0的窗口
            bbox.append(box)
        if class_ids[i] == 2:
            # 只处理类别为1的门
            door.append(box)    
    ori_image=image.copy()
    draw_detections(image, door)
    draw_detections_by_wireframe(ori_image, door)
    draw_detections_by_wireframe(ori_image, bbox)
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
    pltShow(ori_image, image_BIP, image_bbox_SWIN,image_r,image_rE)
    # cvSave(ori_image, image_BIP, image_bbox_SWIN,image_r,image_rE)
    
    return iou_BIP,iou_SWIN,iou_r,iou_rE
def cal_BoxesBounding(boxes):
    
        # 转换为 (x1, y1, x2, y2) 格式
    x1_values = [rect[0] for rect in boxes]
    y1_values = [rect[1] for rect in boxes]
    x2_values = [rect[0] + rect[2] for rect in boxes]  # x + w
    y2_values = [rect[1] + rect[3] for rect in boxes]  # y + h

    # 计算最小外接矩形
    min_x1 = min(x1_values)
    min_y1 = min(y1_values)
    max_x2 = max(x2_values)
    max_y2 = max(y2_values)

    bounding_box = (min_x1, min_y1, max_x2, max_y2)
    return bounding_box
def symmetrScore(window_W,window_H,boxes,A=0.5):
    mask=np.zeros((window_H,window_W),dtype=bool)
    boxes = np.array(boxes)
    area=boxes[:, 2] * boxes[:, 3]  
    areaScore= np.sum(area) / (window_W * window_H) if (window_W * window_H) > 0 else 0
    for box in boxes:
        mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = True
    mask_left = mask[:, :window_W // 2]
    mask_right = mask[:, window_W // 2:]
    if  mask_right.shape[1] > mask_left.shape[1]:
        # 如果右半部分比左半部分宽，进行填充
        mask_right=mask_right[:, 1:]
    mask_right_flipped = mask_right[:, ::-1]
    intersection = np.logical_and(mask_left, mask_right_flipped)
    union = np.logical_or(mask_left, mask_right_flipped)
    iouScore = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

    return A*areaScore + (1-A)*iouScore

def centerAlign(window_W,window_H,glass_boxes):
    glass_boxes= np.array(glass_boxes)
    boundingBox= cal_BoxesBounding(glass_boxes)
    min_x1, min_y1, max_x2, max_y2 = boundingBox
    bounding_W = min_x1+(max_x2 - min_x1)/2
    bounding_H = min_y1+(max_y2 -  min_y1)/2
    T_x= window_W/2 - bounding_W
    T_y= window_H /2- bounding_H
    return (T_x, T_y)

def glassRegularizationReplace(boxes,image,delta=5):
    """
    识别窗户的玻璃，除了进行规则化处理，还进行替换处理
    """
    w=[]
    h=[]
    result=[]
    for box in boxes:
        w.append(box[2])
        h.append(box[3])
    W,W_label= pre_cluster(w, delta=delta)
    H,H_label= pre_cluster(h, delta=delta)
    HW_label = boxes_classification_HW_Get_idx(boxes,W_label, H_label)
    for key, value in HW_label.items():
        idxs= value['id']
        for idx in idxs:
            box = boxes[idx]
            window_x1, window_y1, window_w, window_h = box
            glass_detection = YOLO11_glass( onnx_model="checkpoint\YOLO_glass.onnx",
                        input_image=image[window_y1:window_y1+window_h,window_x1:window_x1+window_w,: ],
                        confidence_thres=0.5,
                        iou_thres=0.25,
                        )
            glass_result,glass_output_image = glass_detection.main()
            glass_result = glass_result['boxes']
            if len(glass_result) > 0:
                glassRe=ruleAlignment(glass_result)
                T= centerAlign(window_w, window_h, glassRe)
                for g in glassRe:
                    g[0] +=  T[0]
                    g[1] +=  T[1]
            if 'glass' not in value:
                value['glass'] = []  
            if 'glassScore' not in value:
                value['glassScore'] = []
            value['glass'].append(glassRe)
            value['glassScore'].append(symmetrScore(window_w, window_h, glassRe))
    for key, value in HW_label.items():
        idxs= value['id']
        glass= value['glass']
        if len(idxs) <=1:
            curGlass = value['glass'][0]
            for g in curGlass:
                id= idxs[0]
                box = boxes[id]
                g[0] += box[0]
                g[1] += box[1]
                result.append(g)
        else:
            score= value['glassScore']      
            max_score_idx = np.argmax(score)     
            max_score_glass = glass[max_score_idx]
            max_score_window = boxes[idxs[max_score_idx]]     
            window_W, window_H = max_score_window[2], max_score_window[3]

            boxes=np.array(boxes)

            window=boxes[idxs]  
            scale_M=window[:, 2:]/(window_W, window_H)
            scale_M=np.hstack((scale_M, scale_M))
            glassGroup=[]
            for g in max_score_glass:
                glassNew=scale_M * np.array(g)
                glassGroup.append(glassNew)
            glassGroup=np.array(glassGroup)
            for i in range(len(glassGroup)):
                for j in range(len(idxs)):
                    g_box = glassGroup[i][j]
                    g_box[0] += window[j][0]
                    g_box[1] += window[j][1]
                    result.append(g_box)
            # print(f"HW_label: {HW_label}")  
    return result

def glassRegularization(boxes,image):
    """
    识别窗户中的玻璃，并进行规则化处理 包括对齐和居中
    """
    glass_boxes = []
    for box in boxes:
        window_x1, window_y1, window_w, window_h = box
        glass_detection = YOLO11_glass( onnx_model="checkpoint\YOLO_glass.onnx",
                    input_image=image[window_y1:window_y1+window_h,window_x1:window_x1+window_w,: ],
                    confidence_thres=0.5,
                    iou_thres=0.25,
                    )
        glass_result,glass_output_image = glass_detection.main()
        if len(glass_result['boxes']) == 0:
            continue
        glass_result = glass_result['boxes']
        glassRe=ruleAlignment(glass_result)
        T= centerAlign(window_w, window_h, glassRe)
        for g_box in glassRe:
            g_box[0] += T[0]
            g_box[1] += T[1]
          

            g_box[0] += window_x1
            g_box[1] += window_y1
            glass_boxes.append(g_box)
    return glass_boxes
def glassInfer(boxes,image):
    glass_boxes = []
    for box in boxes:
        window_x1, window_y1, window_w, window_h = box
        glass_detection = YOLO11_glass( onnx_model="checkpoint\YOLO_glass.onnx",
                    input_image=image[window_y1:window_y1+window_h,window_x1:window_x1+window_w,: ],
                    confidence_thres=0.5,
                    iou_thres=0.25,
                    )
        glass_result,glass_output_image = glass_detection.main()
       
        for g_box in glass_result['boxes']:
            g_box[0] += window_x1
            g_box[1] += window_y1
            glass_boxes.append(g_box)
    return glass_boxes
def glassMain(image_path):
    image= cv2.imread(image_path)
    facade_detection = YOLO11( onnx_model="checkpoint\YOLO_window.onnx",
                                input_image=image_path,
                                confidence_thres=0.5,
                                iou_thres=0.25,
    )
    result,output_image = facade_detection.main()
    boxes = result['boxes']
    class_ids = result['class_ids']
    bbox=[ ]
    for i, box in enumerate(boxes):
        if class_ids[i] == 0:
            # 只处理类别为0的窗口
            bbox.append(box)
        # if class_ids[i] == 2:
        #     bbox.append(box)

    
    image_ori=image.copy()
    image_no_Replace=image.copy()
    image_Re=image.copy()
    draw_detections_by_wireframe(image, bbox)

    bbox =cluster_completion_HW(bbox,image_ori)
    bbox=ruleAlignment(bbox)
    bbox=np.rint(bbox).astype(int)

    # glass_ori_boxes = glassInfer(bbox,image_ori)
    # draw_detections_by_wireframe(image_ori, bbox)
    # draw_detections_by_wireframe(image_ori, glass_ori_boxes,fillColor=(110,210,130))
    
    # glass_No_Replace_boxes = glassRegularization_No_Replace(bbox,image_no_Replace)
    # draw_detections_by_wireframe(image_no_Replace, bbox)
    # draw_detections_by_wireframe(image_no_Replace, glass_No_Replace_boxes,fillColor=(110,210,130))

    glass_Regular_boxes = glassRegularizationReplace(bbox,image_Re)
    draw_detections_by_wireframe(image_Re, bbox)
    draw_detections_by_wireframe(image_Re, glass_Regular_boxes,fillColor=(110,210,130))
    pltShow(image,image_Re)
    # pltShow(image,image_ori,image_no_Replace,image_Re)

def infer_completion_regularization(image_path):
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
        if class_ids[i] == 2:
            bbox.append(box)
    image_ori=image.copy()
    draw_detections_by_wireframe(image_ori, bbox)


    image_com=image.copy()
    len_bbox = len(bbox)
    com_boxes= cluster_completion_HW(bbox,image_com)
    len_com_boxes = len(com_boxes)
    new_boxes = com_boxes[len_bbox:len_com_boxes]
    before_boxes = com_boxes[:len_bbox]
    draw_detections_by_wireframe(image_com, before_boxes)
    draw_detections_by_wireframe(image_com, new_boxes,fillColor=(110,210,130))
    
    # draw_detections(image_com, com_boxes)

    regular_boxes= ruleAlignment(com_boxes)
    image_regular=image.copy()
    # draw_detections(image_regular, regular_boxes)
    pltShow(image_ori,image_com,image_regular)
    cvSave(image_ori, image_com, image_regular)
    # pltShow(image_com,)
    # pltShow(image_regular)
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
    # image_path = r"E:\WorkSpace\FacadeRegularization\dataset\all\00334.jpg"
    # singleImageInference(image_path)


    """
    流程展示
    """

    # image_path = r"E:\WorkSpace\FacadeRegularization\dataset\all\00050.jpg"
    # infer_completion_regularization(image_path)
    """
    玻璃补全可视化
    """
    image_path = r"E:\WorkSpace\FacadeRegularization\dataset\citydataset\1_poly4.jpg"
    glassMain(image_path)
if __name__ == "__main__":
   main()