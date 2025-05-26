import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import json
import numpy as np
from typing import Literal
import random
from scipy.optimize import linear_sum_assignment
import copy
from Atools import pre_cluster,YOLO11,plt_show_image,pltShow,draw_detections
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
class Rectangles:
    def __init__(self,rectangles):
        self.rectangles=rectangles
        self.min_Horizontal_Distance,self.min_Vertical_Distance=rectangle_min_distance(rectangles)
        self.boundaries = self.add_boundary()
    def add_boundary(self):
        x_min,y_min=math.inf,math.inf
        x_max,y_max=-math.inf,-math.inf
        for i in range(len(self.rectangles)):
            rect = self.rectangles[i]
            x1, y1, w, h = rect
            # 计算矩形的四个顶点
           
            x_min=min(x_min,x1)
            y_min=min(y_min,y1)
            x_max=max(x_max,x1+w)
            y_max=max(y_max,y1+h)
        
        return (x_min-10,y_min-10,x_max+10,y_max+10)
    def is_Inside(self,rectangle):
        x_min,y_min,x_max,y_max=self.boundaries
        x,y,w,h=rectangle
        if not (x_min<x<x_max and 
            x_min<x+w<x_max and
            y_min<y<y_max and
            y_min<y+h<y_max 
            ):
            return False 
        start1=(x,y)
        end1=(x+w,y+h)
        start2=(x+w,y)
        end2=(x,y+h)
        flag1=not intersection_detection(start1,end1,self.rectangles,self.min_Horizontal_Distance,self.min_Vertical_Distance)
        flag2=not intersection_detection(start2,end2,self.rectangles,self.min_Horizontal_Distance,self.min_Vertical_Distance)
        return flag1 and flag2


def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：{e}")
        return None
def xyxy_to_xywh(bboxes):
    array=[]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        array.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(array)

# 计算矩形中心点
def get_center(rect):
    x1, y1, w, h = rect
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy


# 可视化矩形和标签
def visualize_rectangles(rectangles,before_len,after_len):
    """
    可视化矩形及其相邻关系，并将坐标原点设为左上角（y 轴朝下，x 轴朝右）。
    
    参数:
        rectangles: 矩形列表，格式为 [(x1, y1, width, height), ...]
        nearest_rectangles: 相邻矩形信息，格式为 [(i, {'up': ..., 'down': ..., 'left': ..., 'right': ...}), ...]
        data: 一维数据点，用于绘制散点图（可选）
        labels: 数据点的标签，用于着色（可选）
        x_threshold: x 轴阈值（可选）
        y_threshold: y 轴阈值（可选）
        cls: 类别信息（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制矩形
    for i, rect in enumerate(rectangles[:before_len]):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle((x1, y1), width, height, linewidth=0.5, edgecolor='black', facecolor='none')
        ax.add_patch(rect_patch)
        
        # 标注矩形编号
        cx, cy = get_center(rect)
        ax.text(cx, cy, str(i), fontsize=12, ha='center', va='center', color='blue')
    

    for i, rect in enumerate(rectangles[before_len:after_len],start=before_len):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle((x1, y1), width, height, linewidth=0.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect_patch)
        
        # 标注矩形编号
        cx, cy = get_center(rect)
        ax.text(cx, cy, str(i), fontsize=12, ha='center', va='center', color='blue')

    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))  # 修改：将图例放置在绘图区域
    # 设置坐标轴
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)  # 将 y 轴反转，使原点在左上角
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title("Rectangles and Their Nearest Neighbors")
    plt.show()


def remove_random_element(rectangles,delete_count=1):
    if rectangles.size == 0:
        return rectangles  # 如果数组为空，直接返回
    # 生成随机索引

       
    random_index = np.random.randint(0, len(rectangles), size=delete_count)
    # 删除随机索引处的元素
    result = np.delete(rectangles, random_index, axis=0)
    
    return result
def ray_intersection_detection(start,end,boxes,i,j):
    for idx,box in enumerate(boxes):
        if idx==i or idx==j:
            continue
        x1, y1, w, h = box
        # 计算矩形的四个顶点
        A = (x1, y1)
        B = (x1 + w, y1)
        C = (x1 + w, y1 + h)
        D = (x1, y1 + h)
        # 检查线段是否与矩形相交
        if segments_intersect(start, end, A, B) or segments_intersect(start, end, B, C) or \
           segments_intersect(start, end, C, D) or segments_intersect(start, end, D, A):
            return False
    return True 
def ray_intersection(start,end,boxes,cur_box,tar_box):
    for box in boxes:
        if np.array_equal(box,cur_box) or np.array_equal(box,tar_box):
            continue
        x1, y1, w, h = box
        # 计算矩形的四个顶点
        A = (x1, y1)
        B = (x1 + w, y1)
        C = (x1 + w, y1 + h)
        D = (x1, y1 + h)
        # 检查线段是否与矩形相交
        if segments_intersect(start, end, A, B) or segments_intersect(start, end, B, C) or \
           segments_intersect(start, end, C, D) or segments_intersect(start, end, D, A):
            return False
    return True 
def intersection_detection(start,end,rectangles,min_Horizontal_Distance=0,min_Vertical_Distance=0):
    """
    有相交或重叠的矩形返回True，否则返回False
    """
    for rect in rectangles:
      
        x1, y1, w, h = rect
        x1=x1-min_Horizontal_Distance
        y1=y1-min_Vertical_Distance
        w=w+min_Horizontal_Distance*2
        h=h+min_Vertical_Distance*2
        # 计算矩形的四个顶点
        A = (x1, y1)
        B = (x1 + w, y1)
        C = (x1 + w, y1 + h)
        D = (x1, y1 + h)
        # 检查线段是否与矩形相交
        if segments_intersect(start, end, A, B) or segments_intersect(start, end, B, C) or \
           segments_intersect(start, end, C, D) or segments_intersect(start, end, D, A):
            return True
        
         
        if  (   x1 <= start[0]<=x1+w and
                x1 <= end[0]<=x1+w and
                y1 <= start[1]<=y1+h and
                y1 <= end[1]<=y1+h 
            ):
            return True
    return False 
def rectangle_neiborhood(rectangles):
    relations = []
    for i in range(len(rectangles)):
        neighbor=[]
        vectorGroup = []
        rect1 = rectangles[i]
        for j in range(len(rectangles)):
            if j == i:
                continue
            rect2 = rectangles[j]
            r_w,r_h=rect2[2],rect2[3]
            # 计算矩形的中心点
            start,end=get_StartAndEnd(rect1,rect2)
            # 计算向量
            vec_ = tuple([end[0] - start[0], end[1] - start[1],r_w,r_h])
            # 检查射线是否与其他矩形相交
            if ray_intersection_detection(start,end,rectangles,i,j):
                neighbor.append(j)
                vectorGroup.append(tuple(vec_))
            
        relations.append((i, tuple(neighbor), tuple(vectorGroup))) 
    return relations  
def  rectangle_min_distance(rectangles):
    """
    计算矩形之间的最小距离
    :param rectangles: 矩形列表 [(x1, y1, w, h), ...]
    :return: 最小距离
    """
    min_Horizontal_Distance = float('inf')
    min_Vertical_Distance = float('inf')
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            rect1 = rectangles[i]
            rect2 = rectangles[j]
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            # 计算矩形的中心点
            cx1 = x1 + w1 / 2
            cy1 = y1 + h1 / 2
            cx2 = x2 + w2 / 2
            cy2 = y2 + h2 / 2
            if cx1<=cx2 and abs(cx1-cx1)>max(w1,w2)/2:
                min_Horizontal_Distance = min(min_Horizontal_Distance, cx2 - cx1-w1/2-w2/2)
            elif cx1>cx2 and abs(cx1-cx2)>max(w1,w2)/2:
                min_Horizontal_Distance = min(min_Horizontal_Distance, cx1 - cx2-w1/2-w2/2)
            else :
                continue

            if cy1<=cy2 and abs(cy1-cy2)>max(h1,h2)/2: 
                min_Vertical_Distance = min(min_Vertical_Distance, cy2 - cy1-h1/2-h2/2)
            elif cy1>cy2 and abs(cy1-cy2)>max(h1,h2)/2:
                min_Vertical_Distance = min(min_Vertical_Distance, cy1 - cy2-h1/2-h2/2)
            else:
                continue
    return min_Horizontal_Distance, min_Vertical_Distance
def segments_intersect(A, B, C, D):
    # Unpack points
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    
    # Compute bounding boxes
    def bounding_box_intersect():
        return (max(x1, x2) >= min(x3, x4) and
                min(x1, x2) <= max(x3, x4) and
                max(y1, y2) >= min(y3, y4) and
                min(y1, y2) <= max(y3, y4))
    
    if not bounding_box_intersect():
        return False
    
    # Compute cross products
    def cross_product(a, b, c):
        return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
    
    d1 = cross_product(A, B, C)
    d2 = cross_product(A, B, D)
    d3 = cross_product(C, D, A)
    d4 = cross_product(C, D, B)
    
    # Check straddling
    if (d1 * d2 < 0) and (d3 * d4 < 0):
        return True
    
    # Check collinear overlapping
    if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
        # Check overlap in x or y
        def on_segment(a, b, c):
            return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                    min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
        
        return (on_segment(A, B, C) or on_segment(A, B, D) or
                on_segment(C, D, A) or on_segment(C, D, B))
    
    return False
def get_StartAndEnd(rect1,rect2):
    """
    计算两个矩形的邻域关系
    :param rect1: 第一个矩形 (x1, y1, w, h)
    :param rect2: 第二个矩形 (x1, y1, w, h)
    :return: 邻域关系字典
    """
    x1_1, y1_1, w1_1, h1_1 = rect1
    x2_2, y2_2, w2_2, h2_2 = rect2
    
    # 计算中心点坐标
    cx1 = x1_1 + w1_1 / 2
    cy1 = y1_1 + h1_1 / 2
    cx2 = x2_2 + w2_2 / 2
    cy2 = y2_2 + h2_2 / 2
    
    start=(cx1, cy1)
    end=(cx2, cy2)
    return start,end
def pairwise_L2_distance(A, B):
    """计算 A 和 B 之间的 L2 距离矩阵"""
    v1= np.array(A)
    v2= np.array(B)
    
    return  np.linalg.norm(v1 - v2)
def L2_And_Size_distance(A, B,w1=0.5):
    """计算 A 和 B 之间的 L2 距离矩阵"""
    v1= np.array(A)
    v2= np.array(B)
    vector_error = np.linalg.norm(v1[:2] - v2[:2])
    # size_error = abs(v1[2]*v1[3] - v2[2]*v2[3])
    size_error=np.linalg.norm(v1[2:] - v2[2:])
    # error=np.linalg.norm(v1 - v2)
    return  w1*vector_error + (1-w1)*size_error
    return  error
    # return  size_error*vector_errors
def L2_And_Size_distance_single(A):
    """计算 A 和 B 之间的 L2 距离矩阵"""
    v1= np.array(A)
    vector_error = np.linalg.norm(v1[:2])
    size_error = abs(v1[2]*v1[3])
    # error=np.linalg.norm(v1)
    # return  error
    return  vector_error + size_error
def similar_grow():
    # 读取 JSON 文件
    data = read_json(r'E:\WorkSpace\FacadeRegularization\data3.json')

    # 矩形表示: (x1, y1, w, h)
    rectangles = data['poly4']['window']
    # rectangles = xyxy_to_xywh(rectangles)
    # rectangles=remove_random_element(rectangles,delete_count=4)
    relations = rectangle_neiborhood(rectangles)
    min_Horizontal_Distance, min_Vertical_Distance = rectangle_min_distance(rectangles)
    for r in relations:
        print(r)
        print("\n") 
    # print("邻域关系：", relations)
    # k = 2
    result=[]
    for i in range(len(relations)):
        idx=-1
        min_error = math.inf
        min_indices_Group = None
        cur_Group=relations[i][2]
        min_vector_Group=None
        for j in range(len(relations)):
            if i == j:
                continue
            tar_Group=relations[j][2]
            m,n=len(cur_Group),len(tar_Group),
            cost_matrix=np.zeros((m,n))
            for temp_i in range(m):
                for temp_j in range(n):
                    cost_matrix[temp_i][temp_j]=L2_And_Size_distance(cur_Group[temp_i],tar_Group[temp_j])
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            error1 = cost_matrix[row_ind, col_ind].sum()
            if m>n:
                all_rows = set(range(cost_matrix.shape[1]))
                matched_rows = set(row_ind)
                unmatched_rows = all_rows - matched_rows
                for row_idx in unmatched_rows:
                    temp_error=L2_And_Size_distance_single(tar_Group[row_idx])
                    error1+=temp_error
            elif n>m:
                all_cols = set(range(cost_matrix.shape[0]))
                matched_cols = set(col_ind)
                unmatched_cols = all_cols - matched_cols
                for col_idx in unmatched_cols:
                    temp_error=L2_And_Size_distance_single(tar_Group[col_idx])
                    error1+=temp_error

            cost_matrix_reverse=np.zeros((m,n))
            tar_Group_copy=np.array(tar_Group)
            tar_Group_copy[:,:1]=tar_Group_copy[:,:1]*(-1)
            for temp_i in range(m):
                for temp_j in range(n):
                    cost_matrix_reverse[temp_i][temp_j]=L2_And_Size_distance(cur_Group[temp_i],tar_Group_copy[temp_j])
            row_ind, col_ind = linear_sum_assignment(cost_matrix_reverse)
            error2= cost_matrix_reverse[row_ind, col_ind].sum()
            if m>n:
                all_rows = set(range(cost_matrix.shape[1]))
                matched_rows = set(row_ind)
                unmatched_rows = all_rows - matched_rows
                for row_idx in unmatched_rows:
                    temp_error=L2_And_Size_distance_single(tar_Group[row_idx])
                    error2+=temp_error
            elif n>m:
                all_cols = set(range(cost_matrix.shape[0]))
                matched_cols = set(col_ind)
                unmatched_cols = all_cols - matched_cols
                for col_idx in unmatched_cols:
                    temp_error=L2_And_Size_distance_single(tar_Group[col_idx])
                    error2+=temp_error
            temp_vector_Group=None
            # temp_vector_Group=tar_Group
            if error1<error2:
                error=error1
                temp_vector_Group=tar_Group
            else:
                error=error2
                temp_vector_Group=tar_Group_copy
            
            if error < min_error:
                min_error = error
                idx=j
                min_vector_Group=temp_vector_Group
        
        print(f"最小距离的邻域关系为：{relations[i][0]} and {relations[idx][0]}")
        print(f"最小距离为：{min_error}")
        
        result.append({
            'id':i,
            'neighbor':idx,
            'min_error':min_error,
            'min_vector_Group':min_vector_Group,
        })
    before_len=len(rectangles)
    sorted_result = sorted(result, key=lambda x: x['min_error'])
    # rectangles=rectangles.tolist()
    rectangles_copy=copy.deepcopy(rectangles)
    for s in sorted_result: 
        rect=rectangles_copy[s['id']]
        x,y,w,h=rect
        centroid=(x+w/2,y+h/2)
        min_vector_Group=s['min_vector_Group']
        rect_box=Rectangles(rectangles_copy)
        for g in min_vector_Group:
            v=g[:2]
            g_w,g_h=g[2:]
            tar_centroid_x=centroid[0]+v[0]
            tar_centroid_y=centroid[1]+v[1]
            start=tuple([tar_centroid_x-g_w/2,tar_centroid_y-g_h/2])
            end=tuple([tar_centroid_x+g_w/2,tar_centroid_y+g_h/2])
            newRect=[start[0], start[1],g_w,g_h]
            # if rect_box.is_Inside(newRect):
            if 1:
                print(f"当前索引对为{s['id']}和{s['neighbor']}")
                
                rectangles_copy.append(newRect)
                rect_box.rectangles=rectangles_copy
                print(f"已添加矩形{len(rectangles_copy)}")
            # if not intersection_detection(start,end,rectangles):
            #     rectangles.append([start[0], start[1], w, h]) 
    after_len=len(rectangles_copy)
    visualize_rectangles(rectangles_copy,before_len,after_len)

    for s in sorted_result: 
        rect=rectangles[s['id']]
        x,y,w,h=rect
        centroid=(x+w/2,y+h/2)
        min_vector_Group=s['min_vector_Group']
        rect_box=Rectangles(rectangles)
        for g in min_vector_Group:
            v=g[:2]
            g_w,g_h=g[2:]
            tar_centroid_x=centroid[0]+v[0]
            tar_centroid_y=centroid[1]+v[1]
            start=tuple([tar_centroid_x-g_w/2,tar_centroid_y-g_h/2])
            end=tuple([tar_centroid_x+g_w/2,tar_centroid_y+g_h/2])
            newRect=[start[0], start[1],g_w,g_h]
            if rect_box.is_Inside(newRect):
            # if 1:
                print(f"当前索引对为{s['id']}和{s['neighbor']}")
                
                rectangles.append(newRect)
                rect_box.rectangles=rectangles
                print(f"已添加矩形{len(rectangles)}")
            # if not intersection_detection(start,end,rectangles):
            #     rectangles.append([start[0], start[1], w, h]) 
    after_len=len(rectangles)
    visualize_rectangles(rectangles,before_len,after_len)
def boxes_classification(boxes,X_label,Y_label,x_w_label,y_h_label):
    x_group={}
    y_group={}
    for i,(x1,x2) in enumerate(zip(X_label,x_w_label)):
        key=str(x1)+'_'+str(x2)
        if key not in x_group:
            x_group[key] = {
                'boxes': [],  # 原来的列表换个名字存储
                'vec': []    # 新的键值对
            }
        x_group[key]['boxes'].append(boxes[i])
        # x_group.setdefault(key, []).append(boxes[i])
    for i,(y1,y2) in enumerate(zip(Y_label,y_h_label)):
        key=str(y1)+'_'+str(y2)
        if key not in y_group:
            y_group[key] = {
                'boxes': [],  # 原来的列表换个名字存储
                'vec': []    # 新的键值对
            }
        y_group[key]['boxes'].append(boxes[i])
        # y_group.setdefault(key, []).append(boxes[i])
    
    return x_group,y_group
def boxes_classification_HW(boxes,X_label,x_w_label,):
    x_group={}
   
    for i,(x1,x2) in enumerate(zip(X_label,x_w_label)):
        key=str(x1)+'_'+str(x2)
        if key not in x_group:
            x_group[key] = {
                'boxes': [],  # 原来的列表换个名字存储
                'vec': []    # 新的键值对
            }
        x_group[key]['boxes'].append(boxes[i])
        # x_group.setdefault(key, []).append(boxes[i])
    return x_group
def add_relations(boxes,x_group):
    for key,value in x_group.items():
        cur_boxes=value['boxes']
        if len(cur_boxes)==1:
            continue
        for cur_box in cur_boxes:
            for tar_box in boxes:
                if np.array_equal(cur_box,tar_box):
                    continue
                start,end=get_StartAndEnd(cur_box,tar_box)
                vec_=tuple([end[0] - start[0], end[1] - start[1],tar_box[2],tar_box[3]])
                if ray_intersection(start,end,boxes,cur_box,tar_box):
                    # print(f"当前索引对为{cur_box}和{tar_box}")
                    # print(f"已添加矩形{len(rectangles)}")
                    value['vec'].append(vec_)
def is_rect_inside(rect_a, rect_b):
    """判断矩形A是否完全在矩形B内部"""

    a_left, a_top, a_right, a_bottom = rect_a
    a_right+= a_left
    a_bottom+= a_top
    
    b_left, b_top, b_right, b_bottom = rect_b
    b_right+= b_left
    b_bottom+= b_top
    return (
        a_left >= b_left and
        a_right <= b_right and
        a_bottom <= b_bottom and
        a_top >= b_top
    )
def is_rect_overlap(rect_a, rect_b):
    # rect = (x_min, y_min, x_max, y_max)
    
    a_left, a_top, a_right, a_bottom = rect_a
    a_right+= a_left
    a_bottom+= a_top
    b_left, b_top, b_right, b_bottom = rect_b
    b_right+= b_left
    b_bottom+= b_top
    return not (a_right < b_left or 
                a_left > b_right or 
                a_top > b_bottom or 
                a_bottom < b_top) 
def intersect_Boxes(boxes, newRect):
    for tar_box in boxes:
        if  is_rect_inside(newRect,tar_box) or  is_rect_inside(tar_box,newRect):
            return True
        if  is_rect_overlap(newRect,tar_box):
            return True
    return False
def generate_box(x_group,boxes,x_len,y_len,xw_len,yh_len,image):
    imageH,imageW,_=image.shape

    for key,value in x_group.items():
            cur_boxes=value['boxes']
            vec_s= value['vec']
            if len(cur_boxes)==1:
                continue
            for cur_box in cur_boxes:
                for vec_ in vec_s:
                    newRect=[cur_box[0]+vec_[0], cur_box[1]+vec_[1],vec_[2],vec_[3]]
                    if intersect_Boxes(boxes, newRect):
                        continue    
                    if not is_rect_inside(newRect,(0,0,imageW,imageH)):
                        continue
                    newboxes=deepcopy(boxes)
                    newboxes=np.vstack((newboxes, newRect))
                    newboxes_X=newboxes[:,0]
                    newboxes_Y=newboxes[:,1]
                    newboxes_X_w=newboxes[:,0]+newboxes[:,2]
                    newboxes_Y_h=newboxes[:,1]+newboxes[:,3]
                    new_X,new_X_label=pre_cluster(newboxes_X, delta=3)
                    new_Y,new_Y_label=pre_cluster(newboxes_Y, delta=3)
                    new_X_w,new_X_w_label=pre_cluster(newboxes_X_w, delta=3)
                    new_Y_h,new_Y_h_label=pre_cluster(newboxes_Y_h, delta=3)
                    conditions=[
                        len(new_X)<=x_len and len(new_X_w)<=xw_len,
                        len(new_Y)<=y_len and len(new_Y_h)<=yh_len,
                    ]
                    trueCount=sum(conditions)
                    match trueCount:
                        case 0:
                            continue
                        case 2:
                            boxes=np.vstack((boxes, newRect))
                            draw_detections(image,boxes)
                            # pltShow(image)
    return boxes
def cluster_completion(boxes,image):
    
    
    # X,X_label= pre_cluster(x, delta=3)
    # Y,Y_label= pre_cluster(y, delta=3)
    # X_w,x_w_label= pre_cluster(x_w, delta=3)
    # Y_h,y_h_label= pre_cluster(y_h, delta=3)
    # boxes = np.array(boxes)
    # X_label = np.array(X_label)
    # Y_label = np.array(Y_label)
    # x_w_label = np.array(x_w_label)
    # y_h_label = np.array(y_h_label)
    # x_group,y_group=boxes_classification(boxes,X_label,Y_label,x_w_label,y_h_label)
    # add_relations(boxes,x_group)
    # add_relations(boxes,y_group)
    # x_len,y_len,xw_len,yh_len=len(X),len(Y),len(X_w),len(Y_h)
    cur_len=len(boxes)
    last_len=0
    while last_len<cur_len:
        x=[]
        y=[]
        x_w=[]
        y_h=[]
        for box in boxes:
            x.append(box[0])
            y.append(box[1])
            x_w.append(box[0]+box[2])
            y_h.append(box[1]+box[3])
        X,X_label= pre_cluster(x, delta=3)
        Y,Y_label= pre_cluster(y, delta=3)
        X_w,x_w_label= pre_cluster(x_w, delta=3)
        Y_h,y_h_label= pre_cluster(y_h, delta=3)
        boxes = np.array(boxes)
        X_label = np.array(X_label)
        Y_label = np.array(Y_label)
        x_w_label = np.array(x_w_label)
        y_h_label = np.array(y_h_label)
        x_group,y_group=boxes_classification(boxes,X_label,Y_label,x_w_label,y_h_label)

        add_relations(boxes,x_group)
        add_relations(boxes,y_group)
        x_len,y_len,xw_len,yh_len=len(X),len(Y),len(X_w),len(Y_h)

        last_len=len(boxes)
        # print(f"before len{cur_len}")
        boxes=generate_box(y_group,boxes,x_len,y_len,xw_len,yh_len,image)
        boxes=generate_box(x_group,boxes,x_len,y_len,xw_len,yh_len,image)
       
        cur_len=len(boxes)
        # print(f"after len{cur_len}")
    return boxes
    
def cluster_completion_HW(boxes,image):
    cur_len=len(boxes)
    last_len=0
    while last_len<cur_len:
        x=[]
        y=[]
        x_w=[]
        y_h=[]
        w=[]
        h=[]
        for box in boxes:
            x.append(box[0])
            y.append(box[1])
            x_w.append(box[0]+box[2])
            y_h.append(box[1]+box[3])
            w.append(box[2])
            h.append(box[3])
        X,X_label= pre_cluster(x, delta=3)
        Y,Y_label= pre_cluster(y, delta=3)
        X_w,x_w_label= pre_cluster(x_w, delta=3)
        Y_h,y_h_label= pre_cluster(y_h, delta=3)
        W,W_label= pre_cluster(w, delta=3)
        H,H_label= pre_cluster(h, delta=3)

        boxes = np.array(boxes)
        X_label = np.array(X_label)
        Y_label = np.array(Y_label)
        x_w_label = np.array(x_w_label)
        y_h_label = np.array(y_h_label)
        # x_group,y_group=boxes_classification(boxes,X_label,Y_label,x_w_label,y_h_label)
        HW_group=boxes_classification_HW(boxes,W_label,H_label,)
        add_relations(boxes,HW_group)
        # add_relations(boxes,y_group)
        x_len,y_len,xw_len,yh_len=len(X),len(Y),len(X_w),len(Y_h)

        last_len=len(boxes)
        # print(f"before len{cur_len}")
        boxes=generate_box(HW_group,boxes,x_len,y_len,xw_len,yh_len,image)
        # boxes=generate_box(x_group,boxes,x_len,y_len,xw_len,yh_len,image)
       
        cur_len=len(boxes)
        # print(f"after len{cur_len}")
    return boxes
def main():
    image_path = r"E:\WorkSpace\FacadeRegularization\dataset\poly3.jpg"
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
    height,width,_=image.shape
    boxes = result['boxes']
    # pltShow(output_image)
    newboxes=cluster_completion(boxes,height,width,image)
    draw_detections(image,boxes)
    pltShow(output_image,image)
if __name__ == "__main__":
    main(   )