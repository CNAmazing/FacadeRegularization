from scipy.optimize import minimize
import numpy as np
import cv2
from typing import Literal
from Atools import YOLO11,pre_cluster


def cal_iou_subtraction(bbox1, bbox2):
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
    return -np.log(iou + 1e-6)


def x_error(data):
    # 计算第1列（索引0）和第3列（索引2）的均值
    col1_mean = data[:, 0].mean()  # 第1列均值
    col3_mean = data[:, 2].mean()  # 第3列均值
    # print("第1列均值:", col1_mean)  # 输出: 5.0
    # print("第3列均值:", col3_mean)  # 输出: 7.0
    # 计算每个元素与第1列均值的平方误差（仅第1列）
    sq_error_col1 = (data[:, 0] - col1_mean) ** 2  # 仅计算第1列
    # 计算每个元素与第3列均值的平方误差（仅第3列）
    sq_error_col3 = (data[:, 2] - col3_mean) ** 2  # 仅计算第3列
    # 计算平方误差和
    total_sq_error_col1 = sq_error_col1.sum()  # 第1列平方误差和
    total_sq_error_col3 = sq_error_col3.sum()  # 第3列平方误差和
    return total_sq_error_col1+ total_sq_error_col3

def y_error(data):
    # 计算第1列（索引0）和第3列（索引2）的均值
    col1_mean = data[:, 0].mean()  # 第1列均值
    col3_mean = data[:, 2].mean()  # 第3列均值
    # print("第1列均值:", col1_mean)  # 输出: 5.0
    # print("第3列均值:", col3_mean)  # 输出: 7.0
    # 计算每个元素与第1列均值的平方误差（仅第1列）
    sq_error_col1 = (data[:, 1] - col1_mean) ** 2  # 仅计算第1列
    # 计算每个元素与第3列均值的平方误差（仅第3列）
    sq_error_col3 = (data[:, 3] - col3_mean) ** 2  # 仅计算第3列
    # 计算平方误差和
    total_sq_error_col1 = sq_error_col1.sum()  # 第1列平方误差和
    total_sq_error_col3 = sq_error_col3.sum()  # 第3列平方误差和
    return total_sq_error_col1+ total_sq_error_col3

def Energy(variables,init, X_label,  Y_label,x_w_label, y_h_label,x_median,y_median, x_w_median, y_h_median):
    boxes = []
    for i in range(0, len(variables), 4):
        x1 = variables[i]
        y1 = variables[i + 1]
        w = variables[i + 2]
        h = variables[i + 3]
        boxes.append([x1, y1, w, h])
    E=0
    init_boxes = np.array(init).reshape(-1, 4)
    boxes= np.array(boxes)
    # E+= cal_iou_subtraction(init_boxes, boxes)
    E+=np.sum(np.linalg.norm(boxes - init_boxes,axis=1))
    for i in range(len(X_label)):
       temp=boxes[X_label==i]
    #    x_mean = np.median(temp[:, 0])
       errors = np.sqrt((temp[:,0] - x_median[i])** 2) 
       E+= errors.sum()
    for i in range(len(Y_label)):
         temp=boxes[Y_label==i]
        #  y_mean = np.median(temp[:, 1])
         errors = np.sqrt((temp[:,1] - y_median[i])** 2) 
         E+= errors.sum()
    for i in range(len(x_w_label)):
        indices = np.where(x_w_label == i)
        temp=boxes[x_w_label==i]
        # x_w_mean = np.median(temp[:, 2])
        errors = np.sqrt((temp[:,2] - x_w_median[i])** 2) 
        E+= errors.sum()
    for i in range(len(y_h_label)):
        indices = np.where(y_h_label == i)
        temp=boxes[y_h_label==i]
        # y_h_mean = np.median(temp[:, 3])
        errors = np.sqrt((temp[:,3] - y_h_median[i])** 2) 
        E+= errors.sum()
    return E



def ruleAlignment_EnergySingle(boxes):
    x=[]
    y=[]
    x_w=[]
    y_h=[]
    variables = []
    delta=5
    for box in boxes:
        x.append(box[0])
        y.append(box[1])
        x_w.append(box[0]+box[2])
        y_h.append(box[1]+box[3])
        variables.extend([box[0], box[1], box[2], box[3]])
    X,X_label= pre_cluster(x, delta=delta)
    Y,Y_label= pre_cluster(y, delta=delta)
    X_w,x_w_label= pre_cluster(x_w, delta=delta)
    Y_h,y_h_label= pre_cluster(y_h, delta=delta)
    boxes = np.array(boxes)
    X_label = np.array(X_label)
    Y_label = np.array(Y_label)
    x_w_label = np.array(x_w_label)
    y_h_label = np.array(y_h_label)
    X_eq= []
    Y_eq= []
    XW_eq= []
    YH_eq= []
    #X约束
    for i in range(len(X_label)):
        indices = np.where(X_label == i)[0]
        # indices[:]=indices[:]*4
        for j in range(len(indices)-1):
            v=np.zeros(len(x))
            idx=indices[j]
            next_idx=indices[j+1]
            v[idx] = 1
            v[next_idx] = -1
            X_eq.extend([v])
    #Y约束
    for i in range(len(Y_label)):
        indices = np.where(Y_label == i)[0]
        # indices[:]=indices[:]*4+1
        for j in range(len(indices)-1):
            v=np.zeros(len(y))
            idx=indices[j]
            next_idx=indices[j+1]
            v[idx] = 1
            v[next_idx] = -1
            Y_eq.extend([v])
    # X_w约束
    for i in range(len(x_w_label)):
        indices = np.where(x_w_label == i)[0]
        # indices[:]=indices[:]*4+2
        for j in range(len(indices)-1):
            v=np.zeros(len(x_w))
            idx=indices[j]
            next_idx=indices[j+1]
            v[idx] = 1
            v[next_idx] = -1
            XW_eq.extend([v])
    #Y_h约束
    for i in range(len(y_h_label)):
        indices = np.where(y_h_label == i)[0]
        # indices[:]=indices[:]*4+3
        for j in range(len(indices)-1):
            v=np.zeros(len(y_h))
            idx=indices[j]
            next_idx=indices[j+1]
            v[idx] = 1
            v[next_idx] = -1
            YH_eq.extend([v])
    X_eq = np.array(X_eq)
    Y_eq = np.array(Y_eq)
    XW_eq = np.array(XW_eq)
    YH_eq = np.array(YH_eq)
    x_median, y_median, x_w_median, y_h_median=cal_median(boxes, X_label, Y_label, x_w_label, y_h_label)
    print('len(x)',len(x),'len(X_eq)',len(X_eq))  
    print('len(x)',len(y),'len(Y_eq)',len(Y_eq))  
    print('len(x)',len(x_w),'len(XW_eq)',len(XW_eq))  
    print('len(x)',len(y_h),'len(YH_eq)',len(YH_eq))  
    result_x = minimize(
                EnergySingle,
                x,
                args=(variables.copy(),X_label,x_median,'X'),
                constraints={'type': 'eq', 'fun': lambda x: X_eq @ x } if len(X_eq) > 0 else None,
                method='trust-constr',
                options={'disp': True} # 显示优化过程
                )
    result_y = minimize(
                EnergySingle,
                y,
                args=(variables.copy(),Y_label,y_median,'Y'),
                constraints={'type': 'eq', 'fun': lambda y: Y_eq @ y } if len(Y_eq) > 0 else None,
                method='trust-constr',
                options={'disp': True} # 显示优化过程
                )
    result_x_w = minimize(
                EnergySingle,
                x_w,
                args=(variables.copy(),x_w_label,x_w_median,'XW'),
                constraints={'type': 'eq', 'fun': lambda x_w: XW_eq @ x_w } if len(XW_eq) > 0 else None,
                method='trust-constr',
                options={'disp': True} # 显示优化过程
                )
    result_y_h = minimize(
                EnergySingle,
                y_h,
                args=(variables.copy(),y_h_label,y_h_median,'YH'),
                constraints={'type': 'eq', 'fun': lambda y_h: YH_eq @ y_h } if len(YH_eq) > 0 else None,
                method='trust-constr',
                options={'disp': True} # 显示优化过程
                )
    boxes=[]
    for x,y,w,h in zip(result_x.x,result_y.x,result_x_w.x,result_y_h.x):
        boxes.append([x, y, w, h])
    return boxes
def EnergySingle(variables,init, X_label,x_median,key:Literal['X','Y','XW','YH'], ):
    boxes = []
    init_boxes = np.array(init).reshape(-1, 4)

    E=0
    match key:
        case 'X':
            for i in range(len(variables)):
                x1 = variables[i]
                y1 = init_boxes[i][1]
                w = init_boxes[i][2]
                h = init_boxes[i][3]
                boxes.append([x1, y1, w, h])
            boxes = np.array(boxes)
            E+=np.sum(np.linalg.norm(boxes - init_boxes,axis=1))
            # E+= cal_iou_subtraction(init_boxes, boxes)
            for i in range(len(X_label)):
                temp=boxes[X_label==i]
                #    x_mean = np.median(temp[:, 0])
                errors = np.sqrt((temp[:,0] - x_median[i])** 2) 
                E+= errors.sum()
        case 'Y':
            for i in range(len(variables)):
                x1 = init_boxes[i][0]
                y1 = variables[i]
                w = init_boxes[i][2]
                h = init_boxes[i][3]
                boxes.append([x1, y1, w, h])
            boxes = np.array(boxes)
            E+=np.sum(np.linalg.norm(boxes - init_boxes,axis=1))
            # E+= cal_iou_subtraction(init_boxes, boxes)
            for i in range(len(X_label)):
                temp=boxes[X_label==i]
                #    x_mean = np.median(temp[:, 0])
                errors = np.sqrt((temp[:,1] - x_median[i])** 2) 
                E+= errors.sum()
        case 'XW':
            for i in range(len(variables)):
                x1 = init_boxes[i][0]
                y1 = init_boxes[i][1]
                w = variables[i]
                h = init_boxes[i][3]
                boxes.append([x1, y1, w, h])
            # E+= cal_iou_subtraction(init_boxes, boxes)
            boxes = np.array(boxes)
            E+=np.sum(np.linalg.norm(boxes - init_boxes,axis=1))
            for i in range(len(X_label)):
                temp=boxes[X_label==i]
                #    x_mean = np.median(temp[:, 0])
                errors = np.sqrt((temp[:,2] - x_median[i])** 2) 
                E+= errors.sum()
        case 'YH':
            for i in range(len(variables)):
                x1 = init_boxes[i][0]
                y1 = init_boxes[i][1]
                w = init_boxes[i][2]
                h = variables[i]
                boxes.append([x1, y1, w, h])
            # E+= cal_iou_subtraction(init_boxes, boxes)
            boxes = np.array(boxes)
            E+=np.sum(np.linalg.norm(boxes - init_boxes,axis=1))
            for i in range(len(X_label)):
                temp=boxes[X_label==i]
                #    x_mean = np.median(temp[:, 0])
                errors = np.sqrt((temp[:,3] - x_median[i])** 2) 
                E+= errors.sum()
    return E

def ruleAlignment_Energy(boxes):
    x=[]
    y=[]
    x_w=[]
    y_h=[]
    variables = []
    for box in boxes:
        x.append(box[0])
        y.append(box[1])
        x_w.append(box[0]+box[2])
        y_h.append(box[1]+box[3])
        variables.extend([box[0], box[1], box[2], box[3]])
    X,X_label= pre_cluster(x, delta=3)
    Y,Y_label= pre_cluster(y, delta=3)
    X_w,x_w_label= pre_cluster(x_w, delta=3)
    Y_h,y_h_label= pre_cluster(y_h, delta=3)
    boxes = np.array(boxes)
    X_label = np.array(X_label)
    Y_label = np.array(Y_label)
    x_w_label = np.array(x_w_label)
    y_h_label = np.array(y_h_label)
    A_eq= []
    #X约束
    for i in range(len(X_label)):
        indices = np.where(X_label == i)[0]
        indices[:]=indices[:]*4
        for i in range(len(indices)-1):
            v=np.zeros(len(variables))
            idx=indices[i]
            next_idx=indices[i+1]
            v[idx] = 1
            v[next_idx] = -1
            A_eq.extend([v])
    #Y约束
    for i in range(len(Y_label)):
        indices = np.where(Y_label == i)[0]
        indices[:]=indices[:]*4+1
        for i in range(len(indices)-1):
            v=np.zeros(len(variables))
            idx=indices[i]
            next_idx=indices[i+1]
            v[idx] = 1
            v[next_idx] = -1
            A_eq.extend([v])
    # X_w约束
    for i in range(len(x_w_label)):
        indices = np.where(x_w_label == i)[0]
        indices[:]=indices[:]*4+2
        for i in range(len(indices)-1):
            v=np.zeros(len(variables))
            idx=indices[i]
            next_idx=indices[i+1]
            v[idx] = 1
            v[next_idx] = -1
            A_eq.extend([v])
    #Y_h约束
    for i in range(len(y_h_label)):
        indices = np.where(y_h_label == i)[0]
        indices[:]=indices[:]*4+3
        for i in range(len(indices)-1):
            v=np.zeros(len(variables))
            idx=indices[i]
            next_idx=indices[i+1]
            v[idx] = 1
            v[next_idx] = -1
            A_eq.extend([v])
    A_eq = np.array(A_eq)
    x_median, y_median, x_w_median, y_h_median=cal_median(boxes, X_label, Y_label, x_w_label, y_h_label)
    result = minimize(
                Energy,
                variables,
                args=(variables.copy(),X_label,Y_label, x_w_label, y_h_label,x_median,y_median, x_w_median, y_h_median),
                constraints={'type': 'eq', 'fun': lambda x: A_eq @ x } if len(A_eq) > 0 else None,
                method='SLSQP',
                options={'disp': True} # 显示优化过程
                )
    
    jie=result.x
    boxes_result = []
    for i in range(0, len(jie), 4):
        boxes_result.append([jie[i], jie[i + 1], jie[i + 2], jie[i + 3]])
    return boxes_result
def cal_median(boxes, X_label, Y_label, x_w_label, y_h_label):
    x_median, y_median, x_w_median, y_h_median = [], [], [], []
    for i in range(len(X_label)):
       temp=boxes[X_label==i]
       x_median.append(np.median(temp[:, 0]))
    for i in range(len(Y_label)):
        temp=boxes[Y_label==i]
        y_median.append(np.median(temp[:, 1]))
    for i in range(len(x_w_label)):
        temp=boxes[x_w_label==i]
        x_w_median.append(np.median(temp[:, 2]))
    for i in range(len(y_h_label)):
        temp=boxes[y_h_label==i]
        y_h_median.append(np.median(temp[:, 3]))
    return x_median, y_median, x_w_median, y_h_median
def ruleAlignment(boxes):
    delta=3
    x=[]
    y=[]
    x_w=[]
    y_h=[]
    variables = []
    for box in boxes:
        x.append(box[0])
        y.append(box[1])
        x_w.append(box[0]+box[2])
        y_h.append(box[1]+box[3])
        variables.extend([box[0], box[1], box[2], box[3]])
    X,X_label= pre_cluster(x, delta=delta)
    Y,Y_label= pre_cluster(y, delta=delta)
    X_w,x_w_label= pre_cluster(x_w, delta=delta)
    Y_h,y_h_label= pre_cluster(y_h, delta=delta)
    boxes = np.array(boxes)
    X_label = np.array(X_label)
    Y_label = np.array(Y_label)
    x_w_label = np.array(x_w_label)
    y_h_label = np.array(y_h_label)
    for i in range(len(X_label)):
       indices = np.where(X_label == i)
       temp=boxes[X_label==i]
       x_mean = np.median(temp[:, 0])
       boxes[indices, 0] = x_mean
    for i in range(len(Y_label)):
         indices = np.where(Y_label == i)
         temp=boxes[Y_label==i]
         y_mean = np.median(temp[:, 1])
         boxes[indices, 1] = y_mean
    for i in range(len(x_w_label)):
        indices = np.where(x_w_label == i)
        temp=boxes[x_w_label==i]
        x_w_mean = np.median(temp[:, 2])
        boxes[indices, 2] = x_w_mean
    for i in range(len(y_h_label)):
        indices = np.where(y_h_label == i)
        temp=boxes[y_h_label==i]
        y_h_mean = np.median(temp[:, 3])
        boxes[indices, 3] = y_h_mean

   # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )  # 如果是彩色图，去掉 cmap 参数
   
   # x_group={}
   # y_group={}
   # for i,(x1,x2) in enumerate(zip(X_label,x_w_label)):
   #    key=str(x1)+'_'+str(x2)
   #    x_group.setdefault(key, []).append(boxes[i])
   # for i,(y1,y2) in enumerate(zip(Y_label,y_h_label)):
   #    key=str(y1)+'_'+str(y2)
   #    y_group.setdefault(key, []).append(boxes[i])

#    bounds = [(0, None), ] * len(variables)
#    result = minimize(
#     Energy,          # 目标函数
#     variables,           # 初始猜测值
#     method='L-BFGS-B', # 优化方法
#     args=(variables,X_label, x_w_label, Y_label,y_h_label), # 额外参数
#     bounds=bounds,# 变量边界
#     options={'disp': True} # 显示优化过程
# )
#    jie=result.x
    # boxes_result = []
    # for i in range(0, len(jie), 4):
    #     boxes_result.append([jie[i], jie[i + 1], jie[i + 2], jie[i + 3]])
    return boxes
   # # 在图像上绘制一维数据点（假设 y=50，即水平线上的点）
   # # y = [1] * len(X)  # 所有点的 y 坐标固定为 50
   # plt.scatter(X, [1] * len(X), color='red', s=50)  # 红色圆点，大小 50
   # plt.scatter([1] * len(Y), Y, color='blue', s=50)  # 蓝色圆点，大小 50
   # # 显示图形  
   # plt.show()
   # print('Done')
def main():
   image_path = r"E:\WorkSpace\FacadeRegularization\dataset\all\01253.jpg"
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
   ruleAlignment_EnergySingle(boxes)
if __name__ == "__main__":
   main()