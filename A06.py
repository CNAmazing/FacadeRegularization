from scipy.optimize import minimize
from sklearn.cluster import MeanShift
import numpy as np
import cv2
import onnxruntime as ort
# from A_05YOLOinfer import YOLO11
import matplotlib.pyplot as plt
CLASS_NAMES = {
    0: 'window',   # 类别 0 名称
    1: 'balcony',   # 类别 1 名称
    2: 'door'    # 类别 1 名称
                        # 可以添加更多类别...
}
class YOLO11:
    """YOLO11 目标检测模型类，用于处理推理和可视化。"""
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        初始化 YOLO11 类的实例。
        参数：
            onnx_model: ONNX 模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
 
        # 加载类别名称
        self.classes = CLASS_NAMES
 
        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
 
    def preprocess(self):
        """
        对输入图像进行预处理，以便进行推理。
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.input_image)
        black_pixels = np.all(self.img == [0, 0, 0], axis=-1)
        self.img[black_pixels] = [114, 114, 114]  
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]
 
        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))
 
        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0
 
        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
 
        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
 
        # 返回预处理后的图像数据
        return image_data
 
 
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        print(f"Original image shape: {shape}")
 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
 
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)
 
        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
 
        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2
 
        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
 
        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        if img.shape[0] != new_shape[0] or img.shape[1] != new_shape[1]:
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        print(f"Final letterboxed image shape: {img.shape}")
 
        return img, (r, r), (dw, dh)
    def postprocess(self, input_image, output):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        参数：
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        # 计算缩放比例和填充
        ratio = self.img_width / self.input_width, self.img_height / self.input_height
 
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
 
                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw  # 移除填充
                y -= self.dh
                x /= self.ratio[0]  # 缩放回原图
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
 
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        print('indices',indices)
        result_boxes = []
        result_scores = []
        result_class_ids = []
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            result_boxes.append(box)
            result_scores.append(score)
            result_class_ids.append(class_id)
            self.draw_detections(input_image, box, score, class_id)
        result={
            'boxes':result_boxes,
            'scores':result_scores,
            'class_ids':result_class_ids
            }
        return result, input_image
    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        参数：
            img: 用于绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测分数。
            class_id: 检测到的目标类别 ID。
        
        返回：
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box
 
        # 获取类别对应的颜色
        color = self.color_palette[class_id]
 
        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
 
        # 创建包含类别名和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"
 
        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
 
        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
 
        # # 绘制填充的矩形作为标签文本的背景
        # cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
 
        # # 在图像上绘制标签文本
        # cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
 
 
    def main(self):
        # 使用 ONNX 模型创建推理会话，自动选择CPU或GPU
        session = ort.InferenceSession(
            self.onnx_model, 
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],
        )
        # 打印模型的输入尺寸
        print("YOLO11 🚀 目标检测 ONNXRuntime")
        print("模型名称：", self.onnx_model)
        
        # 获取模型的输入形状
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape  
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height}")
 
        # 预处理图像数据，确保使用模型要求的尺寸 (640x640)
        img_data = self.preprocess()
       
        # 使用预处理后的图像数据运行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})
 
        # 对输出进行后处理以获取输出图像
        return self.postprocess(self.img, outputs)  # 输出图像

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
def pre_cluster(x: list[float], delta: float) -> list[float]:
    """
    使用MeanShift对一维数据x进行聚类，返回聚类中心
    参数:
        x: 原始一维数据列表
        delta: 带宽参数（控制聚类半径）
    返回:
        聚类中心列表X
    """
    # 将一维数据转为二维 [(x1,0), (x2,0), ...]
    points = np.array([[xi, 0] for xi in x])
    
    # 初始化MeanShift模型（带宽=delta）
    ms = MeanShift(bandwidth=delta)
    ms.fit(points)
    
    # 提取聚类中心（取第一维，忽略补0的维度）
    X = [center[0] for center in ms.cluster_centers_]
    labels = ms.labels_.tolist()
    return X,labels
def Energy(variables,init, X_label, x_w_label, Y_label,y_h_label):
   boxes = []
   for i in range(0, len(variables), 4):
      x1 = variables[i]
      y1 = variables[i + 1]
      w = variables[i + 2]
      h = variables[i + 3]
      boxes.append([x1, y1, w, h])
   E=0
   x_group={}
   y_group={}
   for i,(x1,x2) in enumerate(zip(X_label,x_w_label)):
      key=str(x1)+'_'+str(x2)
      x_group.setdefault(key, []).append(boxes[i])
   for i,(y1,y2) in enumerate(zip(Y_label,y_h_label)):
      key=str(y1)+'_'+str(y2)
      y_group.setdefault(key, []).append(boxes[i])
   
   for key, value in x_group.items():
      value = np.array(value)
      E+=x_error(value)
   for key, value in y_group.items():
      value = np.array(value)
      E+=y_error(value)
   
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
    

def ruleAlignment(boxes):
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
   image_path = r"E:\WorkSpace\FacadeRegularization\dataset\all\00752.jpg"
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
   ruleAlignment(boxes)
if __name__ == "__main__":
   main()