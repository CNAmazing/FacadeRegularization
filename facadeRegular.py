import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 输入图片尺寸和xywh的box信息
image_width = 800
image_height = 600
facade_Para={

    'windows':[[100, 150, 200, 300],[400, 250, 150, 200]],
    'doors':[[50, 50, 20, 20]],
    
}

# 创建一个子图布局 (1行2列)
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1行2列的子图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.xlim(0, image_width)
plt.ylim(image_height, 0)
for key,value in facade_Para.items():
        match key:
            case 'windows':
                for box in value:
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2], box[3], 
                        linewidth=2, edgecolor='b', facecolor='none'
                    )
                    plt.gca().add_patch(rect)
            case 'doors':
                for box in value:
                    rect = patches.Rectangle(
                        (box[0], box[1]), box[2], box[3], 
                        linewidth=2, edgecolor='g', facecolor='none'
                    )
                    plt.gca().add_patch(rect)
plt.subplot(1, 2, 2)
plt.title('Original Image')
plt.xlim(0, image_width)
plt.ylim(image_height, 0)

# 调整子图间距
plt.tight_layout()

# 显示图像
plt.show()

