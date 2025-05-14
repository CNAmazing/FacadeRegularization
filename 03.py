import numpy as np
from sklearn.cluster import AgglomerativeClustering
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def get_center(rect):
        x1, y1, w, h = rect
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy
def xyxy_to_xywh(bboxes):
    array=[]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        array.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(array)
def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：{e}")
        return None
def SLOD2_WIN(facade_img, detected_windows, lambda_scale=16, max_clusters=3):
    """
    Align detected windows via clustering-based alignment.
    
    Args:
        facade_img: Input facade image (for dimensions).
        detected_windows: List of detected window bounding boxes in format 
                         [(x1, y1, x2, y2, shape_type), ...].
        lambda_scale: Scale factor for width/height thresholds (default=16).
        max_clusters: Max clusters for width/height (default=3).
    
    Returns:
        aligned_windows: List of aligned bounding boxes.
        cluster_labels: Cluster labels for each window.
    """
    # Extract window properties
    centers = []
    widths = []
    heights = []
    for (x1, y1, w, h) in detected_windows:
        centers.append([x1 + w/ 2, y1 + h / 2])  # Center coordinates
        widths.append(w)                           # Width
        heights.append(h)                          # Height
    
    centers = np.array(centers)
    widths = np.array(widths)
    heights = np.array(heights)
    
    # --- Step 1: Center Alignment ---
    # Calculate thresholds (Eq.1)
    min_width, min_height = np.min(widths), np.min(heights)
    gamma_cx = min_width / 2
    gamma_cy = min_height / 2
    
    # Cluster centers (horizontal and vertical separately)
    clustering_cx = AgglomerativeClustering(
        n_clusters=None, 
        linkage='complete', 
        distance_threshold=gamma_cx
    ).fit(centers[:, 0].reshape(-1, 1))
    
    clustering_cy = AgglomerativeClustering(
        n_clusters=None, 
        linkage='complete', 
        distance_threshold=gamma_cy
    ).fit(centers[:, 1].reshape(-1, 1))
    
    # Update centers to cluster means
    aligned_centers = centers.copy()
    for i in np.unique(clustering_cx.labels_):
        mask = (clustering_cx.labels_ == i)
        aligned_centers[mask, 0] = np.mean(centers[mask, 0])
    
    for i in np.unique(clustering_cy.labels_):
        mask = (clustering_cy.labels_ == i)
        aligned_centers[mask, 1] = np.mean(centers[mask, 1])
    
    # --- Step 2: Width/Height Alignment ---
    # Calculate thresholds (Eq.2)
    h, w = facade_img.shape[:2]
    N = len(detected_windows)
    gamma_w = w * np.sqrt(np.max(widths) - np.min(widths)) / (lambda_scale * N**2)
    gamma_h = h * np.sqrt(np.max(heights) - np.min(heights)) / (lambda_scale * N**2)
    
    # Cluster widths/heights with constraints
    def constrained_clustering(features, gamma, max_clusters):
        clustering = AgglomerativeClustering(
            n_clusters=None,
            linkage='complete',
            distance_threshold=gamma
        ).fit(features.reshape(-1, 1))
        
        # Enforce max_clusters by doubling threshold if needed
        while len(np.unique(clustering.labels_)) > max_clusters:
            gamma *= 2
            clustering = AgglomerativeClustering(
                n_clusters=None,
                linkage='complete',
                distance_threshold=gamma
            ).fit(features.reshape(-1, 1))
        return clustering
    
    clustering_w = constrained_clustering(widths, gamma_w, max_clusters)
    clustering_h = constrained_clustering(heights, gamma_h, max_clusters)
    
    # Update widths/heights to cluster means
    aligned_widths = widths.copy()
    aligned_heights = heights.copy()
    for i in np.unique(clustering_w.labels_):
        mask = (clustering_w.labels_ == i)
        aligned_widths[mask] = np.mean(widths[mask])
    
    for i in np.unique(clustering_h.labels_):
        mask = (clustering_h.labels_ == i)
        aligned_heights[mask] = np.mean(heights[mask])
    
    # --- Step 3: Generate Aligned Bounding Boxes ---
    aligned_windows = []
    for i in range(len(detected_windows)):
        cx, cy = aligned_centers[i]
        new_w, new_h = aligned_widths[i], aligned_heights[i]
        x1 = int(cx - new_w / 2)
        y1 = int(cy - new_h / 2)
        x2 = int(new_w)
        y2 = int(new_h)
        aligned_windows.append((x1, y1, x2, y2))
    
    # Combined cluster labels (for visualization)
    cluster_labels = clustering_w.labels_ * max_clusters + clustering_h.labels_
    
    return aligned_windows, cluster_labels


# --- Example Usage ---
if __name__ == "__main__":
    # Mock input: facade image and detected windows (x1, y1, x2, y2, shape_type)
    facade_img = np.zeros((600, 800, 3), dtype=np.uint8)  # Placeholder image
    # detected_windows = [
    #     (100, 100, 180, 220, "rect"),  # Window 1
    #     (120, 110, 190, 230, "rect"),  # Window 2 (slightly misaligned)
    #     (400, 100, 480, 220, "rect"),  # Window 3
    #     (420, 110, 490, 230, "rect"),  # Window 4 (misaligned)
    #     (100, 400, 180, 520, "arch"),  # Window 5 (different height)
    # ]
    data = read_json(r'E:\WorkSpace\FacadeRegularization\data2.json')
    # 矩形表示: (x1, y1, w, h)
    detected_windows = data['window']
    detected_windows = xyxy_to_xywh(detected_windows)
        # Align windows
    aligned_windows, cluster_labels = SLOD2_WIN(facade_img, detected_windows)
    rect1 = detected_windows
    rect2 = aligned_windows
    # # Visualize results
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10))
    # 绘制rect1（蓝色边框+蓝色编号）
    for i, rect in enumerate(rect1):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=1, edgecolor='blue', 
            facecolor='none', linestyle='-'
        )
        ax.add_patch(rect_patch)
        cx, cy = get_center(rect)
        ax.text(cx, cy, f"1-{i}", fontsize=10, 
                ha='center', va='center', color='blue')

    # 绘制rect2（红色边框+红色编号）
    for i, rect in enumerate(rect2):
        x1, y1, width, height = rect
        rect_patch = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=1, edgecolor='red', 
            facecolor='none', linestyle='--'
        )
        ax.add_patch(rect_patch)
        cx, cy = get_center(rect)
        ax.text(cx, cy, f"2-{i}", fontsize=10, 
                ha='center', va='center', color='red')

    # 设置坐标轴等比例+显示网格
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.title('Rectangles Comparison (Blue: rect1, Red: rect2)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 自动调整坐标范围
    all_rects = np.vstack([rect1, rect2])
    x_margin = max(all_rects[:, 2]) * 0.2
    y_margin = max(all_rects[:, 3]) * 0.2
    ax.set_xlim([np.min(all_rects[:, 0]) - x_margin, 
                np.max(all_rects[:, 0] + all_rects[:, 2]) + x_margin])
    ax.set_ylim([np.min(all_rects[:, 1]) - y_margin, 
                np.max(all_rects[:, 1] + all_rects[:, 3]) + y_margin])

    plt.tight_layout()
    plt.show()
    # for i, (x1, y1, x2, y2) in enumerate(detected_windows):
    #     cv2.rectangle(facade_img, ((x1, y1), (x2, y2)), (0, 255, 0), 2)  # Original (green)
    
    # for i, (x1, y1, x2, y2) in enumerate(aligned_windows):
    #     cv2.rectangle(facade_img, ((x1, y1), (x2, y2)), (0, 0, 255), 2)  # Aligned (red)
    #     cv2.putText(facade_img, f"C:{cluster_labels[i]}", (x1, y1-5), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # cv2.imshow("Original (Green) vs Aligned (Red)", facade_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()