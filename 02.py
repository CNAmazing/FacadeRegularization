import numpy as np
import mosek.fusion as mf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

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
# ----------------------------
# 1. 数据准备（与之前相同）
# ----------------------------
data = read_json(r'E:\WorkSpace\FacadeRegularization\data2.json')
# 矩形表示: (x1, y1, w, h)
rect = data['window']
rect = xyxy_to_xywh(rect)
rectangles= []
for r in rect:
    rectangles.append({
        'left': r[0],
        'top': r[1],
        'right': r[0] + r[2],
        'bottom': r[1] + r[3],
        'type': 'window'
    })
original_rectangles = [r.copy() for r in rectangles]

def plot_rectangles(rectangles, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for rect in rectangles:
        width = rect['right'] - rect['left']
        height = rect['bottom'] - rect['top']
        color = 'blue' if rect['type'] == 'window' else 'red'
        ax.add_patch(patches.Rectangle(
            (rect['left'], rect['top']), width, height,
            linewidth=1, edgecolor=color, facecolor='none'
        ))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_title(title)
    plt.gca().invert_yaxis()
    plt.show()

plot_rectangles(rectangles, "Original Rectangles")

# ----------------------------
# 2. MILP模型构建与求解（MOSEK版本）
# ----------------------------
def align_rectangles_with_mosek(rectangles, epsilon=20, m=6, n=8, m_prime=3, n_prime=3):
    """使用MOSEK实现MILP对齐"""
    with mf.Model('FacadeAlignment') as model:
        N = len(rectangles)
        M = 1000  # 大M常数
        
        # --- 变量定义 ---
        # 对齐类别变量
        t = model.variable('t', m, mf.Domain.unbounded())
        b = model.variable('b', m, mf.Domain.unbounded())
        l = model.variable('l', n, mf.Domain.unbounded())
        r = model.variable('r', n, mf.Domain.unbounded())
        
        # 尺寸类别变量
        h = model.variable('h', m_prime, mf.Domain.unbounded())
        w = model.variable('w', n_prime, mf.Domain.unbounded())
        
        # 坐标调整变量
        delta_left_plus = model.variable('dlp', N, mf.Domain.inRange(0, epsilon))
        delta_left_minus = model.variable('dlm', N, mf.Domain.inRange(0, epsilon))
        delta_right_plus = model.variable('drp', N, mf.Domain.inRange(0, epsilon))
        delta_right_minus = model.variable('drm', N, mf.Domain.inRange(0, epsilon))
        delta_top_plus = model.variable('dtp', N, mf.Domain.inRange(0, epsilon))
        delta_top_minus = model.variable('dtm', N, mf.Domain.inRange(0, epsilon))
        delta_bottom_plus = model.variable('dbp', N, mf.Domain.inRange(0, epsilon))
        delta_bottom_minus = model.variable('dbm', N, mf.Domain.inRange(0, epsilon))
        
        # 二元指示变量
        a_top = model.variable('a_top', [N, m], mf.Domain.binary())
        a_bottom = model.variable('a_bottom', [N, m], mf.Domain.binary())
        a_left = model.variable('a_left', [N, n], mf.Domain.binary())
        a_right = model.variable('a_right', [N, n], mf.Domain.binary())
        a_height = model.variable('a_height', [N, m_prime], mf.Domain.binary())
        a_width = model.variable('a_width', [N, n_prime], mf.Domain.binary())
        
        # --- 目标函数 ---
        F = 1 / (8 * N * epsilon * (max(n + n_prime, m + m_prime) + 1))
        alignment_terms = []
        adjustment_terms = []
        
        for i in range(N):
            # 对齐奖励
            for k in range(m):
                # 使用 expr_add 相加变量，然后 expr_mul 乘以标量
                sum_top_bottom = model.expr_add(a_top.index(i,k), a_bottom.index(i,k))
                scaled_term = model.expr_mul(sum_top_bottom, 1.0 / (k + 1))
                alignment_terms.append(scaled_term)
            
            for k in range(m_prime):
                # 直接乘以标量（不需要相加）
                scaled_term = model.expr_mul(a_height.index(i,k), 1.0 / (m + k + 1))
                alignment_terms.append(scaled_term)
            
            for k in range(n):
                # 使用 expr_add 相加变量，然后 expr_mul 乘以标量
                sum_left_right = model.expr_add(a_left.index(i,k), a_right.index(i,k))
                scaled_term = model.expr_mul(sum_left_right, 1.0 / (k + 1))
                alignment_terms.append(scaled_term)
            
            for k in range(n_prime):
                # 直接乘以标量（不需要相加）
                scaled_term = model.expr_mul(a_width.index(i,k), 1.0 / (n + k + 1))
                alignment_terms.append(scaled_term)
            
            # 调整惩罚
            # 使用 expr_add 相加变量
            term1 = model.expr_add(delta_left_plus.index(i), delta_left_minus.index(i))
            term2 = model.expr_add(delta_right_plus.index(i), delta_right_minus.index(i))
            term3 = model.expr_add(delta_top_plus.index(i), delta_top_minus.index(i))
            term4 = model.expr_add(delta_bottom_plus.index(i), delta_bottom_minus.index(i))
            adjustment_terms.extend([term1, term2, term3, term4])
        model.objective(
            mf.ObjectiveSense.Maximize,
            mf.Expr.sub(mf.Expr.add(alignment_terms), mf.Expr.mul(F, mf.Expr.add(adjustment_terms)))
        )
        
        # --- 约束条件 ---
        for i in range(N):
            rect = rectangles[i]
            
            # 每个属性最多属于一个对齐类别
            model.constraint(mf.Expr.sum(a_top.slice([i,0], [i+1,m])), mf.Domain.lessThan(1))
            model.constraint(mf.Expr.sum(a_bottom.slice([i,0], [i+1,m])), mf.Domain.lessThan(1))
            model.constraint(mf.Expr.sum(a_left.slice([i,0], [i+1,n])), mf.Domain.lessThan(1))
            model.constraint(mf.Expr.sum(a_right.slice([i,0], [i+1,n])), mf.Domain.lessThan(1))
            model.constraint(mf.Expr.sum(a_height.slice([i,0], [i+1,m_prime])), mf.Domain.lessThan(1))
            model.constraint(mf.Expr.sum(a_width.slice([i,0], [i+1,n_prime])), mf.Domain.lessThan(1))
            
            # 对齐约束
            for k in range(m):
                # 上边缘对齐到t_k
                model.constraint(
                    mf.Expr.sub(
                        rect['top'] + delta_top_plus.index(i) - delta_top_minus.index(i),
                        t.index(k)
                    ),
                    mf.Domain.inRange(-M*(1-a_top.index(i,k)), M*(1-a_top.index(i,k)))
                )
                # 下边缘对齐到b_k
                model.constraint(
                    mf.Expr.sub(
                        rect['bottom'] + delta_bottom_plus.index(i) - delta_bottom_minus.index(i),
                        b.index(k)
                    ),
                    mf.Domain.inRange(-M*(1-a_bottom.index(i,k)), M*(1-a_bottom.index(i,k)))
                )
            
            for k in range(n):
                # 左边缘对齐到l_k
                model.constraint(
                    mf.Expr.sub(
                        rect['left'] + delta_left_plus.index(i) - delta_left_minus.index(i),
                        l.index(k)
                    ),
                    mf.Domain.inRange(-M*(1-a_left.index(i,k)), M*(1-a_left.index(i,k)))
                )
                # 右边缘对齐到r_k
                model.constraint(
                    mf.Expr.sub(
                        rect['right'] + delta_right_plus.index(i) - delta_right_minus.index(i),
                        r.index(k)
                    ),
                    mf.Domain.inRange(-M*(1-a_right.index(i,k)), M*(1-a_right.index(i,k)))
                )
            
            for k in range(m_prime):
                # 高度对齐到h_k
                model.constraint(
                    mf.Expr.sub(
                        (rect['bottom'] + delta_bottom_plus.index(i) - delta_bottom_minus.index(i)) - 
                        (rect['top'] + delta_top_plus.index(i) - delta_top_minus.index(i)),
                        h.index(k)
                    ),
                    mf.Domain.inRange(-M*(1-a_height.index(i,k)), M*(1-a_height.index(i,k)))
                )
            
            for k in range(n_prime):
                # 宽度对齐到w_k
                model.constraint(
                    mf.Expr.sub(
                        (rect['right'] + delta_right_plus.index(i) - delta_right_minus.index(i)) - 
                        (rect['left'] + delta_left_plus.index(i) - delta_left_minus.index(i)),
                        w.index(k)
                    ),
                    mf.Domain.inRange(-M*(1-a_width.index(i,k)), M*(1-a_width.index(i,k)))
                )
        
        # 消除重叠约束
        for i in range(N):
            for j in range(i+1, N):
                if (rectangles[i]['top'] < rectangles[j]['bottom'] + 2*epsilon and 
                    rectangles[j]['top'] < rectangles[i]['bottom'] + 2*epsilon):
                    model.constraint(
                        rectangles[i]['top'] + delta_top_plus.index(i) - delta_top_minus.index(i),
                        mf.Domain.lessThan(
                            rectangles[j]['bottom'] + delta_bottom_plus.index(j) - delta_bottom_minus.index(j) - 5
                        )
                    )
        
        # --- 求解 ---
        model.solve()
        
        # --- 后处理 ---
        aligned_rectangles = []
        for i in range(N):
            new_rect = {
                'left': rectangles[i]['left'] + delta_left_plus.index(i).level()[0] - delta_left_minus.index(i).level()[0],
                'right': rectangles[i]['right'] + delta_right_plus.index(i).level()[0] - delta_right_minus.index(i).level()[0],
                'top': rectangles[i]['top'] + delta_top_plus.index(i).level()[0] - delta_top_minus.index(i).level()[0],
                'bottom': rectangles[i]['bottom'] + delta_bottom_plus.index(i).level()[0] - delta_bottom_minus.index(i).level()[0],
                'type': rectangles[i]['type']
            }
            aligned_rectangles.append(new_rect)
        
        return aligned_rectangles

# 执行对齐
aligned_rectangles = align_rectangles_with_mosek(rectangles)
plot_rectangles(aligned_rectangles, "Aligned Rectangles (MOSEK)")

# ----------------------------
# 3. 结果分析（与之前相同）
# ----------------------------
def calculate_iou(rect1, rect2):
    x_left = max(rect1['left'], rect2['left'])
    y_top = max(rect1['top'], rect2['top'])
    x_right = min(rect1['right'], rect2['right'])
    y_bottom = min(rect1['bottom'], rect2['bottom'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (rect1['right'] - rect1['left']) * (rect1['bottom'] - rect1['top'])
    area2 = (rect2['right'] - rect2['left']) * (rect2['bottom'] - rect2['top'])
    return intersection / (area1 + area2 - intersection)

total_iou = sum(calculate_iou(orig, aligned) for orig, aligned in zip(original_rectangles, aligned_rectangles))
avg_iou = total_iou / len(original_rectangles)
print(f"Average IoU with original detections: {avg_iou:.4f}")