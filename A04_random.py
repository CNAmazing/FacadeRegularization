import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_rectangle_grid(m, n, width_range, height_range, spacing_range=(10, 30)):  # 间距放大100倍
    """
    生成带有固定间距的矩形网格（尺寸放大100倍）
    每行之间的垂直间距固定（随机1-2种值）
    每列之间的水平间距固定（随机1-2种值）
    参数:
        m: 行数
        n: 列数
        width_range: 矩形宽度范围 (min, max) 单位：厘米/像素等
        height_range: 矩形高度范围 (min, max)
        spacing_range: 间距大小范围 (min, max)
    """
    # 尺寸放大100倍
    width_range = (width_range[0]*100, width_range[1]*100)
    height_range = (height_range[0]*100, height_range[1]*100)
    spacing_range = (spacing_range[0]*100, spacing_range[1]*100)
    
    # 生成基准列宽和行高
    base_col_width = random.uniform(*width_range)
    base_row_height = random.uniform(*height_range)
    
    # 为每列生成接近的宽度
    col_widths = [base_col_width * (1 + random.uniform(-0.2, 0.2)) for _ in range(n)]
    
    # 为每行生成接近的高度
    row_heights = [base_row_height * (1 + random.uniform(-0.2, 0.2)) for _ in range(m)]
    
    # 生成固定间距模式
    def generate_fixed_spacing(length, spacing_range):
        """生成1-2种固定间距值"""
        num_spacing_types = random.randint(1, 2)
        spacing_values = [random.uniform(*spacing_range) for _ in range(num_spacing_types)]
        spacings = [spacing_values[i % num_spacing_types] for i in range(length)]
        return spacings
    
    # 生成列间水平间距（n-1个）
    col_spacings = generate_fixed_spacing(n-1, spacing_range)
    
    # 生成行间垂直间距（m-1个）
    row_spacings = generate_fixed_spacing(m-1, spacing_range)
    
    # 计算每个矩形的坐标和尺寸
    rectangles = []
    y_pos = 0
    for i in range(m):
        x_pos = 0
        row = []
        for j in range(n):
            width = col_widths[j]
            height = row_heights[i]
            row.append((x_pos, y_pos, width, height))
            
            # 移动到下一列（添加列间距）
            if j < n-1:
                x_pos += width + col_spacings[j]
            else:
                x_pos += width
        
        rectangles.append(row)
        
        # 移动到下一行（添加行间距）
        if i < m-1:
            y_pos += row_heights[i] + row_spacings[i]
        else:
            y_pos += row_heights[i]
    
    # 计算总宽度和高度
    total_width = sum(col_widths) + sum(col_spacings)
    total_height = sum(row_heights) + sum(row_spacings)
    
    return rectangles, col_spacings, row_spacings, total_width, total_height

def plot_rectangles(rectangles, col_spacings, row_spacings, total_width, total_height):
    """绘制矩形网格（自动调整显示比例）"""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    
    # 绘制列间距标记
    for j, spacing in enumerate(col_spacings):
        x_pos = sum(rectangles[0][k][2] for k in range(j+1)) + sum(col_spacings[:j])
        ax.axvline(x=x_pos, color='red', linestyle=':', alpha=0.5)
        ax.text(x_pos, total_height*0.02, f"{spacing:.0f}", color='red', ha='center', fontsize=8)
    
    # 绘制行间距标记
    for i, spacing in enumerate(row_spacings):
        y_pos = sum(rectangles[k][0][3] for k in range(i+1)) + sum(row_spacings[:i])
        ax.axhline(y=y_pos, color='blue', linestyle=':', alpha=0.5)
        ax.text(total_width*0.98, y_pos, f"{spacing:.0f}", color='blue', va='center', fontsize=8)
    
    # 绘制矩形
    for row in rectangles:
        for x, y, width, height in row:
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=1, edgecolor='black', 
                facecolor=(random.random(), random.random(), random.random()),
                alpha=0.7
            )
            ax.add_patch(rect)
    
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.title(f'Rectangle Grid (Scale: 1:100)\nTotal Size: {total_width:.0f}x{total_height:.0f}', pad=20)
    plt.show()

# 示例使用
if __name__ == "__main__":
    # random.seed(42)  # 固定随机种子
    
    m = random.randint(3, 10)  # 行数
    n = random.randint(3, 10)  # 列数
    original_width_range = (0.8, 1.5)  # 原始宽度范围（放大前）
    original_height_range = (0.8, 1.5)  # 原始高度范围
    original_spacing_range = (0.1, 0.3)  # 原始间距范围
    
    # 生成网格（内部会自动放大100倍）
    rectangles, col_spacings, row_spacings, total_width, total_height = generate_rectangle_grid(
        m, n, original_width_range, original_height_range, original_spacing_range
    )
    
    # 绘制
    plot_rectangles(rectangles, col_spacings, row_spacings, total_width, total_height)
    
    # 打印信息（显示原始尺寸和放大后尺寸）
    print(f"【原始参数】")
    print(f"行数: {m}, 列数: {n}")
    print(f"原始宽度范围: {original_width_range} → 放大后: ({original_width_range[0]*100:.0f}, {original_width_range[1]*100:.0f})")
    print(f"原始高度范围: {original_height_range} → 放大后: ({original_height_range[0]*100:.0f}, {original_height_range[1]*100:.0f})")
    print(f"原始间距范围: {original_spacing_range} → 放大后: ({original_spacing_range[0]*100:.0f}, {original_spacing_range[1]*100:.0f})")
    
    print(f"\n【放大后结果】")
    print(f"网格总尺寸: {total_width:.0f} x {total_height:.0f}")
    print(f"列间距模式: {[f'{s:.0f}' for s in col_spacings]}")
    print(f"行间距模式: {[f'{s:.0f}' for s in row_spacings]}")
    print(f"矩形宽度范围: {min(w for row in rectangles for x,y,w,h in row):.0f}-{max(w for row in rectangles for x,y,w,h in row):.0f}")
    print(f"矩形高度范围: {min(h for row in rectangles for x,y,w,h in row):.0f}-{max(h for row in rectangles for x,y,w,h in row):.0f}")