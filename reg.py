import numpy as np
from sklearn.cluster import MeanShift
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

# Input bounding boxes
IN_PATH = "in.xywh"
# Regularized bounding boxes
OUT_PATH = "out.xywh"
# The lower bound
delta_x = 5
delta_y = 5
delta_w = 3
delta_h = 3
# The weights
alpha_x = 100
alpha_y = 100
alpha_h = 1
alpha_w = 1

def read_file(path):
    with open(path, 'r') as file:
        content = file.read()
    return content

def get_xywh(s):
    x, y, w, h = [], [], [], []
    for line in s.strip().split('\n'):
        parts = line.split()
        x.append(float(parts[0]))
        y.append(float(parts[1]))
        w.append(float(parts[2]))
        h.append(float(parts[3]))
    return x, y, w, h

def pre_cluster(x, delta):
    points = [[xi, 0] for xi in x]
    ms = MeanShift(bandwidth=delta)
    ms.fit(points)
    clusters = ms.cluster_centers_
    return [cluster[0] for cluster in clusters]

def regularize(x, y, w, h, X, Y, W, H):
    N = len(x)
    m = len(X)
    n = len(Y)
    m_ = len(W)
    n_ = len(H)
    
    idx = N * (m + n + m_ + n_)
    count = N * (m + n + m_ + n_) + (m + n + m_ + n_)
    
    # Objective function coefficients
    c = np.empty(count)
    idx_obj = 0
    for i in range(N):
        for k in range(m):
            t = (x[i] - X[k]) ** 2
            c[idx_obj] = t
            idx_obj += 1
    for i in range(N):
        for k in range(n):
            t = (y[i] - Y[k]) ** 2
            c[idx_obj] = t
            idx_obj += 1
    for i in range(N):
        for k in range(m_):
            t = (w[i] - W[k]) ** 2
            c[idx_obj] = t
            idx_obj += 1
    for i in range(N):
        for k in range(n_):
            t = (h[i] - H[k]) ** 2
            c[idx_obj] = t
            idx_obj += 1
    for _ in range(m):
        c[idx_obj] = alpha_x
        idx_obj += 1
    for _ in range(n):
        c[idx_obj] = alpha_y
        idx_obj += 1
    for _ in range(m_):
        c[idx_obj] = alpha_w
        idx_obj += 1
    for _ in range(n_):
        c[idx_obj] = alpha_h
        idx_obj += 1
    
    # Constraints
    A_eq = []
    b_eq = []
    for i in range(N):
        a = np.zeros(count)
        for k in range(m):
            a[i * m + k] = 1.0
        A_eq.append(a)
        b_eq.append(1.0)
    for i in range(N):
        a = np.zeros(count)
        for k in range(n):
            a[N * m + i * n + k] = 1.0
        A_eq.append(a)
        b_eq.append(1.0)
    for i in range(N):
        a = np.zeros(count)
        for k in range(m_):
            a[N * (m + n) + i * m_ + k] = 1.0
        A_eq.append(a)
        b_eq.append(1.0)
    for i in range(N):
        a = np.zeros(count)
        for k in range(n_):
            a[N * (m + n) + N * m_ + i * n_ + k] = 1.0
        A_eq.append(a)
        b_eq.append(1.0)
    
    A_ineq = []
    b_ineq = []
    for i in range(m):
        a = np.zeros(count)
        a[idx + i] = -1.0
        for k in range(N):
            a[k * m + i] = 1.0
        A_ineq.append(a)
        b_ineq.append(0.0)
    for i in range(n):
        a = np.zeros(count)
        a[idx + m + i] = -1.0
        for k in range(N):
            a[N * m + k * n + i] = 1.0
        A_ineq.append(a)
        b_ineq.append(0.0)
    for i in range(m_):
        a = np.zeros(count)
        a[idx + m + n + i] = -1.0
        for k in range(N):
            a[N * m + N * n + k * m_ + i] = 1.0
        A_ineq.append(a)
        b_ineq.append(0.0)
    for i in range(n_):
        a = np.zeros(count)
        a[idx + m + n + m_ + i] = -1.0
        for k in range(N):
            a[N * m + N * n + N * m_ + k * n_ + i] = 1.0
        A_ineq.append(a)
        b_ineq.append(0.0)
    
    for i in range(m):
        for k in range(N):
            a = np.zeros(count)
            a[k * m + i] = 1.0
            a[idx + i] = -1.0
            A_ineq.append(a)
            b_ineq.append(0.0)
    for i in range(n):
        for k in range(N):
            a = np.zeros(count)
            a[N * m + k * n + i] = 1.0
            a[idx + m + i] = -1.0
            A_ineq.append(a)
            b_ineq.append(0.0)
    for i in range(m_):
        for k in range(N):
            a = np.zeros(count)
            a[N * m + N * n + k * m_ + i] = 1.0
            a[idx + m + n + i] = -1.0
            A_ineq.append(a)
            b_ineq.append(0.0)
    for i in range(n_):
        for k in range(N):
            a = np.zeros(count)
            a[N * m + N * n + N * m_ + k * n_ + i] = 1.0
            a[idx + m + n + m_ + i] = -1.0
            A_ineq.append(a)
            b_ineq.append(0.0)
    
    # Bounds
    lb = np.zeros(count)
    ub = np.ones(count)
    
    # Integer constraints
    integrality = np.ones(count, dtype=int)
    
    # Solve MILP
    result = milp(c=c, 
                  constraints=LinearConstraint(np.vstack([A_eq, A_ineq]), 
                                               np.concatenate([b_eq, b_ineq]), 
                                               np.concatenate([b_eq, b_ineq])),
                  bounds=Bounds(lb, ub),
                  integrality=integrality)
    
    if result.success:
        return result.x
    else:
        print("Failed to find a solution:", result.message)
        return None

def write_file(X, Y, W, H, r, path):
    if r is None:
        print("No solution found, cannot write to file.")
        return
    
    r_x, r_y, r_w, r_h = [], [], [], []
    N = len(X)
    m = len(X)
    n = len(Y)
    m_ = len(W)
    n_ = len(H)
    
    for i in range(len(r) // m):
        for k in range(m):
            if r[i * m + k] != 0:
                r_x.append(k)
        for k in range(n):
            if r[N * m + i * n + k] != 0:
                r_y.append(k)
        for k in range(m_):
            if r[N * m + N * n + i * m_ + k] != 0:
                r_w.append(k)
        for k in range(n_):
            if r[N * m + N * n + N * m_ + i * n_ + k] != 0:
                r_h.append(k)
    
    result = ""
    for i in range(len(r_x)):
        x = X[r_x[i]]
        y = Y[r_y[i]]
        w = W[r_w[i]]
        h = H[r_h[i]]
        result += f"{x} {y} {w} {h}\n"
    
    with open(path, 'w') as file:
        file.write(result)

def main():
    content = read_file(IN_PATH)
    x, y, w, h = get_xywh(content)
    
    X = pre_cluster(x, delta_x)
    Y = pre_cluster(y, delta_y)
    W = pre_cluster(w, delta_w)
    H = pre_cluster(h, delta_h)
    
    sol = regularize(x, y, w, h, X, Y, W, H)
    if sol is not None:
        write_file(X, Y, W, H, sol, OUT_PATH)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()