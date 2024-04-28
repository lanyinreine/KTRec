import csv
import numpy as np

adj = 'D:\\data\\Assist\\assist09\\matrix\\adj_m.csv'
de = 'D:\\data\\Assist\\assist09\\matrix\\de.csv'

# 读取输入的CSV文件
with open(adj, 'r') as file:
    reader = csv.reader(file)
    matrix = list(reader)

n = len(matrix)  # 矩阵的大小

# 计算每行和每列的总和并更新矩阵
for i in range(1, n):
    row_sum = sum(int(value) for value in matrix[i][1:])
    col_sum = sum(int(matrix[j][i]) for j in range(1, n))
    matrix[i][i] = str(row_sum + col_sum - int(matrix[i][i]))
    for j in range(1, n):
        if j != i:
            matrix[i][j] = '0'
            matrix[j][i] = '0'

# 将更新后的矩阵保存到新的CSV文件
with open(de, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(matrix)