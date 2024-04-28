import csv


def constructFeMatrix():
    input_file = 'D:\\data\\Assist\\assist09\\Assist2009_unique.csv'
    output_file = 'D:\\data\\Assist\\assist09\\matrix\\fe.csv'

    # 读取输入的CSV文件，提取problem_id和skill_id
    problem_ids = set()
    skill_ids = set()

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 跳过表头
        for row in reader:
            problem_ids.add(row[3])
            skill_ids.add(int(row[5]))

    problem_ids = sorted(problem_ids)
    skill_ids = sorted(skill_ids)

    M = len(problem_ids)
    N = len(skill_ids)

    # 创建初始的全零矩阵
    matrix = [[0] * (N + 1) for _ in range(M + 1)]
    matrix[0] = [''] + skill_ids  # 第一行是skill_id
    for i in range(1, M + 1):
        matrix[i][0] = problem_ids[i - 1]  # 第一列是problem_id

    # 遍历输入的CSV文件，更新矩阵
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            problem_id = row[3]
            skill_id = int(row[5])
            problem_index = problem_ids.index(problem_id) + 1
            skill_index = skill_ids.index(skill_id) + 1
            matrix[problem_index][skill_index] = 1

    # 将更新后的矩阵保存到新的CSV文件
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)


def updateMatrix():
    input_file = 'D:\\data\\Assist\\assist09\\matrix\\fe.csv'
    output_file = 'D:\\data\\Assist\\assist09\\matrix\\newfe.csv'

    # 读取输入的CSV文件
    data = []

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    # 统计完全相同的数据行
    unique_data = {}
    for row in data[1:]:
        key = tuple(row[1:])  # 从第1列开始作为键
        if key not in unique_data:
            unique_data[key] = [row[0]]
        if row[0] not in unique_data[key]:
            unique_data[key].append(row[0])  # 存储对应的problem_id

    # 将统计结果转换为新的矩阵
    matrix = [data[0]]
    for key, problem_ids in unique_data.items():
        matrix.append([problem_ids] + list(key))

    # 将新的矩阵保存到新的CSV文件
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)
constructFeMatrix()
updateMatrix()