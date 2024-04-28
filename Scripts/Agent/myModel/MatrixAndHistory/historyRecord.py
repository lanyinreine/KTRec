# 按照user_id划分不同文件，以user_id命名文件用以区分
# 依照newfe.csv文件构建当前的做题记录的特征矩阵，第0行为表头，从第0列表示problem_id，从第1列起表示不同skill_id
# 从第1行开始，表示做的每一道题后的知识掌握程度，根据原文件中correct标记，统计是否答对当前skill_id,答对标1，否则0

import csv
import os

import numpy as np


def split_data_by_user(csv_file):
    # 创建一个字典，用于存储根据user_id划分的数据
    data_by_user = {}

    # 打开CSV文件
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)

        # 读取表头
        headers = next(reader)

        # 获取user_id的索引
        user_id_index = headers.index('user_id')

        # 遍历每一行数据
        for row in reader:
            user_id = row[user_id_index]

            # 检查user_id是否已经存在于字典中
            if user_id in data_by_user:
                data_by_user[user_id].append(row)
            else:
                data_by_user[user_id] = [row]

    # 创建目录用于存储划分后的文件
    output_dir = 'D:\\data\\Assist\\assist09\\history'  # 指定输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历字典中的数据，将每个user_id的数据写入相应的文件
    for user_id, data in data_by_user.items():
        output_file = os.path.join(output_dir, f'{user_id}.csv')

        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)

            # 写入表头
            writer.writerow(headers)

            # 写入数据
            writer.writerows(data)

    print("数据已成功划分并保存到不同的文件。")


#
# # 调用函数并传递CSV文件路径作为参数
# split_data_by_user('D:\\data\\Assist\\assist09\\Assist2009_unique.csv')


# 将学生学习记录，变成与特征矩阵相同格式的形式，即每行表示各skill_id是否掌握，根据作答情况置1或0
def generate_one_history(user_id_file, newfe_file, u_id):
    # 读取user_id文件
    with open(user_id_file, 'r') as file:
        reader = csv.reader(file)
        user_id_data = list(reader)
        # print(user_id_data)

    # 读取newfe.csv文件
    with open(newfe_file, 'r') as file:
        reader = csv.reader(file)
        newfe_data = list(reader)

    # 提取newfe表头和problem_id列头
    newfe_header = newfe_data[0]
    # print(user_id_data)
    # print(newfe_data)
    # 创建新文件的文件名
    useridfile = os.path.join('D:\\data\\Assist\\assist09\\history\\learninghistory', u_id)

    # 写入新文件
    with open(useridfile, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        header_row = newfe_header

        header_row[0] = 'problem_id'
        # print(header_row)
        writer.writerow(header_row)
        lastproblem = user_id_data[1][3]
        skills = [0 for _ in range(len(newfe_data[0]))]
        history = []
        flag = 0
        # 遍历user_id数据
        for row in user_id_data[1:]:
            # print(row)
            problem_id = row[3]
            skill_id = row[5]
            correct = row[4]
            if lastproblem != problem_id:
                skills[0] = lastproblem
                lastproblem = problem_id
                history.append(skills)
                # writer.writerow(skills)
                skills = [0 for _ in range(len(newfe_data[0]))]
            skills[header_row.index(skill_id)] = int(correct)

            flag += 1
            if (flag == len(user_id_data[1:])) and (lastproblem == problem_id):
                skills[0] = problem_id
                history.append(skills)
            # print('this is history')
            # for i in history:
            #     print(i)
        for row in history:
            writer.writerow(row)
        return history


# 将每个单独的user_id变成一个单独的历史学习记录数据
def output_history(file_path):
    # 获取文件夹下的所有文件
    files = os.listdir(file_path)
    # 遍历文件列表
    for file in files:
        # 检查文件是否为CSV文件
        if file.endswith(".csv"):
            # 构建文件的完整路径
            useridfile = os.path.join(file_path, file)
            # 调用函数并传递user_id文件和newfe.csv文件的路径作为参数
            generate_one_history(useridfile, 'D:\\data\\Assist\\assist09\\matrix\\newfe.csv', file)



# output_history('D:\\data\\Assist\\assist09\\history')
