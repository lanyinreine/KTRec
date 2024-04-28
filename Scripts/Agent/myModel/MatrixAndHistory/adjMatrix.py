import csv

# 读取CSV文件并提取数据
import numpy as np


def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


# 写入CSV文件
def write_csv(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


# 获取所有不同的skill_id
def get_unique_skills(data):
    skill_ids = set()
    for row in data[1:]:
        skill_ids.add(int(row[5]))
    return sorted(list(skill_ids))


# 创建邻接矩阵
def create_adjacency_matrix(data, unique_skills):
    n = len(unique_skills)
    adjacency_matrix = [[0] * n for _ in range(n)]

    user_data = {}
    for row in data[1:]:
        user_id = row[2]
        problem_id = row[3]
        skill_id = int(row[5])

        if user_id not in user_data:
            user_data[user_id] = []
        if (problem_id, skill_id) not in user_data[user_id]:
            user_data[user_id].append((problem_id, skill_id))

    for user_id, skills in user_data.items():
        sorted_skills = sorted(skills, key=lambda x: unique_skills.index(x[1]))
        # print(sorted_skills)
        for i in range(len(sorted_skills) - 1):
            skill1 = sorted_skills[i][1]
            skill2 = sorted_skills[i + 1][1]
            index1 = unique_skills.index(skill1)
            index2 = unique_skills.index(skill2)
            adjacency_matrix[index1][index2] = 1
            adjacency_matrix[index2][index1] = 1
            for j in range(i, len(sorted_skills) - 1):
                proid1 = sorted_skills[i][0]
                proid2 = sorted_skills[j][0]
                skill3 = sorted_skills[j][1]
                pindex1 = unique_skills.index(skill1)
                pindex2 = unique_skills.index(skill3)
                if proid1 == proid2:
                    adjacency_matrix[pindex1][pindex2] = 1
                    adjacency_matrix[pindex2][pindex1] = 1

    return adjacency_matrix


# 主函数
def main():
    input_file = 'D:\\data\\Assist\\assist09\\Assist2009_unique.csv'
    output_file = 'D:\\data\\Assist\\assist09\\matrix\\adj.csv'

    data = read_csv(input_file)
    unique_skills = get_unique_skills(data)
    adjacency_matrix = create_adjacency_matrix(data, unique_skills)

    # 添加表头
    adjacency_matrix.insert(0, [''] + unique_skills)
    for i in range(1, len(adjacency_matrix)):
        adjacency_matrix[i] = [unique_skills[i - 1]] + adjacency_matrix[i]

    write_csv(output_file, adjacency_matrix)
    print("邻接矩阵已生成并保存到output.csv文件中。")


# if __name__ == '__main__':
#     main()

# 判断是否是GCN所需的邻接矩阵
def is_adjacency_matrix(matrix):
    # Check if the matrix is symmetric
    if not np.array_equal(matrix, matrix.T):
        return False

    # Check if the diagonal elements are all 0s
    if np.any(np.diag(matrix) != 0):
        return False

    return True


def modify_csv_adjacency(csv_file, output_file):
    # Read the CSV file
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        rows = [row for row in csv_reader]

    # Convert the CSV data into a numpy array, excluding the first row and column
    matrix = np.array([row[1:] for row in rows[1:]], dtype=int)

    # Check if the matrix is an adjacency matrix
    if not is_adjacency_matrix(matrix):
        # Modify the matrix to meet the requirements
        matrix = np.maximum(matrix, matrix.T)  # Make the matrix symmetric
        np.fill_diagonal(matrix, 0)  # Set diagonal elements to 0

        # Write the modified matrix to a new CSV file
        with open(output_file, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(rows[0])  # Write the first row
            csv_writer.writerows([[rows[i+1][0]] + row.tolist() for i, row in enumerate(matrix)])


        return True, "The matrix has been modified and saved to the output file."

    return False, "The matrix is already a valid adjacency matrix for GCN."


# Example usage
csv_file = "D:\\data\\Assist\\assist09\\matrix\\adj.csv"
modified, message = modify_csv_adjacency(csv_file, "D:\\data\\Assist\\assist09\\matrix\\adj_m.csv")
print(message)
