"""
This script is used to generate the dataset path to pytorch.
-----------------------
Training set: 1, 2, 3, 4, 5, 6, 9
Validation set: 7, 8
Test set: 10
"""


for case in range(1, 11):  # 1-10
    for i in range(36):
        if case == 10:
            filename = './data_path_MR/test_list_whole_MR.txt'     # case10中的0-35号数据作为测试集
        else:
            filename = './data_path_MR/train_list_whole_MR.txt'    # case1,2,3,4,5,6,9中的0-35号数据作为训练集
        if case == 7 or case == 8:
            filename = './data_path_MR/val_list_whole_MR.txt'      #case7,8中的0-35号数据作为验证集
        with open(filename, 'a') as f:
            f.write("../data_MR/case{0}/ct_{1}.png ../data_MR/case{0}/mr_{1}.png\n".format(case, str(i)))


result_path = './result_path_MR'
version = '/my_result'
for case in range(1, 11):
    for i in range(36):
        if case == 10:
            filename = result_path + version + '_test_list_whole_MR.txt'
        else:
            filename = result_path + version + '_train_list_whole_MR.txt'
        if case == 7 or case == 8:
            filename = result_path + version + '_val_list_whole_MR.txt'
        with open(filename, 'a') as f:
            f.write("./my_result/{0}_{1}.png ../data_MR/case{0}/ct_{1}.png ../data_MR/case{0}/mr_{1}.png\n".format(case, str(i)))


for case in range(1, 11):
    for i in range(0, 36):
        filename = './data_path_MR/final_all_data_MR.txt'
        with open(filename, 'a') as f:
            f.write("../data_MR/case{0}/ct_{1}.png ../data_MR/case{0}/mr_{1}.png\n".format(case, str(i)))