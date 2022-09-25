import os,shutil
import random
from tqdm import tqdm

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        # print( "move %s -> %s"%( srcfile,dstfile))

if __name__ == '__main__':

    split_rate = 0.8

    file_dir = 'C:\\Users\\25852\\Desktop\\test'
    for class_name in os.listdir(file_dir):
        print(class_name)

        for root, dirs, files in os.walk(os.path.join(file_dir, class_name)):
            sum_num = len(files)
            random.shuffle(files)

            for file_train in tqdm(files[0: int(sum_num * split_rate)]):
                root_new = os.path.join('./dataset_tomato', 'train')
                # print(file_train)
                srcfile = os.path.join(file_dir, class_name, file_train)
                # print(file_train)
                dstfile = os.path.join(root_new, class_name, file_train)

                mymovefile(srcfile, dstfile)

            for file_train in tqdm(files[int(sum_num * split_rate): ]):
                root_new = os.path.join('./dataset_tomato', 'test')
                # print(file_train)
                srcfile = os.path.join(file_dir, class_name, file_train)
                # print(file_train)
                dstfile = os.path.join(root_new, class_name, file_train)

                mymovefile(srcfile, dstfile)








        # print(location)



        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        # e =5
