import os
import json
from sklearn.metrics import accuracy_score , recall_score

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from model import resnet34 as create_model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
# from scipy import interp
from itertools import cycle
from sklearn import metrics
import matplotlib as mpl


plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

def eva(model, dataloaders, device):
    classes = ['0', '1', '2', '3', "4",'5', '6', '7', '8', "9"]
    preds_list = [[], [], [], [], [],[], [], [], [], []]
    labels_list = [[], [], [], [], [],[], [], [], [], []]
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            index = 0
            for la in labels:
                labels_list[la].append(int(la.detach().cpu().numpy()))
                preds_list[la].append(int(outputs[index].argmax(dim=0).detach().cpu().numpy()))
                index += 1
    accs = []
    for j in range(len(classes)):
        acc = accuracy_score(labels_list[j], preds_list[j])


        accs.append(acc)
    print(accs)

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        # print("the model accuracy is ", round(acc,3))

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall","Sensitivity" ,"Specificity","F1 socre"]

        precision_total = []
        recall_total = []
        sensitivity_total = []
        specificity_total = []
        f1_score_total = []


        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.###准确率
            precision_total.append(Precision)

            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.###召回率
            recall_total.append(Recall)

            Sensitivity = round(TP / (TP + FN), 3) if TN + FP != 0 else 0.###敏感性
            sensitivity_total.append(Sensitivity)

            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.  ###特异性
            specificity_total.append(Specificity)

            f1_socre = round((2 * Recall * Precision) / (Recall + Precision),3)if Recall + Precision != 0 else 0.
            f1_score_total.append(f1_socre)

            table.add_row([self.labels[i], Precision, Recall,Sensitivity, Specificity,f1_socre])
        print(table)

        Precision_mean = round(np.array(precision_total).mean(),3)
        Recall_mean = round(np.array(recall_total).mean(),3)
        Sensitivity_mean = round(np.array(sensitivity_total).mean(),3)
        Specificity_mean = round(np.array(specificity_total).mean(),3)
        f1_socre_mean = round(np.array(f1_score_total).mean(),3)

        print('准确率：',Precision_mean)
        print('召回率：', Recall_mean)
        print('敏感性：', Sensitivity_mean)
        print('特异性：', Specificity_mean)
        print('F1：', f1_socre_mean)
        print('Accuracy',round(acc,3))



    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig('./CONFUSION.jpg')
        plt.show()



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    json_label_path = './tomato.json'  ########读取标签类别

    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    num_classes = len(class_indict)  ########设置类别数

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path

    image_path = './dataset_caomei/test' ######设置测试集路径

    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=image_path,
                                            transform=data_transform)

    batch_size = 32
    validate_loader = torch.utils.data.DataLoader(validate_dataset, ###############载入测试集数据
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    net = create_model(num_classes=num_classes)##########################创建模型

    # load pretrain weights

    model_weight_path = "best.pth"############载入权重地址

    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict


    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)#############生成混淆矩阵
    net.eval()

    val_scores_total = np.empty(shape=[0, num_classes])
    val_lables_total = []

    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)

            val_scores_total = np.concatenate((val_scores_total, outputs.cpu().numpy()))
            val_lables_total.extend(val_labels.numpy().tolist())
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
    # plt.savefig('./CONFUSION.jpg')

    #########################################################################
######ROC曲线
#########################################################################
    ee = np.array(list(range(num_classes)))
    val_one_hot = label_binarize(val_lables_total, classes=ee) ##onehot编码

    if num_classes==2:
        bio_onehot = np.empty(shape=[0, 2])
        for i,value in enumerate(val_lables_total):
            if value == 0:
                bio_onehot = np.concatenate((bio_onehot, [[1,0]]),0)
            if value == 1:
                bio_onehot = np.concatenate((bio_onehot, [[0, 1]]),0)
        val_one_hot = bio_onehot

    fpr, tpr, thresholds = metrics.roc_curve(val_one_hot.ravel(),val_scores_total.ravel())
    auc = metrics.auc(fpr, tpr)
    print ('AUC：', round(auc,3))
    #绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    #FPR就是横坐标,TPR就是纵坐标
    # np.save('./efficientnet_fpr',fpr)
    # np.save('./efficientnet_tpr',tpr)

    plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'ROC和AUC', fontsize=17)
    plt.show()

