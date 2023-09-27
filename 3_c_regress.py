'''regress'''
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn

nature_colors = ['#0072B2', '#009E73']

class MultiInputNetwork(nn.Module):
    def __init__(self):
        super(MultiInputNetwork, self).__init__()
        self.personal_info_subnetwork = nn.Sequential(
            nn.Linear(19, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.image_info1_subnetwork = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.image_info2_subnetwork = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32*3, 1)  # 修改这里：输出一个值

    def forward(self, x1, x2, x3):
        x1 = self.personal_info_subnetwork(x1.float())
        x2 = self.image_info1_subnetwork(x2.float())
        x3 = self.image_info2_subnetwork(x3.float())
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x


from torch.utils.data import Dataset, DataLoader
import torch
class CustomDataset(Dataset):
    def __init__(self, personal_info, image_info1, image_info2, labels):
        self.personal_info = personal_info
        self.image_info1 = image_info1
        self.image_info2 = image_info2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.personal_info.iloc[idx].values,
                self.image_info1.iloc[idx].values,
                self.image_info2.iloc[idx].values,
                self.labels.iloc[idx])

class TestDataset(Dataset):
    def __init__(self, personal_info, image_info1, image_info2):
        self.personal_info = personal_info
        self.image_info1 = image_info1
        self.image_info2 = image_info2

    def __len__(self):
        return len(self.personal_info)

    def __getitem__(self, idx):
        return (self.personal_info.iloc[idx].values,
                self.image_info1.iloc[idx].values,
                self.image_info2.iloc[idx].values)

def __getitem__(self, idx):
    return (torch.from_numpy(self.personal_info.iloc[idx].values).float(),
            torch.from_numpy(self.image_info1.iloc[idx].values).float(),
            torch.from_numpy(self.image_info2.iloc[idx].values).float(),
            torch.tensor(self.labels.iloc[idx], dtype=torch.long))

def base_regression(df_1, df_2, df_3):
    """Preprocess and merge dataframes."""
    df_1.rename(columns={"入院首次影像检查流水号": "PatientID"}, inplace=True)
    # df_temp = df_1.iloc[:160]
    # all_label = df_temp['90天mRS']
    df_1 = df_1.iloc[:100]
    label = df_1['90天mRS']

    df_2.rename(columns={"首次检查流水号": "PatientID"}, inplace=True)
    df_3.rename(columns={"流水号": "PatientID"}, inplace=True)
    patient_id_df = df_1['PatientID'].copy()

    selected_columns_1 = df_1.columns[4:22].tolist()
    selected_columns_1.append('PatientID')
    new_df_1 = df_1[selected_columns_1]
    # new_df_1['性别'] = new_df_1['性别'].replace({'男': 0, '女': 1})
    new_df_1 = new_df_1.copy()
    new_df_1['性别'] = new_df_1['性别'].replace({'男': 0, '女': 1})

    new_df_1[['高压', '低压']] = new_df_1['血压'].str.extract('(\d+)/(\d+)')
    new_df_1.drop('血压', axis=1, inplace=True)
    new_df_1 = new_df_1.drop(columns=['PatientID'])

    selected_columns_2 = df_2.columns[2:24].tolist()
    selected_columns_2.append('PatientID')
    new_df_2 = df_2[selected_columns_2]
    new_df_2 = pd.merge(patient_id_df, new_df_2, on='PatientID', how='inner')
    new_df_2 = new_df_2.drop(columns=['PatientID'])

    selected_columns_3 = df_3.columns[2:32].tolist()
    selected_columns_3.append('PatientID')
    new_df_3 = df_3[selected_columns_3]
    new_df_3 = pd.merge(patient_id_df, new_df_3, on='PatientID', how='inner')
    new_df_3 = new_df_3.drop(columns=['PatientID'])

    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()
    scaler_3 = StandardScaler()

    new_df_1_scaled = pd.DataFrame(scaler_1.fit_transform(new_df_1), columns=new_df_1.columns)
    new_df_2_scaled = pd.DataFrame(scaler_2.fit_transform(new_df_2), columns=new_df_2.columns)
    new_df_3_scaled = pd.DataFrame(scaler_3.fit_transform(new_df_3), columns=new_df_3.columns)

    return new_df_1_scaled, new_df_2_scaled, new_df_3_scaled, label

def test_data(df_1, df_2, df_3):
    df_1.rename(columns={"入院首次影像检查流水号": "PatientID"}, inplace=True)
    df_1 = df_1.iloc[:160]
    df_2.rename(columns={"首次检查流水号": "PatientID"}, inplace=True)
    df_3.rename(columns={"流水号": "PatientID"}, inplace=True)
    patient_id_df = df_1['PatientID'].copy()

    selected_columns_1 = df_1.columns[4:22].tolist()
    selected_columns_1.append('PatientID')
    new_df_1 = df_1[selected_columns_1]
    new_df_1 = new_df_1.copy()
    new_df_1['性别'] = new_df_1['性别'].replace({'男': 0, '女': 1})

    new_df_1[['高压', '低压']] = new_df_1['血压'].str.extract('(\d+)/(\d+)')
    new_df_1.drop('血压', axis=1, inplace=True)
    new_df_1 = new_df_1.drop(columns=['PatientID'])

    selected_columns_2 = df_2.columns[2:24].tolist()
    selected_columns_2.append('PatientID')
    new_df_2 = df_2[selected_columns_2]
    new_df_2 = pd.merge(patient_id_df, new_df_2, on='PatientID', how='inner')
    new_df_2 = new_df_2.drop(columns=['PatientID'])

    selected_columns_3 = df_3.columns[2:32].tolist()
    selected_columns_3.append('PatientID')
    new_df_3 = df_3[selected_columns_3]
    new_df_3 = pd.merge(patient_id_df, new_df_3, on='PatientID', how='inner')
    new_df_3 = new_df_3.drop(columns=['PatientID'])

    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()
    scaler_3 = StandardScaler()

    new_df_1_scaled = pd.DataFrame(scaler_1.fit_transform(new_df_1), columns=new_df_1.columns)
    new_df_2_scaled = pd.DataFrame(scaler_2.fit_transform(new_df_2), columns=new_df_2.columns)
    new_df_3_scaled = pd.DataFrame(scaler_3.fit_transform(new_df_3), columns=new_df_3.columns)

    for column in new_df_1_scaled.columns:
        new_df_1_scaled[column].fillna(new_df_1_scaled[column].mean(), inplace=True)
    for column in new_df_2_scaled.columns:
        new_df_2_scaled[column].fillna(new_df_2_scaled[column].mean(), inplace=True)
    for column in new_df_3_scaled.columns:
        new_df_3_scaled[column].fillna(new_df_3_scaled[column].mean(), inplace=True)
    return new_df_1_scaled, new_df_2_scaled, new_df_3_scaled

import torch
from torch.utils.data import DataLoader

model = MultiInputNetwork()
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from sklearn.model_selection import train_test_split
# # 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
def train(personal_info_data, image_info1_data, image_info2_data, labels):
    info_train, info_val, img1_train, img1_val, img2_train, img2_val, labels_train, labels_val = train_test_split(
        personal_info_data, image_info1_data, image_info2_data, labels, test_size=0.2, random_state=42
    )

    train_dataset = CustomDataset(info_train, img1_train, img2_train, labels_train)
    val_dataset = CustomDataset(info_val, img1_val, img2_val, labels_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        for personal_info_data, image_info1_data, image_info2_data, targets in train_loader:
            outputs = model(personal_info_data, image_info1_data, image_info2_data)
            targets = targets.float()  # 将标签转换为float类型
            loss = criterion(outputs.squeeze(), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for personal_info_data, image_info1_data, image_info2_data, targets in val_loader:
                outputs = model(personal_info_data, image_info1_data, image_info2_data)
                targets = targets.float()  # 将标签转换为float类型
                val_loss += criterion(outputs.squeeze(), targets).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / len(val_dataset)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        torch.save(model.state_dict(), "model.pth")


def plots(final_predictions,label):
    final_predictions_100 = final_predictions[:100]

    # 计算准确率
    accuracy = accuracy_score(label, final_predictions_100)
    print(f"准确度: {accuracy:.2f}")

    # 使用对比度较高的颜色
    colors = ['#0057E7', '#33FF57']  # 蓝色和绿色

    # 可视化实际标签和预测结果
    plt.figure(figsize=(10, 5))

    # 实际标签
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(label, return_counts=True)
    plt.bar(unique, counts, color=colors[0])
    plt.title('患者mRS实际评分')
    plt.xlabel('Class')
    plt.ylabel('Count')

    # 预测结果
    plt.subplot(1, 2, 2)
    unique, counts = np.unique(final_predictions_100, return_counts=True)
    plt.bar(unique, counts, color=colors[1])
    plt.title('患者mRS预测评分')
    plt.xlabel('mRS评分')
    plt.ylabel('患者人数')
    plt.show()


def main():
    df_1 = pd.read_excel('data/表1-患者列表及临床信息.xlsx', engine='openpyxl')
    df_2 = pd.read_excel('data/表2-患者影像信息血肿及水肿的体积及位置.xlsx', engine='openpyxl')
    df_3 = pd.read_excel('data/表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx', engine='openpyxl')
    personal_info_data, image_info1_data, image_info2_data, label = base_regression(df_1, df_2, df_3)

    train(personal_info_data, image_info1_data, image_info2_data, label )

    # 加载模型
    model = MultiInputNetwork()
    model.load_state_dict(torch.load("model.pth"))

    # 假设 test_data 函数返回测试数据
    test_personal_info_data, test_image_info1_data, test_image_info2_data = test_data(df_1, df_2, df_3)

    # 创建测试 Dataset 和 DataLoader
    test_dataset = TestDataset(test_personal_info_data, test_image_info1_data, test_image_info2_data)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 在测试集上执行预测
    model.eval()
    predictions = []
    with torch.no_grad():
        for personal_info_data, image_info1_data, image_info2_data in test_loader:
            outputs = model(personal_info_data, image_info1_data, image_info2_data)
            preds = torch.round(outputs).int()  # 四舍五入到最近的整数
            predictions.extend(preds.numpy())

    print(f"Predictions: {predictions}")
    plots(predictions,label)

if __name__ == '__main__':
    main()
