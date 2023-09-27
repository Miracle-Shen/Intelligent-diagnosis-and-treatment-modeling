import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_excel('data/big_table.xlsx', engine='openpyxl')
train_data = data.iloc[:100]
test_data = data.iloc[130:161]

# 分离特征和标签
X_train = train_data.drop(columns=['ID', '90天mRS'])
y_train = train_data['90天mRS']
X_test = test_data.drop(columns=['ID', '90天mRS'])
y_test = test_data['90天mRS']

y_train = torch.tensor(y_train.values, dtype=torch.long)
# 分离非时间序列和时间序列特征
X_nts_train = X_train.iloc[:, 0:20]
X_ts_train = X_train.iloc[:, 20:]
X_nts_test = X_test.iloc[:, 0:20]
X_ts_test = X_test.iloc[:, 20:]

null_columns = X_ts_train.columns[ X_ts_train.isnull().any()]
if not null_columns.empty:
    mean_values = X_ts_train.mean()
    X_ts_train.fillna(mean_values, inplace=True)

null_columns_test = X_ts_test.columns[X_ts_test.isnull().any()]
if not null_columns_test.empty:
    mean_values_train = X_ts_train.mean()
    X_ts_test.fillna(mean_values_train, inplace=True)

X_nts_train_np = X_nts_train.values
X_ts_train_np = X_ts_train.values
X_nts_test_np = X_nts_test.values
X_ts_test_np = X_ts_test.values
scaler_nts = StandardScaler()
X_nts_train_scaled = scaler_nts.fit_transform(X_nts_train_np)
X_nts_test_scaled = scaler_nts.transform(X_nts_test_np)

scaler_ts = StandardScaler()
X_ts_train_scaled = scaler_ts.fit_transform(X_ts_train_np)
X_ts_test_scaled = scaler_ts.transform(X_ts_test_np)

# 将标准化后的NumPy数组转换为PyTorch tensors
X_ts_train = torch.tensor(X_ts_train_scaled, dtype=torch.float32).unsqueeze(1)
X_nts_train = torch.tensor(X_nts_train_scaled, dtype=torch.float32)

X_ts_test = torch.tensor(X_ts_test_scaled, dtype=torch.float32).unsqueeze(1)
X_nts_test = torch.tensor(X_nts_test_scaled, dtype=torch.float32)

# 3. 构建模型
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size_ts, input_size_nts, hidden_size, output_size):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size_ts, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(input_size_nts, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x_ts, x_nts):
        out_ts, _ = self.lstm(x_ts.unsqueeze(1))
        out_ts = out_ts[:, -1, :]
        out_nts = self.fc1(x_nts)
        out = torch.cat((out_ts, out_nts), dim=1)
        out = self.fc2(out)
        return out
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # 保留95%的方差
X_ts_train_np = pca.fit_transform(X_ts_train_np)
X_ts_test_np = pca.transform(X_ts_test_np)

class EnhancedTimeSeriesModel(nn.Module):
    def __init__(self, input_size_ts, input_size_nts, hidden_size, output_size=7):  # 设置输出大小为7
        super(EnhancedTimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size_ts, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(input_size_nts, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  # 设置输出大小为7

    def forward(self, x_ts, x_nts):
        out_ts, _ = self.lstm(x_ts)
        out_ts = out_ts[:, -1, :]
        out_nts = self.fc1(x_nts)
        out = torch.cat((out_ts, out_nts), dim=1)
        out = self.fc2(out)
        return self.fc3(out)


model = EnhancedTimeSeriesModel(input_size_ts=X_ts_train.shape[2], input_size_nts=X_nts_train.shape[1], hidden_size=128, output_size=1)
epochs = 200
optimizer = optim.Adam(model.parameters(), lr=0.0005)

weights = torch.tensor([1, 1, 1, 1, 1, 1, 3], dtype=torch.float32)  # 增强第六个类别的权重
criterion = nn.CrossEntropyLoss(weight=weights)


epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_ts_train, X_nts_train)
    y_train = y_train.float()
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 模型评估
model.eval()
with torch.no_grad():
    test_outputs = model(X_ts_test, X_nts_test)
    _, predicted = torch.max(test_outputs, 1)  # 获取预测的类别

# 结果可视化
plt.figure(figsize=(10, 6))
plt.plot(test_outputs, label='Predicted')
plt.legend()
plt.show()

# 将预测结果和实际结果转换为numpy数组，以便更容易地与pandas一起使用
predicted = test_outputs.numpy().squeeze()
actual = y_test.numpy().squeeze()

# 创建一个新的pandas DataFrame来保存预测和实际结果
results = pd.DataFrame({
    'Predicted': predicted,
    'Actual': actual
})

# 将结果保存到本地CSV文件
results.to_csv('predicted_results.csv', index=False)
