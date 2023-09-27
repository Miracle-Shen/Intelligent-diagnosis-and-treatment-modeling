import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from sklearn.model_selection import train_test_split
import xgboost as xgb

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

df_V = pd.read_excel("data/水肿体积变化.xlsx", engine='openpyxl')

df_plan = pd.read_excel('data/表1-患者列表及临床信息.xlsx', engine='openpyxl')
df_plan= df_plan.iloc[:100]
y = df_V ["体积变化"]
selected_columns_1 = df_plan .columns[16:23].tolist()
# selected_columns_1.append('PatientID')
X = df_plan[selected_columns_1]
column_names = [
    "Ventricular drainage",
    "Hemostatic treatment",
    "Intracranial pressure reduction treatment",
    "Hypotensive treatment",
    "Sedation and analgesic treatment",
    "Antiemetic and gastric protection",
    "Nutritional nerve"]#selected_columns_1

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#[脑室引流,止血治疗,降颅压治疗,降压治疗,镇静,镇痛治疗,止吐护胃,营养神经,]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=column_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=column_names)

param = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

epochs = 50

evals_result = {}
evals = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(param, dtrain, epochs, evals=evals, evals_result=evals_result, early_stopping_rounds=10,
                verbose_eval=10)

train_errors = evals_result['train']['rmse']
val_errors = evals_result['eval']['rmse']

plt.plot(train_errors, label='Train')
plt.plot(val_errors, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Logloss')
plt.title('Loss over epochs')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(bst, importance_type='weight', ax=ax)
plt.tight_layout()
plt.savefig('feature_importance_hd.png', dpi=300)
plt.show()