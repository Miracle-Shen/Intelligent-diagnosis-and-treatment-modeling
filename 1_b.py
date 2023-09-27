import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from scipy.stats import mode
import numpy as np

def train_logistic_regression(X, y):
    """Train a logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)

    # 绘制损失
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Train Loss', 'Test Loss'], y=[train_loss, test_loss])
    plt.title(f'logistic_regression - Train vs Test Loss')
    plt.ylabel('Loss')
    plt.show()
    return model, X_test, y_test

def train_random_forest(X, y):
    """Train a random forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)

    # 绘制损失
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Train Loss', 'Test Loss'], y=[train_loss, test_loss])
    plt.title(f'random_forest - Train vs Test Loss')
    plt.ylabel('Loss')
    plt.show()
    return model, X_test, y_test

def train_lightgbm(X, y):
    """Train a LightGBM classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)

    # 绘制损失
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Train Loss', 'Test Loss'], y=[train_loss, test_loss])
    plt.title(f'LightGBM - Train vs Test Loss')
    plt.ylabel('Loss')
    plt.show()
    return model, X_test, y_test

def base_regression(df_1, df_2, df_3):
    """Preprocess and merge dataframes."""
    df_1.rename(columns={"入院首次影像检查流水号": "PatientID"}, inplace=True)
    df_1 = df_1.iloc[:100]
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

    label = pd.read_excel('ans/表4C字段（是否发生血肿扩张）.xlsx')
    label = label['value']

    from sklearn.preprocessing import StandardScaler

    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()
    scaler_3 = StandardScaler()

    new_df_1_scaled = pd.DataFrame(scaler_1.fit_transform(new_df_1), columns=new_df_1.columns)

    new_df_2_scaled = pd.DataFrame(scaler_2.fit_transform(new_df_2), columns=new_df_2.columns)

    new_df_3_scaled = pd.DataFrame(scaler_3.fit_transform(new_df_3), columns=new_df_3.columns)

    return new_df_1_scaled , new_df_2_scaled, new_df_3_scaled, label

def test(df_1, df_2, df_3):
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

    from sklearn.preprocessing import StandardScaler

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


import seaborn as sns
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

def plot_loss(model, X_train, y_train, X_test, y_test, model_name):
    # 计算训练和测试的损失
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)

    # 绘制损失
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Train Loss', 'Test Loss'], y=[train_loss, test_loss])
    plt.title(f'{model_name} - Train vs Test Loss')
    plt.ylabel('Loss')
    plt.show()

def main():
    df_1 = pd.read_excel('data/表1-患者列表及临床信息.xlsx', engine='openpyxl')
    df_2 = pd.read_excel('data/表2-患者影像信息血肿及水肿的体积及位置.xlsx', engine='openpyxl')
    df_3 = pd.read_excel('data/表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx', engine='openpyxl')
    new_df_1, new_df_2, new_df_3, label = base_regression(df_1, df_2, df_3)

    lr_model, X_test_lr, y_test_lr = train_logistic_regression(new_df_1, label)
    rf_model, X_test_rf, y_test_rf = train_random_forest(new_df_2, label)
    lgbm_model, X_test_lgbm, y_test_lgbm = train_lightgbm(new_df_3, label)

    test_1, test_2, test_3 = test(df_1, df_2, df_3)
    # 定义权重
    weights = {
        'Logistic Regression': 0.3,
        'Random Forest': 0.3,
        'LightGBM': 0.4
    }

    # 使用每个模型对其相应的数据集进行预测并四舍五入到四位小数
    probs_lr = np.round(lr_model.predict_proba(test_1)[:, 1], 4)
    probs_rf = np.round(rf_model.predict_proba(test_2)[:, 1], 4)
    probs_lgbm = np.round(lgbm_model.predict_proba(test_3)[:, 1], 4)

    # 使用权重对预测的概率进行加权平均
    final_probs = (probs_lr * weights['Logistic Regression'] +
                   probs_rf * weights['Random Forest'] +
                   probs_lgbm * weights['LightGBM'])


    final_probs = np.round(final_probs, 4)
    df_ans = pd.DataFrame(final_probs, columns=["probabilities"])
    df_ans.to_excel("ans/1_b预测扩张概率.xlsx", index=False)
    print(final_probs)

if __name__ == '__main__':
    main()
