import pandas as pd
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LinearRegression

import seaborn as sns
from matplotlib import pyplot as plt, cm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

def main():
    df_2a = pd.read_excel("data/2a.xlsx", header=None)
    df_2a = df_2a.drop(columns=0)

    df_1= pd.read_excel("data/表1-患者列表及临床信息.xlsx", engine='openpyxl')
    df_1 = df_1.iloc[:100]
    selected_columns_1 = df_1.columns[4:14].tolist()
    selected_columns_1.append('血压')
    features = df_1[selected_columns_1].copy()
    features['性别'] = features['性别'].replace({'男': 0, '女': 1})
    features[['高压', '低压']] = features['血压'].str.extract('(\d+)/(\d+)')
    features['高压'] = features['高压'].astype(float)
    features['低压'] = features['低压'].astype(float)
    features.drop('血压', axis=1, inplace=True)

    time_list = []
    V_list = []
    df_patients_info = df_1["发病到首次影像检查时间间隔"].copy()
    times = []
    for i in range(1, 101):
        row_2a = df_2a.iloc[i]

        additional_hours = df_patients_info[i-1]

        patient_first_check_time = df_2a.iloc[i,1]
        time_list.append(additional_hours)
        V_list.append(df_2a.iloc[i,2])
        count = 1

        for j in range(4, len(row_2a)+1, 3):
            # 如果当前单元格不为空
            if pd.notna(row_2a[j]):
                count = count+1
                date_value = row_2a[j]
                date_obj = datetime.strptime(str(date_value), "%Y-%m-%d %H:%M:%S")
                time_diff = (date_obj - patient_first_check_time).total_seconds()/ 3600.0
                time_diff += additional_hours

                V_value = row_2a[j + 2]
                if (V_value == 0.0):
                    V_value = V_list[-1]

                time_list.append(time_diff)
                V_list.append(V_value)
            else:
                break

        times.append(count)

    result_df = pd.DataFrame({"time_list": time_list, "V_list": V_list})

    result_df.to_excel("ans/newt_v.xlsx", index=False)
    train(result_df,times,features)


import itertools
def check(list1):
    diff = pd.read_csv('clustered_data_with_distances.csv') #452
    list2 = diff['Distance_to_Centroid'].tolist()


    total_wallets = sum(list1)
    total_values = len(list2)

    if total_wallets == total_values:
        print("The total number of wallets in list1 matches the number of values in list2.")
    else:
        print("The numbers do not match. There's a discrepancy.")

    def get_sums(list1, list2):
        iterator = iter(list2)
        return [sum(itertools.islice(iterator, num)) for num in list1]

    sums = get_sums(list1, list2)

    print(sums)

    df = pd.DataFrame(sums, columns=['Total Money'])
    df.to_csv('ans/sums.csv', index=False)

from sklearn.ensemble import RandomForestClassifier
def ranfom_forest_importance(data,data_clusters):    # 2. 使用随机森林对特征进行重要性排序
    rf = RandomForestClassifier()
    rf.fit(data, data_clusters)
    feature_importances = rf.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=feature_importances[sorted_idx], y=np.array(range(12))[sorted_idx], palette="viridis")
    plt.xlabel('Importance')
    plt.ylabel('Feature Index')
    plt.title('Feature Importances')
    plt.show()

def train(result_df,times,data):
    df = result_df.fillna(result_df.mean())
    X = df["time_list"].values#.reshape(-1, 1)
    y = df['V_list'].values#.reshape(-1, 1)

    kmeans = KMeans(n_clusters=3)
    data_clusters = kmeans.fit_predict(data)
    tnse(data,data_clusters)

    cluster_labels = list(data_clusters)

    cluster_linear(cluster_labels,times,df)


'''线性拟合'''
def cluster_linear(cluster_labels,times,dataf):
    labels_list = []
    people_id_list = []

    start_idx = 0
    people_data = []
    for idx, t in enumerate(times):
        end_idx = start_idx + t
        labels_list.extend([cluster_labels[idx]] * t)
        people_id_list.extend([idx] * t)
        people_data.append(dataf.iloc[start_idx:end_idx])
        start_idx = end_idx

    dataf['label'] = labels_list
    dataf['people_id'] = people_id_list

    clusters_data = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters_data:
            clusters_data[label] = []
        clusters_data[label].append(people_data[idx])

    models = {}
    for label, data_list in clusters_data.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        X = combined_data[['time_list']]
        y = combined_data['V_list']

        model = LinearRegression().fit(X, y)
        models[label] = model

    residuals_list = []

    for index, row in dataf.iterrows():
        label = row['label']
        time_list_value = pd.DataFrame([row['time_list']], columns=['time_list'])  # 使用DataFrame进行预测
        predicted_V_list = models[label].predict(time_list_value)
        residual = abs(row['V_list'] - predicted_V_list[0])
        residuals_list.append(residual)

    dataf['residual'] = residuals_list

    result_df = calculate_average_residual(dataf, times, cluster_labels)

    plot_all_wallets_and_fits(dataf, models)

def plot_all_wallets_and_fits(dataf, models):
    plt.figure(figsize=(15, 10))
    labels = dataf['label'].unique()
    colors = sns.color_palette("hsv", len(labels))

    for idx, label in enumerate(labels):
        current_data = dataf[dataf['label'] == label]
        sns.scatterplot(data=current_data, x='time_list', y='V_list', color=colors[idx], label=f'Label: {label}')

        model = models[label]

        time_list_values = np.linspace(current_data['time_list'].min(), current_data['time_list'].max(), 100).reshape(
            -1, 1)
        predicted_V_list = model.predict(time_list_values)

        plt.plot(time_list_values, predicted_V_list, color=colors[idx])

    plt.legend()

    plt.savefig('ans/combined_plot.png')
    plt.show()




def calculate_average_residual(dataf, times, cluster_labels):
    def calculate_average_residual(dataf, times, cluster_labels):
        average_residuals = []
        labels = []

        start_idx = 0
        for idx, t in enumerate(times):
            end_idx = start_idx + t
            person_data = dataf.iloc[start_idx:end_idx]

            avg_residual = person_data['residual'].mean()
            average_residuals.append(avg_residual)

            label = cluster_labels[idx]
            labels.append(label)

            start_idx = end_idx

        result_df = pd.DataFrame({
            'average_residual': average_residuals,
            'label': labels
        })

        result_df.to_csv('ans/average_residuals.csv', index=False)

        return result_df

def tnse(data,data_clusters):
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data)

    colors = cm.rainbow(np.linspace(0, 1, len(set(data_clusters))))

    plt.figure(figsize=(10, 7))
    for cluster, color in zip(set(data_clusters), colors):
        plt.scatter(data_tsne[data_clusters == cluster, 0],
                    data_tsne[data_clusters == cluster, 1],
                    c=[color],
                    label=f'Cluster {cluster}',
                    alpha=0.7)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Clusters')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()