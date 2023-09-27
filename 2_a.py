import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.linear_model import LinearRegression

df_2a = pd.read_excel("data/2a.xlsx", header=None)
df_1 = pd.read_excel("data/表1-患者列表及临床信息.xlsx", usecols=[14])

df_2a = df_2a.drop(columns=0)

time_list = []
V_list = []
df_patients_info = df_1["发病到首次影像检查时间间隔"].copy()

for i in range(1, 101):
    row_2a = df_2a.iloc[i]

    time_values = []
    V_values = []
    additional_hours = df_patients_info[i-1]
    time_values.append(additional_hours)
    V_values.append(df_2a.iloc[i, 2])
    for j in range(4, len(row_2a)+1, 3):
        if pd.notna(row_2a[j]):
            date_value = row_2a[j]
            date_obj = datetime.strptime(str(date_value), "%Y-%m-%d %H:%M:%S")
            patient_first_check_time = df_2a.iloc[i, 1]
            time_diff = (date_obj - patient_first_check_time).total_seconds() / 3600.0
            time_diff += additional_hours

            V_value = row_2a[j + 2]

            time_values.append(time_diff)
            V_values.append(V_value)

        else:
            break

    time_list.extend(time_values)
    V_list.extend(V_values)

result_df = pd.DataFrame({"time_list": time_list, "V_list": V_list})
times = pd.read_excel("ans/times.xlsx", engine='openpyxl')
times = times['repeat_times'].tolist()
df = result_df.fillna(result_df.mean())
X = df['time_list'].values.reshape(-1, 1)
y = df['V_list'].values

# 设置多项式度数
degree = 2

# 创建多项式回归模型
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X, y)
y_pred = polyreg.predict(X)

differences = np.abs(y - y_pred)

average_m = []
index = 0
for t in times:
    total_money = sum(differences[index:index+t])
    average = total_money / t
    average_m.append(average)
    index += t

average_m= pd.DataFrame(average_m, columns=['Difference'])
average_m.to_excel("ans/2_a_diff.xlsx", index=False)

losses = [((y[:i] - y_pred[:i]) ** 2).mean() for i in range(1, len(y) + 1)]

coefficients = polyreg.named_steps['linearregression'].coef_
intercept = polyreg.named_steps['linearregression'].intercept_
print(f"多项式回归方程: y = {intercept:.2f} + {coefficients[1]:.2f}x + {coefficients[2]:.5f}x^2")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.regplot(x=X.squeeze(), y=y, scatter=True, order=degree, ci=None, color='#0057E7', line_kws={"color": "#D55E00"})#33FF57 #009E73
plt.title('Data points with Polynomial Fit Curve')

plt.subplot(1, 2, 2)
plt.plot(losses, color='#009E73')
plt.title('Loss during Training')
plt.xlabel('Number of Data Points')
plt.ylabel('Mean Squared Error')

plt.tight_layout()
plt.show()