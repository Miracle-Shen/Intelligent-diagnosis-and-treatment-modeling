import pandas as pd
from datetime import datetime

def A_cal_ill(df_1,dff_1,df_2,df_3):
    ans = []
    time_format = "%Y-%m-%d %H:%M:%S"
    dic = {}
    ppp ={
        0:'2017-3-8 0:50:18',
        1:'2017-3-8 8:38:19'
    }
    V_ans = []
    ratio_V= []
    for i in range(0,100):
        dic[i] = 0

        repeat_times = dff_1.loc[i,'重复次数']
        judge_time = -1  #返回的时间
        deV = None
        deRatio = None
        t1 = dff_1.loc[i,'入院首次检查时间点']#首次检测时间
        det = df_1.loc[i,'发病到首次影像检查时间间隔']#发病时间

        time1 = datetime.strptime(str(t1), time_format)

        v1 = df_2.iloc[i, 2] #首次检测体积

        for t in range(0, repeat_times-1):
            t2 = dff_1.iloc[i, 2 + 2 * (t + 1)]
            time2 = datetime.strptime(str(t2), time_format)
            hours_difference = (time2 - time1).total_seconds() / 3600
            if (hours_difference + det) > 48:
                ans.append(judge_time)
                V_ans.append('None')
                ratio_V.append('None')
                dic[i] = judge_time
                break
            else:
                deV = df_2.iloc[i, 2 + 23 * (t + 1)] - v1
                deRatio = deV / v1
                if ((deV >= 6) or (deRatio >= 0.33)):
                    judge_time = hours_difference + det
                    ans.append(judge_time)
                    V_ans.append(deV)
                    ratio_V.append(deRatio)
                    dic[i] = judge_time
                    break
            if(t==(repeat_times-2)):
                V_ans.append('None')
                ratio_V.append('None')
                ans.append(judge_time)

    return ans,V_ans,ratio_V

def main():
    df_1 = pd.read_excel('data/表1-患者列表及临床信息.xlsx', engine='openpyxl')
    dff_1 = pd.read_excel('data/附表1-检索表格-流水号vs时间.xlsx', engine='openpyxl')
    df_2 = pd.read_excel('data/表2-患者影像信息血肿及水肿的体积及位置.xlsx', engine='openpyxl')
    df_3 = pd.read_excel('data/表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx', engine='openpyxl')
    time_ans, V_ans,ratio_ans = A_cal_ill(df_1,dff_1,df_2,df_3)
    judge_ans = {}
    for i,value in enumerate(time_ans):
        if(value!=-1):
            judge_ans[df_1.loc[i, '入院首次影像检查流水号']]='1'
        else:
            judge_ans[df_1.loc[i, '入院首次影像检查流水号']]='0'
            time_ans[i] = None

    formatted_list = [round(item, 2) if item is not None else None for item in time_ans]

    df = pd.DataFrame(formatted_list, columns=["Time Values"])
    df_v = pd.DataFrame(V_ans, columns=["extendable volumn"])
    df_ratio = pd.DataFrame(ratio_ans, columns=["extendable volumn ratio"])
    patient_ids = list(judge_ans.keys())
    values = list(judge_ans.values())

    df2 = pd.DataFrame({
        'PatientID': patient_ids,
        'value': values
    })
    with pd.ExcelWriter('ans/表4C字段（是否发生血肿扩张）.xlsx') as writer:
        df2.to_excel(writer, index=False, header=True)

    df.to_excel("ans/表4D字段（血肿扩张时间）.xlsx", index=False)
    df_v.to_excel("ans/扩张体积的变化.xlsx", index=False)
    df_ratio.to_excel("ans/扩张体积率.xlsx", index=False)

if __name__ == '__main__':
    main( )


