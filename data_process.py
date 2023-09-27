import os
import pandas as pd
from sqlalchemy import create_engine

from mydb import Database

db = Database()

def print_table_info(table_name):
    column_names, column_types, first_row_values = query_table_info('brain', table_name)

    print(f"表名: {table_name}")
    for i in range(len(column_names)) :
        print(f"字段名: {column_names[i]}, 数据类型: {column_types[i]}")
    print()

def print_ziduan(table_name):
    column_names, column_types, first_row_values = query_table_info('brain', table_name)

    print(f"表名: {table_name}")
    for i in range(len(column_names)) :
        print(f"'{column_names[i]}'", end=',')
    print()

def excel_to_sqlite(excel_path: str, db_name: str, table_name: str = 'table'):
    """
    将Excel文件转换为SQLite数据库文件。

    :param excel_path: Excel文件的路径。
    :param db_name: 生成的SQLite数据库文件的名称。
    :param table_name: 在SQLite数据库中创建的表的名称，默认为'table'。
    """
    db_folder_path = './database'
    os.makedirs(db_folder_path, exist_ok=True)

    df = pd.read_excel(excel_path, engine='openpyxl')

    if table_name == 'patient_time':
        df = pd.read_excel(excel_path, engine='openpyxl', usecols="A:O")

    db_path = os.path.join(db_folder_path, f'{db_name}.db')
    engine = create_engine(f'sqlite:///{db_path}')

    df.to_sql(table_name, con=engine, index=False, if_exists='replace')

def query_table_info(db_name, table_name):
    db = Database(db_name)
    columns_info = db.query(f"PRAGMA table_info({table_name})")

    first_row = db.query_one(f"SELECT * FROM {table_name} LIMIT 1")

    # 获取字段名、数据类型和第一条数据该字段的值
    column_names = [column[1] for column in columns_info]
    column_types = [column[2] for column in columns_info]
    first_row_values = list(first_row) if first_row is not None else None

    return column_names, column_types, first_row_values

def query_one_patient_every_check(patient_id):
    SQL = f"""
    SELECT 
    pf.ID AS ID,
    pf.admission_serial AS admission_serial,
    ci.*
FROM 
    patient_followup AS pf
JOIN 
    check_info AS ci ON pf.admission_serial = ci.流水号
WHERE 
    pf.followup1_serial IS NOT NULL;
    """

    return db.query(SQL)


if __name__ == '__main__':
    db = Database()

    table_names = ['patient', 'patient_check', 'check_info', 'patient_followup']

    for table_name in table_names :
        print_ziduan(table_name)

    # # 导出表格到Excel文件
    # tables_to_export = ['patient', 'patient_followup', 'check_info', 'patient_check']
    # for table_name in tables_to_export :
    #     excel_file_name = f"{table_name}.xlsx"
    #     db.export_table_to_excel(table_name, excel_file_name)
    #     print(f"{table_name} 数据已成功导出到 {excel_file_name}")
    # res = query_one_patient_every_check(1211)
    #
    # for i in res:
    #     print(i)
#     excel_to_sqlite('new_1.xlsx', 'brain', 'patient')  # 病人信息
#     excel_to_sqlite("表2-患者影像信息血肿及水肿的体积及位置.xlsx", 'brain', 'patient_check')  # 病人检查及随访信息
#     excel_to_sqlite("table3.xlsx", 'brain', 'check_info')  # 流水号到检查信息
#     excel_to_sqlite("table4.xlsx", 'brain', 'patient_time')  # 病人各个检查时间
#     excel_to_sqlite("times.xlsx", 'brain', 'repeat')  # 病人检查次数
#
#     db = Database()
#
#     rows = db.query("SELECT * FROM patient LIMIT 5;")
#     for row in rows :
#         print(row)
#
#     rows = db.query("SELECT * FROM patient_check LIMIT 5;")
#     for row in rows :
#         print(row)
#
#
#     rows = db.query("SELECT * FROM check_info LIMIT 5;")
#     for row in rows :
#         print(row)
#
#     rows = db.query("SELECT * FROM patient_time LIMIT 5;")
#     for row in rows :
#         print(row)
#
    # table_names = ['patient', 'patient_check', 'check_info', 'patient_time', 'repeat']
    #
    # for table_name in table_names :
    #     print_table_info(table_name)
#
#     db = Database()
#
#     db.alter('drop table patient_followup')
#     SQL = """
# CREATE TABLE patient_followup AS
# SELECT
#     pi.ID AS ID,
#     pt.入院首次检查流水号 AS admission_serial,
#     pi.发病到首次影像检查时间间隔 AS admission_duration,
#     pt.随访1流水号 AS followup1_serial,
#     (pi.发病到首次影像检查时间间隔 + (strftime('%s', pt.随访1时间点) - strftime('%s', pt.入院首次检查时间点)) / 3600) AS followup1_duration,
#     pt.随访2流水号 AS followup2_serial,
#     (pi.发病到首次影像检查时间间隔 + (strftime('%s', pt.随访2时间点) - strftime('%s', pt.入院首次检查时间点)) / 3600) AS followup2_duration,
#     pt.随访3流水号 AS followup3_serial,
#     (pi.发病到首次影像检查时间间隔 + (strftime('%s', pt.随访3时间点) - strftime('%s', pt.入院首次检查时间点)) / 3600) AS followup3_duration,
#     pt.随访4流水号 AS followup4_serial,
#     (pi.发病到首次影像检查时间间隔 + (strftime('%s', pt.随访4时间点) - strftime('%s', pt.入院首次检查时间点)) / 3600) AS followup4_duration,
#     pt.随访5流水号 AS followup5_serial,
#     (pi.发病到首次影像检查时间间隔 + (strftime('%s', pt.随访5时间点) - strftime('%s', pt.入院首次检查时间点)) / 3600) AS followup5_duration,
#     pt.随访6流水号 AS followup6_serial,
#     (pi.发病到首次影像检查时间间隔 + (strftime('%s', pt.随访6时间点) - strftime('%s', pt.入院首次检查时间点)) / 3600) AS followup6_duration
# FROM
#     patient_time AS pt
# JOIN
#     patient AS pi
# ON
#     pt.ID = pi.ID;
#
#     """
#
#     db.alter(SQL)

#     # print_table_info('patient_followup')
#
#     SQL = """
# -- 1. 创建一个新的表格，与原表格结构相同但带有您想要的数据类型
# CREATE TABLE patient_followup_new AS
# SELECT
#     CAST(ID AS TEXT) AS ID,
#     CAST(admission_serial AS INT) AS admission_serial,
#     CAST(admission_duration AS REAL) AS admission_duration,
#     CAST(followup1_serial AS INT) AS followup1_serial,
#     CAST(followup1_duration AS REAL) AS followup1_duration,
#     CAST(followup2_serial AS INT) AS followup2_serial,
#     CAST(followup2_duration AS REAL) AS followup2_duration,
#     CAST(followup3_serial AS INT) AS followup3_serial,
#     CAST(followup3_duration AS REAL) AS followup3_duration,
#     CAST(followup4_serial AS INT) AS followup4_serial,
#     CAST(followup4_duration AS REAL) AS followup4_duration,
#     CAST(followup5_serial AS INT) AS followup5_serial,
#     CAST(followup5_duration AS REAL) AS followup5_duration,
#     CAST(followup6_serial AS INT) AS followup6_serial,
#     CAST(followup6_duration AS REAL) AS followup6_duration
# FROM patient_followup;
#
# -- 2. 删除原表格
# DROP TABLE patient_followup;
#
# -- 3. 将新表格重命名为原表格的名称
# ALTER TABLE patient_followup_new RENAME TO patient_followup;
#
#     """
#
#     db.execute_multiple_statements(SQL)
#
#     print_table_info('patient_followup')