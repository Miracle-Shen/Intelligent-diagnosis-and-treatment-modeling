from mydb import Database

db = Database()

fields = [
    '流水号','original_shape_Elongation','original_shape_Flatness','original_shape_LeastAxisLength','original_shape_MajorAxisLength','original_shape_Maximum2DDiameterColumn','original_shape_Maximum2DDiameterRow','original_shape_Maximum2DDiameterSlice','original_shape_Maximum3DDiameter','original_shape_MeshVolume','original_shape_MinorAxisLength','original_shape_Sphericity','original_shape_SurfaceArea','original_shape_SurfaceVolumeRatio','original_shape_VoxelVolume','NCCT_original_firstorder_10Percentile','NCCT_original_firstorder_90Percentile','NCCT_original_firstorder_Energy','NCCT_original_firstorder_Entropy','NCCT_original_firstorder_InterquartileRange','NCCT_original_firstorder_Kurtosis','NCCT_original_firstorder_Maximum','NCCT_original_firstorder_MeanAbsoluteDeviation','NCCT_original_firstorder_Mean','NCCT_original_firstorder_Median','NCCT_original_firstorder_Minimum','NCCT_original_firstorder_Range','NCCT_original_firstorder_RobustMeanAbsoluteDeviation','NCCT_original_firstorder_RootMeanSquared','NCCT_original_firstorder_Skewness','NCCT_original_firstorder_Uniformity','NCCT_original_firstorder_Variance',

    # ... 其他字段
]

aliased_fields = []
for i in range(7):  # 7 代表 admission 和 6 次 followup
    for field in fields:
        aliased_fields.append(f"ci{i}.{field} as {field}_{i}")

aliased_fields_str = ", ".join(aliased_fields)

SQL = f"""
SELECT 
    p.*,  -- patient表的所有列
    pc.*,  -- patient_check表的所有列
    pf.*,  -- patient_followup表的所有列
    {aliased_fields_str}
FROM patient p
JOIN patient_check pc ON p.id = pc.ID
JOIN patient_followup pf ON p.id = pf.id
LEFT JOIN check_info ci0 ON pf.followup6_serial = ci0.流水号
LEFT JOIN check_info ci1 ON pf.admission_serial = ci1.流水号 
LEFT JOIN check_info ci2 ON pf.followup1_serial = ci2.流水号
LEFT JOIN check_info ci3 ON pf.followup2_serial = ci3.流水号
LEFT JOIN check_info ci4 ON pf.followup3_serial = ci4.流水号
LEFT JOIN check_info ci5 ON pf.followup4_serial = ci5.流水号
LEFT JOIN check_info ci6 ON pf.followup5_serial = ci6.流水号

"""

print(SQL)



print(db.export_query_to_excel(SQL, 'temp/big_table_2.xlsx'))