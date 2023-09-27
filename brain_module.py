import cv2

import pandas as pd
from collections import Counter


from PIL import Image
import numpy as np


color_to_brain_area = {
    (140, 2, 149): 'HM_ACA_R_Ratio',
    (43, 207, 63): 'HM_ACA_L_Ratio',
    (31, 98, 191): 'HM_MCA_R_Ratio',
    (237, 233, 76): 'HM_MCA_L_Ratio',
    (24, 137, 185): 'HM_PCA_R_Ratio',
    (251, 145, 6): 'HM_PCA_L_Ratio',
    (3,126,102): 'HM_Pons_Medulla_R_Ratio',
    (212, 5, 12): 'HM_Pons_Medulla_L_Ratio',
    (34, 174, 34): 'HM_Cerebellum_R_Ratio',
    (201, 203, 202): 'HM_Cerebellum_L_Ratio',
    # (0, 0, 0): 'Background'
}


def closest_color(rgb, color_dict):
    min_colors = min(
        color_dict.keys(),
        key=lambda color: (color[0] - rgb[0]) ** 2 + (color[1] - rgb[1]) ** 2 + (color[2] - rgb[2]) ** 2
    )
    return min_colors

def is_black(rgb, threshold=50):
    return max(rgb) <= threshold

from sklearn.cluster import KMeans

def segment_image(image, n_segments):
    h, w, _ = image.shape
    image_reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_segments, random_state=0).fit(image_reshaped)
    segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape((h, w, 3))
    return segmented_image.astype(int)

def process_image(image_path, output_path, color_dict, n_segments=10):
    img = Image.open(image_path)
    img = img.convert("RGB")
    pixels = np.array(img)

    # 对图像进行分段
    segmented_pixels = segment_image(pixels, n_segments)

    new_pixels = np.copy(segmented_pixels)  # 复制分段后的像素

    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            if not is_black(new_pixels[i, j]):  # 如果不是黑色
                new_pixels[i, j] = closest_color(new_pixels[i, j], color_dict)

    new_pixels = new_pixels.astype(np.uint8)
    new_img = Image.fromarray(new_pixels)
    new_img.save(output_path)


# 使用方法

def draw_distribution(processed_path, hm_ratios, output_path, nth_data):
    # 读取标准图像
    standard_img = Image.open(processed_path)
    standard_pixels = np.array(standard_img)

    # 创建一个与标准图像同样大小的空白数组来存储输出图像的像素值
    output_pixels = np.zeros((standard_pixels.shape[0], standard_pixels.shape[1], 3), dtype=np.uint8)

    # 为输出图像上色
    for i in range(standard_pixels.shape[0]):
        for j in range(standard_pixels.shape[1]):
            color = tuple(standard_pixels[i, j][:3])  # 取RGB值，忽略alpha值
            if color in color_to_brain_area:
                area = color_to_brain_area[color]
                ratio = hm_ratios.loc[nth_data, area]  # 使用指定行的数据
                red_intensity = int(ratio * 255)  # 将概率转换为0到255的整数值
                output_pixels[i, j] = [red_intensity, 0, 0]  # 染成红色
            else:
                grey_intensity = int(np.mean(standard_pixels[i, j][:3]))
                output_pixels[i, j] = [grey_intensity, grey_intensity, grey_intensity]  # 保持灰色

    output_filename = f"{output_path}/result_{nth_data}.png"

    colored_img = Image.fromarray(output_pixels.astype(np.uint8), 'RGB')
    colored_img.save(output_filename)



if __name__ == '__main__':
    standard_path = '4.png'
    processed_path = "temp/processed.png"

    process_image(standard_path, processed_path, color_to_brain_area)

    df_2 = pd.read_excel('table2.xlsx', engine='openpyxl')
    df_3 = pd.read_excel('table3.xlsx', engine='openpyxl')

    selected_columns_1 = df_2.columns[3 :13].tolist()
    selected_columns_2 = df_2.columns[14 :24].tolist()
    selected_columns_3 = df_3.columns[6 :8].tolist()
    # selected_columns_1.append('PatientID')
    df_2 = df_2.iloc[:160]
    hm_ratios = df_2[selected_columns_1]  # HM血肿  10个字段
    em_ratios = df_2[selected_columns_2]  # EM水肿 10个字段

    for i in range(0, 160):
        draw_distribution(processed_path, hm_ratios, "new_image", i)

    df3 = df_3[selected_columns_3]  # 表3的形态字段






