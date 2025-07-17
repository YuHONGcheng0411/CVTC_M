import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from deepbet import run_bet
from datetime import datetime
import re
from dateutil.parser import parse

# 定义输入和输出文件夹
input_folder = r"G:\test"
output_base_folder = r"F:\dataset\多种模态+SNP"
bet_output_folder = os.path.join(output_base_folder, "BET_Outputs")  # 新建 BET 输出文件夹

# 确保输出文件夹存在
os.makedirs(bet_output_folder, exist_ok=True)
os.makedirs(output_base_folder, exist_ok=True)

# 读取 CSV 文件
excel_file = r"E:\dataset-CVTC\csv\UPENN_血浆生物标志物_cleaned.csv"
df = pd.read_csv(excel_file)

# 创建 subject_id 到 EXAMDATE 列表的映射（支持同一 subject_id 多个 EXAMDATE）
subject_to_examdates = {}
for subject_id, exam_date in zip(df.iloc[:, 1], df.iloc[:, -1]):  # 第二列为 subject_id，最后一列为 EXAMDATE
    if pd.isna(subject_id) or pd.isna(exam_date):
        continue
    if subject_id not in subject_to_examdates:
        subject_to_examdates[subject_id] = []
    subject_to_examdates[subject_id].append(exam_date)

# 打印 subject_id 列表，调试用
print(f"CSV 中的 subject_id 列表：{list(subject_to_examdates.keys())}")

# 记录未匹配的 subject_id
unmatched_subjects = set()


# 解析时间文件夹名称
def parse_time_from_folder_name(folder_name):
    match = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d+\.\d+)", folder_name)
    if match:
        date_part = match.group(1)
        hh = match.group(2)
        mm = match.group(3)
        ss = float(match.group(4))
        time_str = f"{date_part} {hh}:{mm}:{ss:.1f}"
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            print(f"无法解析时间字符串: {time_str}")
            return None
    return None


# 找到包含最接近 exam_date 的时间文件夹的 protocol 文件夹
def find_closest_protocol_folder(protocol_folders, subject_path, exam_date_str):
    try:
        exam_date = parse(exam_date_str)
    except ValueError:
        print(f"无法解析 exam_date: {exam_date_str}")
        return None, None

    if not protocol_folders:
        print(f"protocol_folders 为空")
        return None, None

    closest_protocol = None
    closest_time_folder = None
    min_time_diff = float("inf")

    for protocol_folder in protocol_folders:
        protocol_path = os.path.join(subject_path, protocol_folder)
        if not os.path.isdir(protocol_path) or "Mask" in protocol_folder:
            print(f"跳过 protocol 文件夹 {protocol_folder}：不是文件夹或包含 'Mask'")
            continue

        # 获取 protocol 文件夹下的时间文件夹
        time_folders = [d for d in os.listdir(protocol_path) if os.path.isdir(os.path.join(protocol_path, d))]
        if not time_folders:
            print(f"protocol 文件夹 {protocol_folder} 下无时间文件夹")
            continue

        for time_folder in time_folders:
            folder_time = parse_time_from_folder_name(time_folder)
            if folder_time is None:
                print(f"无法解析时间文件夹: {time_folder}")
                continue

            time_diff = abs((exam_date - folder_time).total_seconds())
            print(f"比较时间：exam_date={exam_date}, folder_time={folder_time}, 差值={time_diff}秒")
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_protocol = protocol_folder
                closest_time_folder = time_folder

    if closest_protocol and closest_time_folder:
        print(
            f"选择 protocol 文件夹: {closest_protocol}, 时间文件夹: {closest_time_folder}, 最小时间差: {min_time_diff}秒")
    else:
        print(f"未找到匹配的 protocol 或时间文件夹")

    return closest_protocol, closest_time_folder


# 遍历文件夹，收集患者编号和对应的 .nii 文件
patient_files = {}  # {image_id: [(nii_file_path, subject_id, time_folder, exam_date)]}
for subject_folder in os.listdir(input_folder):
    subject_path = os.path.join(input_folder, subject_folder)
    if not os.path.isdir(subject_path):
        print(f"跳过 {subject_folder}：不是文件夹")
        continue
    if '_S_' not in subject_folder:
        print(f"跳过 {subject_folder}：不包含 '_S_'")
        continue

    # 提取 subject_id，例如 129_S_1246
    subject_id = subject_folder
    print(f"\n从文件夹提取的 subject_id: {subject_id}")

    # 检查 subject_id 是否在 CSV 文件中
    if subject_id not in subject_to_examdates:
        print(f"subject_id {subject_id} 不在 CSV 文件中，跳过")
        unmatched_subjects.add(subject_id)
        continue

    # 获取所有的 protocol 文件夹
    protocol_folders = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
    if not protocol_folders:
        print(f"subject {subject_id} 未找到有效的 protocol 文件夹")
        continue

    # 对每个 exam_date 进行匹配
    for exam_date in subject_to_examdates[subject_id]:
        print(f"subject_id {subject_id} 对应的 exam_date: {exam_date}")

        # 找到最接近的 protocol 文件夹和时间文件夹
        closest_protocol, closest_time_folder = find_closest_protocol_folder(protocol_folders, subject_path, exam_date)
        if closest_protocol is None or closest_time_folder is None:
            print(f"无法为 subject {subject_id} 的 exam_date {exam_date} 找到匹配的 protocol 或时间文件夹")
            continue

        # 进入最接近的 time_folder
        protocol_path = os.path.join(subject_path, closest_protocol)
        time_folder_path = os.path.join(protocol_path, closest_time_folder)

        # 遍历 image_id 文件夹
        for image_id in os.listdir(time_folder_path):
            if not image_id.startswith('I'):
                continue

            image_folder = os.path.join(time_folder_path, image_id)
            if not os.path.isdir(image_folder):
                continue

            # 收集 .nii 文件
            nii_files = []
            for sub_root, sub_dirs, sub_files in os.walk(image_folder):
                for file in sub_files:
                    if file.endswith('.nii'):
                        nii_file_path = os.path.join(sub_root, file)
                        nii_files.append((nii_file_path, subject_id, closest_time_folder, exam_date))

            if nii_files:
                if image_id not in patient_files:
                    patient_files[image_id] = []
                patient_files[image_id].extend(nii_files)

# 打印所有未匹配的 subject_id
if unmatched_subjects:
    print(f"\n以下 subject_id 未在 CSV 文件中找到：{', '.join(sorted(unmatched_subjects))}")

# 打印提取到的 patient_files
print(f"\n提取到的 patient_files: {list(patient_files.keys())}")

# 遍历收集到的患者编号，与 CSV 文件匹配
for image_id, nii_file_list in patient_files.items():
    for nii_index, (input_path, subject_id, time_folder, exam_date) in enumerate(nii_file_list):
        # 创建以 subject_id 和 exam_date 命名的输出子文件夹（exam_date 格式化为 YYYYMMDD）
        try:
            exam_date_str = parse(exam_date).strftime("%Y%m%d")
        except ValueError:
            print(f"无法解析 exam_date: {exam_date}，跳过")
            continue
        subject_output_folder = os.path.join(output_base_folder, f"{subject_id}_{exam_date_str}")
        os.makedirs(subject_output_folder, exist_ok=True)

        # 定义 BET 输出路径，保存在单独的文件夹，文件名中加入 time_folder 和 exam_date
        brain_path = os.path.join(bet_output_folder,
                                  f"{subject_id}_{exam_date_str}_{image_id}_{time_folder}_file{nii_index + 1}_brain.nii")
        mask_path = os.path.join(bet_output_folder,
                                 f"{subject_id}_{exam_date_str}_{image_id}_{time_folder}_file{nii_index + 1}_mask.nii")

        print(f"Processing file: {input_path}")

        # 运行 deepbet
        run_bet([input_path], [brain_path], [mask_path], threshold=0.5, n_dilate=0, no_gpu=False, skip_broken=False)

        if not os.path.exists(brain_path):
            print(f"文件未生成: {brain_path}")
            continue

        # 加载脑部图像
        brain_img = nib.load(brain_path)
        brain_data = brain_img.get_fdata()

        # 保存所有切片（sagittal, coronal, axial）
        num_slices_x = brain_data.shape[0]  # Sagittal
        num_slices_y = brain_data.shape[1]  # Coronal
        num_slices_z = brain_data.shape[2]  # Axial

        # 保存 Sagittal 切片（X轴）
        for i in range(num_slices_x):
            slice_data = brain_data[i, :, :]
            if not np.all(slice_data == 0):
                plt.figure(figsize=(6, 6))
                plt.imshow(slice_data, cmap='gray')
                plt.axis('off')
                output_image_path = os.path.join(
                    subject_output_folder,
                    f"{subject_id}_{exam_date_str}_{image_id}_{time_folder}_file{nii_index + 1}_sagittal_slice_{i + 1}.png"
                )
                plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

        # 保存 Coronal 切片（Y轴）
        for i in range(num_slices_y):
            slice_data = brain_data[:, i, :]
            if not np.all(slice_data == 0):
                plt.figure(figsize=(6, 6))
                plt.imshow(slice_data, cmap='gray')
                plt.axis('off')
                output_image_path = os.path.join(
                    subject_output_folder,
                    f"{subject_id}_{exam_date_str}_{image_id}_{time_folder}_file{nii_index + 1}_coronal_slice_{i + 1}.png"
                )
                plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

        # 保存 Axial 切片（Z轴）
        for i in range(num_slices_z):
            slice_data = brain_data[:, :, i]
            if not np.all(slice_data == 0):
                plt.figure(figsize=(6, 6))
                plt.imshow(slice_data, cmap='gray')
                plt.axis('off')
                output_image_path = os.path.join(
                    subject_output_folder,
                    f"{subject_id}_{exam_date_str}_{image_id}_{time_folder}_file{nii_index + 1}_axial_slice_{i + 1}.png"
                )
                plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

print("over")