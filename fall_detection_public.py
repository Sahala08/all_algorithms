import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# 设置中文显示（解决Matplotlib中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 参数配置
# ==========================================
ROOT_PATH = '.\\data\\SisFall_dataset' 
FS = 200           # 采样频率 200Hz
G_THRESHOLD = 2.5  # 撞击阈值 (滤波后的SVM)
ANGLE_THRESHOLD = 50 # 姿态夹角阈值 (度)

# 转换因子
ADXL345_G = (2 * 16 * 1.0) / (2**13)
ITG3200_DEG = (2 * 2000 * 1.0) / (2**16)
MMA8451Q_G = (2 * 8 * 1.0) / (2**14)

# ==========================================
# 2. 核心算法函数
# ==========================================
def butter_lowpass_filter(data, cutoff=5, fs=200, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def load_sisfall_file(file_path):
    try:
        # 读取并清理数据
        data = pd.read_csv(file_path, header=None, sep=',').dropna(axis=1, how='all')
        data.iloc[:, -1] = data.iloc[:, -1].astype(str).str.replace(';', '')
        data = data.astype(float)
        # 单位转换
        data.iloc[:, 0:3] = data.iloc[:, 0:3] * ADXL345_G   
        data.iloc[:, 3:6] = data.iloc[:, 3:6] * ITG3200_DEG 
        data.iloc[:, 6:9] = data.iloc[:, 6:9] * MMA8451Q_G  
        data.columns = ['acc1_x', 'acc1_y', 'acc1_z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc2_x', 'acc2_y', 'acc2_z']
        return data
    except:
        return None

def detect_fall_logic(df):
    """
    返回: (bool 判定结果, float 最大撞击值, float 最终夹角)
    """
    acc_raw = df[['acc1_x', 'acc1_y', 'acc1_z']].values
    
    # 1. 5Hz低通滤波，去除类似D11瘫坐产生的瞬间高频震荡
    acc_filtered = butter_lowpass_filter(acc_raw, cutoff=5, fs=FS)
    svm = np.sqrt(np.sum(np.square(acc_filtered), axis=1))

    # # 计算采样点数
    # sample_count = int(1.0 * FS)
    # acc_before = acc_filtered[:sample_count, :]
    # svm_before = svm[:sample_count]

    # # 计算与垂直轴 (Y) 的夹角
    # cos_theta = np.abs(avg_acc[1]) / avg_svm
    # angle = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))


    max_impact = np.max(svm)
    impact_idx = np.argmax(svm)
    
    # 阈值判定1：如果撞击不够强，直接排除
    if max_impact < G_THRESHOLD:
        return False, max_impact, 0.0

    # 阈值判定2：姿态检查 (撞击后 0.5s - 1.5s)
    wait_f = int(0.5 * FS)
    obs_f = int(1.0 * FS)
    if impact_idx + wait_f + obs_f > len(df):
        return False, max_impact, 0.0 # 样本太短无法判定
    
    # 计算均值姿态
    avg_acc = np.mean(acc_filtered[impact_idx + wait_f : impact_idx + wait_f + obs_f, :], axis=0)
    avg_svm = np.sqrt(np.sum(np.square(avg_acc)))
    
    # 计算与垂直轴(Y)的夹角
    cos_theta = np.abs(avg_acc[1]) / avg_svm
    angle = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
    print(f"姿态: {angle:.1f}°")
    
    # 最终逻辑：强撞击 + 身体倾斜超过阈值
    if angle > ANGLE_THRESHOLD:
        return True, max_impact, angle
    else:
        return False, max_impact, angle


# ==========================================
# 3. 单文件检测与绘图函数
# ==========================================
def run_single_file_test_with_plot(file_path):
    """
    对单个文件进行摔倒检测，并绘制SVM滤波前后对比图，标注撞击时刻
    """
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    df = load_sisfall_file(file_path)
    if df is None: return

    file_name = os.path.basename(file_path)
    acc_raw = df[['acc1_x', 'acc1_y', 'acc1_z']].values
    
    # --- 1. 计算原始 SVM ---
    svm_raw = np.sqrt(np.sum(np.square(acc_raw), axis=1))
    
    # --- 2. 低通滤波并计算滤波后 SVM ---
    acc_filtered = butter_lowpass_filter(acc_raw, cutoff=5, fs=FS)
    svm_filtered = np.sqrt(np.sum(np.square(acc_filtered), axis=1))
    
    # --- 3. 运行检测逻辑 ---
    max_impact = np.max(svm_filtered)
    impact_idx = np.argmax(svm_filtered)
    
    is_detected = False
    angle = 0.0
    status_msg = "未触发阈值"
    
    # 检测逻辑判断
    if max_impact >= G_THRESHOLD:
        wait_f = int(0.5 * FS)
        obs_f = int(1.0 * FS)
        
        if impact_idx + wait_f + obs_f <= len(df):
            # 姿态计算
            avg_acc = np.mean(acc_filtered[impact_idx + wait_f : impact_idx + wait_f + obs_f, :], axis=0)
            avg_svm = np.sqrt(np.sum(np.square(avg_acc)))
            cos_theta = np.abs(avg_acc[1]) / avg_svm
            angle = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
            print(f"姿态: {angle:.1f}°")
            
            if angle > ANGLE_THRESHOLD:
                is_detected = True
                status_msg = f"检测到摔倒 (强度:{max_impact:.2f}g, 角度:{angle:.1f}°)"
            else:
                status_msg = f"误报过滤 (强度:{max_impact:.2f}g, 角度:{angle:.1f}°)"
        else:
            status_msg = "撞击太靠后，无法判定姿态"

    # --- 4. 绘图 ---
    plt.figure(figsize=(12, 6))
    
    # 绘制原始数据（浅色）
    plt.plot(svm_raw, color='lightgray', alpha=0.7, label='原始数据 (SVM Raw)')
    
    # 绘制滤波后数据（深色）
    plt.plot(svm_filtered, color='blue', linewidth=1.5, label='5Hz低通滤波后 (SVM Filtered)')
    
    # 标注撞击时刻
    if max_impact >= G_THRESHOLD:
        plt.axvline(x=impact_idx, color='red', linestyle='--', label='检测到撞击点')
        # 标注姿态观察窗口
        plt.axvspan(impact_idx + 100, impact_idx + 300, color='green', alpha=0.2, label='1s姿态观察期')
        # 在图上写文字
        plt.text(impact_idx, max_impact + 0.2, f'撞击点: {max_impact:.2f}g', color='red', weight='bold')

    # 设置图表
    actual_label = "摔倒" if file_name.startswith('F') else "日常"
    plt.title(f"文件分析: {file_name} (真值: {actual_label}) \n结论: {status_msg}")
    plt.xlabel("帧数 (Frame) | 采样率: 200Hz")
    plt.ylabel("合加速度 SVM (g)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 自动保存或显示
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. 批量检测主程序
# ==========================================
def run_batch_evaluation():
    stats = {
        'TP': 0, # 真阳性 (摔倒被正确检测)
        'FP': 0, # 假阳性 (日常被误报为摔倒)
        'TN': 0, # 真阴性 (日常被正确识别)
        'FN': 0  # 假阴性 (摔倒没被检测出来)
    }
    
    results_list = []

    print("开始批量检测...")
    print("-" * 50)

    # 遍历受试者文件夹
    for sub in sorted(os.listdir(ROOT_PATH)):
        sub_path = os.path.join(ROOT_PATH, sub)
        if not os.path.isdir(sub_path): continue
        
        print(f"正在处理: {sub}...")
        
        for file_name in os.listdir(sub_path):
            if not file_name.endswith('.txt'): continue
            
            file_path = os.path.join(sub_path, file_name)
            df = load_sisfall_file(file_path)
            if df is None: continue
            
            # 运行检测算法
            is_detected, impact, angle = detect_fall_logic(df)
            
            # 真实情况标注
            is_actual_fall = file_name.startswith('F')
            
            # 统计
            if is_actual_fall and is_detected:
                stats['TP'] += 1
            elif not is_actual_fall and is_detected:
                stats['FP'] += 1
                print(f"  [误报] {file_name} | 强度:{impact:.2f}g | 角度:{angle:.1f}°")
            elif is_actual_fall and not is_detected:
                stats['FN'] += 1
                print(f"  [漏报] {file_name} | 强度:{impact:.2f}g | 角度:{angle:.1f}°")
            else:
                stats['TN'] += 1

    total = sum(stats.values())
    accuracy = (stats['TP'] + stats['TN']) / total * 100
    sensitivity = stats['TP'] / (stats['TP'] + stats['FN']) * 100 if (stats['TP'] + stats['FN']) > 0 else 0
    specificity = stats['TN'] / (stats['TN'] + stats['FP']) * 100 if (stats['TN'] + stats['FP']) > 0 else 0

    print("-" * 50)
    print("检测完成！统计结果如下：")
    print(f"总文件数: {total}")
    print(f"准确率 (Accuracy):    {accuracy:.2f}%")
    print(f"灵敏度 (Sensitivity): {sensitivity:.2f}% (越高代表漏报越少)")
    print(f"特异度 (Specificity): {specificity:.2f}% (越高代表误报越少)")
    print("-" * 50)
    print(f"详细详情: TP={stats['TP']}, TN={stats['TN']}, FP={stats['FP']}, FN={stats['FN']}")

if __name__ == "__main__":
    # 单文件摔倒检测测试
    run_single_file_test_with_plot(".\\data\\SisFall_dataset\\SA01\\F01_SA01_R01.txt")

    # 批量摔倒检测
    # run_batch_evaluation()