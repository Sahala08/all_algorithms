import pandas as pd
import math
import os
import glob
from enum import Enum
from dataclasses import dataclass

# ================= 1. 跌倒检测核心逻辑 =================

class FallState(Enum):
    STAND = 0
    POTENTIAL_FALL = 1
    IMPACT_DETECTED = 2
    LAYING = 3

@dataclass
class ImuSample:
    ax: float; ay: float; az: float
    gx: float; gy: float; gz: float
    roll: float; pitch: float; yaw: float; dt: float

class FallDetector:
    def __init__(self):
        # 跌倒检测阈值 (与C语言代码一致)
        self.W_FALL_THRESH = 1.0         
        self.TILT_FALL_THRESH = 0.78       
        self.ACC_IMPACT_THRESH = 1.2  
        self.TILT_LYING_THRESH = 1.5        
        self.FALL_TIME_WINDOW = 0.8          
        self.LAYING_MIN_TIME = 1.0         
        self.GYRO_STILL_THRESH = 0.5         
        self.ACC_STILL_THRESH = 0.2 
        self.LPF_BETA = 0.9

        self.reset()

    def reset(self):
        self.state = FallState.STAND
        self.time_since_potential = 0.0
        self.time_in_lying = 0.0
        self.acc_lp = 0.981
        self.gyro_lp = 0.0

    def update(self, s: ImuSample):
        # 计算特征 (对应 Imu_ComputeFeatures)
        acc_norm = math.sqrt(s.ax**2 + s.ay**2 + s.az**2)
        gyro_norm = math.sqrt(s.gx**2 + s.gy**2 + s.gz**2)
        # roll/pitch 合成倾斜角
        tilt_rad = math.sqrt(s.roll**2 + s.pitch**2)

        # 一阶低通滤波
        self.acc_lp = self.LPF_BETA * self.acc_lp + (1.0 - self.LPF_BETA) * acc_norm
        self.gyro_lp = self.LPF_BETA * self.gyro_lp + (1.0 - self.LPF_BETA) * gyro_norm
        
        fall_event = False

        if self.state == FallState.STAND:
            if gyro_norm > self.W_FALL_THRESH and tilt_rad > self.TILT_FALL_THRESH:
                self.state = FallState.POTENTIAL_FALL
                self.time_since_potential = 0.0

        elif self.state == FallState.POTENTIAL_FALL:
            self.time_since_potential += s.dt
            if acc_norm > self.ACC_IMPACT_THRESH:
                self.state = FallState.IMPACT_DETECTED
                self.time_in_lying = 0.0
            elif self.time_since_potential > self.FALL_TIME_WINDOW:
                self.state = FallState.STAND

        elif self.state == FallState.IMPACT_DETECTED:
            self.time_since_potential += s.dt
            # 躺平且静止判定
            if (tilt_rad > self.TILT_LYING_THRESH and 
                self.gyro_lp < self.GYRO_STILL_THRESH and 
                abs(self.acc_lp - 0.981) < self.ACC_STILL_THRESH):
                self.time_in_lying += s.dt
                if self.time_in_lying > self.LAYING_MIN_TIME:
                    self.state = FallState.LAYING
                    fall_event = True 
            else:
                self.time_in_lying = 0.0

            if tilt_rad < self.TILT_FALL_THRESH and self.time_since_potential > self.FALL_TIME_WINDOW:
                self.state = FallState.STAND

        elif self.state == FallState.LAYING:
            if tilt_rad < self.TILT_FALL_THRESH and self.gyro_lp > self.GYRO_STILL_THRESH:
                self.state = FallState.STAND
        
        return fall_event

# ================= 2. 完善后的四元数转欧拉角 (参考C代码) =================

def quaternion_to_euler(q_list):
    """
    参考 C 语言实现：将四元数 [w, x, y, z] 转化为欧拉角 (roll, pitch, yaw)
    输入：q_list = [q0, q1, q2, q3] 分别对应 w, x, y, z
    输出：roll, pitch, yaw (弧度)
    """
    w, x, y, z = q_list[0], q_list[1], q_list[2], q_list[3]

    # 1. 归一化以避免数值误差
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0.0:
        w /= norm
        x /= norm
        y /= norm
        z /= norm
    else:
        return 0.0, 0.0, 0.0

    # 2. roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 3. pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        # 使用 copysign 处理溢出情况
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # 4. yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

# ================= 3. 数据处理与地形分析接口 =================

class FallDataAnalyzer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.detector = FallDetector()

    def analyze_file(self, file_path):
        """分析单个 CSV 文件"""
        try:
            df = pd.read_csv(file_path)
            if df.empty: return 0, 0
            
            # 计算动态采样频率 dt
            if 'time' in df.columns and len(df) > 1:
                avg_dt = df['time'].diff().mean()
            else:
                avg_dt = 0.005 # 默认 200Hz

            self.detector.reset()
            fall_count = 0

            for _, row in df.iterrows():
                # 获取四元数并转换 (q_0=w, q_1=x, q_2=y, q_3=z)
                q = [row['q_0'], row['q_1'], row['q_2'], row['q_3']]
                r, p, y = quaternion_to_euler(q)

                # 构造样本数据
                sample = ImuSample(
                    ax=row['acc_x'], ay=row['acc_y'], az=row['acc_z'],
                    # 假设 CSV 中 gyro 单位是 deg/s，转为弧度制
                    gx=math.radians(row['gyro_x']), 
                    gy=math.radians(row['gyro_y']), 
                    gz=math.radians(row['gyro_z']),
                    roll=r, pitch=p, yaw=y,
                    dt=avg_dt
                )

                if self.detector.update(sample):
                    fall_count += 1
            
            return fall_count, len(df) * avg_dt
        except Exception as e:
            print(f"Error in {file_path}: {e}")
            return 0, 0

    def run_analysis(self):
        """遍历 terrain_data 目录并输出报告"""
        search_path = os.path.join(self.data_root, "terrain_data", "**", "*.csv")
        files = glob.glob(search_path, recursive=True)
        
        file_results = []
        for f in files:
            terrain_type = os.path.basename(os.path.dirname(f))
            file_name = os.path.basename(f)
            falls, duration = self.analyze_file(f)
            
            file_results.append({
                "地形": terrain_type,
                "文件名": file_name,
                "摔倒次数": falls,
                "时长(s)": round(duration, 2)
            })

        # 输出详细表格
        df_res = pd.DataFrame(file_results)
        if df_res.empty:
            print("未找到任何数据文件。")
            return

        print("\n--- 详细检测列表 ---")
        print(df_res.to_string(index=False))

        # 汇总频率分析
        print("\n" + "="*20 + " 跌倒频率汇总报告 " + "="*20)
        summary = df_res.groupby("地形").agg({"摔倒次数": "sum", "时长(s)": "sum"})
        # 计算每小时跌倒频率
        summary["频率(次/小时)"] = (summary["摔倒次数"] / (summary["时长(s)"] / 3600)).round(2)
        print(summary)

# ================= 4. 执行入口 =================

if __name__ == "__main__":
    # 假设数据结构为 ./data/terrain_data/grass/log.csv
    # 修改 data_root 为你的实际 'data' 文件夹路径
    analyzer = FallDataAnalyzer(data_root="./data")
    analyzer.run_analysis()