import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/za_attitude_20250912.csv')

pos_diff = df['left_motor_pos'] - df['right_motor_pos']
pos_sum = df['left_motor_pos'] + df['right_motor_pos']
height = (df['height'] - 8879)

# 绘制pos_diff曲线
plt.figure(figsize=(10, 6))
plt.plot(df['left_motor_pos'][6250:7250], label='Left Motor Position')
plt.plot(df['right_motor_pos'][6250:7250], label='Right Motor Position')

plt.plot(pos_diff[6250:7250], label='Pos Diff')
plt.plot(pos_sum[6250:7250], label='Pos Sum')
plt.plot(height[6250:7250], label='height')
plt.xlabel('Index')
plt.ylabel('Position Difference')
# plt.title('Position Difference between left_motor_pos and right_motor_pos')
plt.legend()
plt.grid(True)
plt.show()

# print(f"pos_diff: {pos_diff}")