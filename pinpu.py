import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置参数
CHUNK = 1024  # 每次读取的样本数
FORMAT = pyaudio.paInt16  # 格式
CHANNELS = 2  # 声道数
RATE = 48000  # 采样率（每秒样本数）
N_FFT_BINS = 512  # FFT 使用的频谱分析点数
FREQ_LIMIT = 8192  # 频率范围上限

# 创建 PyAudio 对象
p = pyaudio.PyAudio()

# 手动指定扬声器设备索引
speaker_device_index = 1  # 你可以将此值修改为你系统中的扬声器设备索引

# 打开音频流，使用指定的扬声器设备作为输入源
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=speaker_device_index,
                frames_per_buffer=CHUNK)

# 创建画布
fig, ax = plt.subplots()
x = np.linspace(0, RATE/2, N_FFT_BINS//2)  # 频率范围从0到采样率的一半
line, = ax.plot(x, np.random.rand(N_FFT_BINS//2))

# 更新函数，用于更新频谱图
def update(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    data = data[::2]  # 只取一个声道的数据
    fft_data = np.fft.fft(data, n=N_FFT_BINS)  # 使用FFT计算频谱
    fft_magnitude = np.abs(fft_data)[:N_FFT_BINS//2]  # 获取频谱幅度，只取一半
    line.set_ydata(fft_magnitude)  # 更新频谱图
    return line,

# 开始动画
ani = FuncAnimation(fig, update, blit=True)
plt.xlim(0, FREQ_LIMIT)  # 设置x轴范围为[0, 4000]
plt.ylim(0, 500000)  # 设置y轴范围为[0, 5000]
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# 关闭流
stream.stop_stream()
stream.close()
p.terminate()
