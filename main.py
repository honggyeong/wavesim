import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from matplotlib.patches import Rectangle, Circle, Polygon

from matplotlib.gridspec import GridSpec

from math import sqrt

from matplotlib.widgets import Slider





class WaveSimulation:

    def __init__(self, width=100, height=100, dx=1, dt=0.1):

        self.width = width

        self.height = height

        self.dx = dx

        self.dt = dt



        # 시뮬레이션 공간 초기화

        self.u = np.zeros((height, width))

        self.u_prev = np.zeros((height, width))

        self.u_next = np.zeros((height, width))



        # 파동 매개변수

        self.c = 5.0

        self.damping = 0.999

        self.diffraction_range = 3

        self.diffraction_strength = 0.2

        self.frequency = 0.5



        # 센서 및 파동원 설정

        self.sensors = [(width - 1, int(y)) for y in np.linspace(0, height - 1, 16)]

        self.sensor_data = []

        self.recorded_sensor_data = None

        self.recording_start_frame = None

        self.recording_frames = int(5.0 / (self.dt))  # 5초 동안의 프레임 수

        self.wave_sources = [(0, y) for y in range(height)]



    def setup_visualization(self, title="Wave Simulation"):

        # 그래프 설정

        self.fig = plt.figure(figsize=(15, 10))

        gs = GridSpec(3, 2, height_ratios=[1, 1, 0.1])



        # 파동 시뮬레이션 플롯

        self.ax1 = self.fig.add_subplot(gs[0, 0])

        self.ax1.set_title(title)

        self.img = self.ax1.imshow(self.u, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')

        self.fig.colorbar(self.img, ax=self.ax1, label='Wave Amplitude')



        # 실시간 센서 데이터 플롯

        self.ax2 = self.fig.add_subplot(gs[0, 1])

        self.ax2.set_title('Real-time Sensor Readings')

        self.ax2.set_xlabel('Time')

        self.ax2.set_ylabel('Amplitude')

        self.lines = [self.ax2.plot([], [], label=f'Sensor {i + 1}')[0] for i in range(16)]

        self.ax2.legend(loc='upper right', ncol=2, fontsize='small')

        self.ax2.set_xlim(0, 100)

        self.ax2.set_ylim(-1, 1)

        self.ax2.grid(True)



        # 센서 데이터 등고선 플롯

        self.ax3 = self.fig.add_subplot(gs[1, :])

        self.ax3.set_title('Sensor Data Contour')

        self.ax3.set_xlabel('Sensor Number')

        self.ax3.set_ylabel('Time')

        self.contour = None



        # 진동수 슬라이더

        self.ax_freq = self.fig.add_subplot(gs[2, :])

        self.freq_slider = Slider(

            ax=self.ax_freq,

            label='Frequency',

            valmin=0.1,

            valmax=2.0,

            valinit=self.frequency,

        )

        self.freq_slider.on_changed(self.update_frequency)



        # 센서 위치 표시

        for x, y in self.sensors:

            self.ax1.plot(x, y, 'go', markersize=3)



        plt.tight_layout()



    def update_frequency(self, val):

        self.frequency = val



    def add_obstacle(self, shape, position=(50, 50), size=10, width=None, height=None):


        x, y = position

        self.obstacle = {"shape": shape, "pos": position, "size": size}

        self.obstacle_mask = np.zeros((self.height, self.width), dtype=bool)



        if width is None:

            width = size

        if height is None:

            height = size



        if shape == "cube":

            rect = Rectangle((x - size // 2, y - size // 2), size, size, facecolor='gray')

            self.ax1.add_patch(rect)

            y_start, y_end = max(0, y - size // 2), min(self.height, y + size // 2)

            x_start, x_end = max(0, x - size // 2), min(self.width, x + size // 2)

            self.obstacle_mask[y_start:y_end, x_start:x_end] = True



        elif shape == "cuboid":

            rect = Rectangle((x - width // 2, y - height // 2), width, height, facecolor='gray')

            self.ax1.add_patch(rect)

            y_start, y_end = max(0, y - height // 2), min(self.height, y + height // 2)

            x_start, x_end = max(0, x - width // 2), min(self.width, x + width // 2)

            self.obstacle_mask[y_start:y_end, x_start:x_end] = True



        elif shape == "cylinder":

            circle = Circle((x, y), size // 2, facecolor='gray')

            self.ax1.add_patch(circle)

            for i in range(self.height):

                for j in range(self.width):

                    if sqrt((i - y) ** 2 + (j - x) ** 2) <= size // 2:

                        self.obstacle_mask[i, j] = True



        elif shape == "triangular_prism":

            triangle_height = size * sqrt(3) / 2

            points = [

                [x, y - size // 2],

                [x - size // 2, y + triangle_height // 2],

                [x + size // 2, y + triangle_height // 2]

            ]

            polygon = Polygon(points, facecolor='gray')

            self.ax1.add_patch(polygon)

            for i in range(self.height):

                for j in range(self.width):

                    if self.point_in_triangle(

                            (j, i),

                            points[0],

                            points[1],

                            points[2]

                    ):

                        self.obstacle_mask[i, j] = True



        self.u[self.obstacle_mask] = np.nan

        self.u_prev[self.obstacle_mask] = np.nan

        self.u_next[self.obstacle_mask] = np.nan



    def point_in_triangle(self, pt, v1, v2, v3):

        def sign(p1, p2, p3):

            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])



        d1 = sign(pt, v1, v2)

        d2 = sign(pt, v2, v3)

        d3 = sign(pt, v3, v1)



        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)

        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)



        return not (has_neg and has_pos)



    def is_near_obstacle(self, y, x):

        if not hasattr(self, 'obstacle'):

            return False


            # 장애물 주변 영역 검사
        for dy in range(-self.diffraction_range, self.diffraction_range + 1):

            for dx in range(-self.diffraction_range, self.diffraction_range + 1):

                ny, nx = y + dy, x + dx

                if (0 <= ny < self.height and 0 <= nx < self.width and

                        self.obstacle_mask[ny, nx]):

                    return True

        return False



    def update(self, frame):

        # 기본 파동 방정식 계산

        for y in range(1, self.height - 1):

            for x in range(1, self.width - 1):

                if not np.isnan(self.u[y, x]):

                    diffraction_factor = 1.0 + self.diffraction_strength if self.is_near_obstacle(y, x) else 1.0



                    self.u_next[y, x] = (

                                                2 * self.u[y, x]

                                                - self.u_prev[y, x]

                                                + (self.c * self.dt / self.dx) ** 2 * (

                                                        (0 if np.isnan(self.u[y, x + 1]) else self.u[y, x + 1])

                                                        + (0 if np.isnan(self.u[y, x - 1]) else self.u[y, x - 1])

                                                        + (0 if np.isnan(self.u[y + 1, x]) else self.u[y + 1, x])

                                                        + (0 if np.isnan(self.u[y - 1, x]) else self.u[y - 1, x])

                                                        - 4 * self.u[y, x]

                                                ) * diffraction_factor

                                        ) * self.damping



        # 경계 조건

        self.u_next[0, :] = self.u_next[1, :]

        self.u_next[-1, :] = self.u_next[-2, :]



        # 파동 생성

        amplitude = 1.0

        for x, y in self.wave_sources:

            self.u_next[y, x] = amplitude * np.sin(self.frequency * frame * self.dt * 2 * np.pi)



        # 장애물 마스크 적용

        self.u_next[self.obstacle_mask] = np.nan



        # 센서 데이터 수집 및 기록

        sensor_values = [self.u[y, x-1] for x, y in self.sensors]

        self.sensor_data.append(sensor_values)

        if len(self.sensor_data) > 100:

            self.sensor_data.pop(0)



        # 센서 기록 시작 조건 확인

        if any(abs(val) > 0.1 for val in sensor_values):  # 임계값을 넘는 센서가 있으면

            if self.recording_start_frame is None:

                self.recording_start_frame = frame

                self.recorded_sensor_data = []



        # 센서 데이터 기록

        if self.recording_start_frame is not None:

            if len(self.recorded_sensor_data) < self.recording_frames:

                self.recorded_sensor_data.append(sensor_values)



            # 등고선 그래프 업데이트

            if len(self.recorded_sensor_data) == self.recording_frames:

                self.update_contour()



        # 시각화 업데이트

        self.img.set_array(self.u_next)



        sensor_data_array = np.array(self.sensor_data)

        for i, line in enumerate(self.lines):

            line.set_data(range(len(sensor_data_array)), sensor_data_array[:, i])



        # 상태 업데이트

        self.u_prev = self.u.copy()

        self.u = self.u_next.copy()



        return [self.img] + self.lines



    def update_contour(self):

        if self.contour is not None:

            for coll in self.contour.collections:

                coll.remove()



        data = np.array(self.recorded_sensor_data)

        X, Y = np.meshgrid(range(1, 17), range(len(data)))

        self.contour = self.ax3.contourf(X, Y, data, levels=20, cmap='coolwarm')

        self.ax3.set_title('Sensor Data Contour (5 seconds)')

        self.fig.colorbar(self.contour, ax=self.ax3, label='Amplitude')



    def run_animation(self, frames=500, interval=50):

        self.anim = FuncAnimation(

            self.fig, self.update, frames=frames,

            interval=interval, blit=False

        )

        plt.show()





# 실험 실행

def run_experiments():

 

    # 1. 정육면체

    sim1 = WaveSimulation(width=100, height=100)

    sim1.setup_visualization("Wave Simulation - Cube Obstacle")

    sim1.add_obstacle("cube", (50, 50), size=60)

    sim1.run_animation(frames=500, interval=30)



    # 2. 직육면체

    sim2 = WaveSimulation(width=100, height=100)

    sim2.setup_visualization("Wave Simulation - Cuboid Obstacle")

    sim2.add_obstacle("cuboid", (50, 50), width=50, height=25)

    sim2.run_animation(frames=500, interval=30)



    # 3. 삼각기둥

    sim3 = WaveSimulation(width=100, height=100)

    sim3.setup_visualization("Wave Simulation - Triangular Prism Obstacle")

    sim3.add_obstacle("triangular_prism", (50, 50), size=50)

    sim3.run_animation(frames=500, interval=30)



    # 4. 원기둥

    sim4 = WaveSimulation(width=100, height=100)

    sim4.setup_visualization("Wave Simulation - Cylinder Obstacle")

    sim4.add_obstacle("cylinder", (50, 50), size=50)

    sim4.run_animation(frames=500, interval=30)



if __name__ == "__main__":

    run_experiments()
