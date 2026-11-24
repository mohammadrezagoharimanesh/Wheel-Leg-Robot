import tkinter as tk
import numpy as np
import math
import time
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

D = 71.70
L_PROX = 88.5
L_DIST = 147.0

LEG_GAP = 149.0

WHEEL_D = 75.0
WHEEL_R = 37.5
WHEEL_W = 34.0

BODY_X = 210.0
BODY_Y = 150.0
BODY_Z = 55.0

L_MIN = 90.0
L_MAX = (L_PROX + L_DIST) * 0.92
PHI_MAX = 30.0

FRAME_MS = 33

HIP_X = {"L": -LEG_GAP / 2, "R": LEG_GAP / 2}

BG = "#050505"
PANEL = "#0b0b0b"
TXT = "#e6e6e6"
SUB = "#6a6a6a"
ACCENT1 = "#00e5ff"
ACCENT2 = "#39ff88"
ACCENT3 = "#ff2e6d"
ACCENT4 = "#ffb020"

try:
    arduino = serial.Serial("COM5", 9600, timeout=1)
    time.sleep(2)
    print("Arduino Connected")
except Exception:
    arduino = None
    print("Simulation mode")


class PID:
    def __init__(self, kp, ki, kd, limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral = 0
        self.previous = 0

    def update(self, error, dt):
        if dt <= 0:
            dt = 0.01
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.limit * 4, self.limit * 4)
        derivative = (error - self.previous) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -self.limit, self.limit)
        self.previous = error
        return output

    def reset(self):
        self.integral = 0
        self.previous = 0


def virtual_to_yz(L, phi):
    p = math.radians(phi)
    y = L * math.sin(p)
    z = -L * math.cos(p)
    return y, z


def ik(y, z):
    yA, zA = D / 2, 0
    yE, zE = -D / 2, 0

    dA = math.hypot(y - yA, z - zA)
    if not (abs(L_PROX - L_DIST) <= dA <= L_PROX + L_DIST):
        return None
    cA = np.clip((L_PROX ** 2 + dA ** 2 - L_DIST ** 2) / (2 * L_PROX * dA), -1, 1)
    thetaA = math.atan2(z - zA, y - yA) + math.acos(cA)
    By = yA + L_PROX * math.cos(thetaA)
    Bz = zA + L_PROX * math.sin(thetaA)

    dE = math.hypot(y - yE, z - zE)
    if not (abs(L_PROX - L_DIST) <= dE <= L_PROX + L_DIST):
        return None
    cE = np.clip((L_PROX ** 2 + dE ** 2 - L_DIST ** 2) / (2 * L_PROX * dE), -1, 1)
    thetaE = math.atan2(z - zE, y - yE) - math.acos(cE)
    Dy = yE + L_PROX * math.cos(thetaE)
    Dz = zE + L_PROX * math.sin(thetaE)

    return (
        math.degrees(thetaA), math.degrees(thetaE),
        (yA, zA), (yE, zE), (By, Bz), (Dy, Dz)
    )


def rotation_matrix(roll, pitch, yaw):
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Robot3D:
    def __init__(self, root):
        self.root = root
        root.title("Wheel-Leg Robot Pro - Full PID Acrobatics")
        root.geometry("1500x860")
        root.configure(bg=BG)

        self.L_left = tk.DoubleVar(value=160)
        self.L_right = tk.DoubleVar(value=160)
        self.phi_left = tk.DoubleVar(value=0)
        self.phi_right = tk.DoubleVar(value=0)
        self.speed_mult = tk.DoubleVar(value=1.8)

        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.body_z = 0
        self.wheel_angle = 0
        self.drive_velocity = 0

        self.last_time = time.time()
        self.mode = "IDLE"
        self.sequence = []
        self.sequence_index = 0
        self.animating = False
        self.keys = set()

        self.pid_roll = PID(2.1, 0.05, 0.42, 18)
        self.pid_pitch = PID(2.2, 0.05, 0.48, 18)
        self.pid_height = PID(1.4, 0.02, 0.30, 25)
        self.pid_drive = PID(3.0, 0.10, 0.20, 40)

        self.target_roll = 0
        self.target_pitch = 0
        self.target_height = 160
        self.target_velocity = 0

        self.build_ui()
        self.update()

        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)
        self.root.after(FRAME_MS, self.control_loop)

    def build_ui(self):
        panel = tk.Frame(self.root, bg=PANEL, width=350)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(panel, text="WHEEL-LEG ROBOT", bg=PANEL, fg=ACCENT1,
                 font=("Consolas", 18, "bold")).pack(pady=(16, 2))
        tk.Label(panel, text="FULL PID / ACROBATICS ENGINE", bg=PANEL, fg=SUB,
                 font=("Consolas", 9)).pack(pady=(0, 14))

        self.slider(panel, "Left leg length", self.L_left, L_MIN, L_MAX, ACCENT1)
        self.slider(panel, "Left leg angle", self.phi_left, -PHI_MAX, PHI_MAX, ACCENT1)
        self.slider(panel, "Right leg length", self.L_right, L_MIN, L_MAX, ACCENT2)
        self.slider(panel, "Right leg angle", self.phi_right, -PHI_MAX, PHI_MAX, ACCENT2)
        self.slider(panel, "Motion speed multiplier", self.speed_mult, 0.5, 4.0, ACCENT4)

        sep = tk.Frame(panel, bg="#1a1a1a", height=2)
        sep.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(panel, text="Basic moves", bg=PANEL, fg=SUB, font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=12)
        self.button(panel, "⚡ Jump", self.jump, ACCENT4)
        self.button(panel, "⚡⚡ Double Jump", self.double_jump, ACCENT4)
        self.button(panel, "⇆ Side Move", self.side_move, ACCENT1)
        self.button(panel, "🚗 PID Drive", self.drive_pid, ACCENT2)

        tk.Label(panel, text="Acrobatics", bg=PANEL, fg=SUB, font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=12, pady=(10, 0))
        self.button(panel, "↻ Front Flip", self.front_flip, ACCENT3)
        self.button(panel, "↺ Back Flip", self.back_flip, ACCENT3)
        self.button(panel, "↻↻ Double Front Flip", self.double_front_flip, ACCENT3)
        self.button(panel, "⤾ Cartwheel", self.cartwheel, ACCENT3)
        self.button(panel, "⟳ 360 Spin", self.spin, ACCENT1)
        self.button(panel, "⟳⟳ 720 Spin", self.double_spin, ACCENT1)
        self.button(panel, "🛞 Wheelie", self.wheelie, ACCENT2)
        self.button(panel, "🕺 Moonwalk", self.moonwalk, ACCENT2)
        self.button(panel, "♻ Final Combo", self.combo, "#ffffff")

        tk.Label(panel, text="Stability", bg=PANEL, fg=SUB, font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=12, pady=(10, 0))
        self.button(panel, "⚖ PID Balance", self.pid_balance, ACCENT1)
        self.button(panel, "■ Stop", self.stop, "#888888")
        self.button(panel, "↺ Reset", self.reset, "#888888")

        self.status = tk.Label(panel, text="", bg=PANEL, fg=TXT, justify=tk.LEFT, font=("Consolas", 9))
        self.status.pack(pady=14, padx=12, anchor=tk.W)

        for line in ["W/S : Forward / Backward", "A/D : Turn left / right", "Q/E : Rotate body", "SPACE : Jump"]:
            tk.Label(panel, text=line, bg=PANEL, fg=SUB, font=("Consolas", 8)).pack(anchor=tk.W, padx=12)

        plot_frame = tk.Frame(self.root, bg=BG)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(9.5, 7.5), facecolor=BG)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def slider(self, parent, text, variable, minimum, maximum, color):
        tk.Label(parent, text=text, bg=PANEL, fg=TXT, font=("Consolas", 9)).pack(anchor=tk.W, padx=12)
        tk.Scale(parent, from_=minimum, to=maximum, resolution=0.5, orient=tk.HORIZONTAL,
                 variable=variable, length=300, bg=PANEL, fg=color, troughcolor="#1a1a1a",
                 highlightthickness=0, activebackground=color,
                 command=lambda _: self.update()).pack(padx=8)

    def button(self, parent, text, command, color):
        tk.Button(parent, text=text, command=command, width=30, bg="#141414", fg=color,
                  activebackground="#242424", activeforeground=color, relief=tk.FLAT,
                  font=("Consolas", 9)).pack(pady=2)

    def transform(self, p):
        R = rotation_matrix(self.roll, self.pitch, self.yaw)
        return R @ np.array(p)

    def draw_body(self):
        x0, x1 = -BODY_X / 2, BODY_X / 2
        y0, y1 = -BODY_Y / 2, BODY_Y / 2
        z0, z1 = -BODY_Z, 0
        points = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
        points = [self.transform(p) for p in points]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            self.ax.plot3D([points[a][0], points[b][0]], [points[a][1], points[b][1]],
                            [points[a][2], points[b][2]], color=ACCENT1, linewidth=2.2, alpha=0.9)

    def draw_wheel(self, p):
        theta = np.linspace(0, 2 * np.pi, 50)
        for xx in [-WHEEL_W / 2, WHEEL_W / 2]:
            pts = np.array([self.transform([p[0] + xx, p[1] + WHEEL_R * math.cos(t),
                                             p[2] + WHEEL_R * math.sin(t)]) for t in theta])
            self.ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], color=ACCENT4, linewidth=1.8, alpha=0.85)

        for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            a_angle = angle + self.wheel_angle
            a = self.transform([p[0] - WHEEL_W / 2, p[1] + WHEEL_R * math.cos(a_angle),
                                 p[2] + WHEEL_R * math.sin(a_angle)])
            b = self.transform([p[0] + WHEEL_W / 2, p[1] + WHEEL_R * math.cos(a_angle),
                                 p[2] + WHEEL_R * math.sin(a_angle)])
            self.ax.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#555555", linewidth=1)

    def draw_leg(self, x, L, phi):
        y, z = virtual_to_yz(L, phi)
        result = ik(y, z)
        if result is None:
            return None
        p1, p4, A, E, B, Dp = result
        pts = [[x, A[0], A[1]], [x, B[0], B[1]], [x, y, z], [x, Dp[0], Dp[1]], [x, E[0], E[1]]]
        pts = [self.transform(p) for p in pts]
        A1, B1, C1, D1, E1 = pts

        self.ax.plot3D([A1[0], B1[0]], [A1[1], B1[1]], [A1[2], B1[2]], color=ACCENT3, linewidth=6)
        self.ax.plot3D([B1[0], C1[0]], [B1[1], C1[1]], [B1[2], C1[2]], color="#ff7a9c", linewidth=4)
        self.ax.plot3D([E1[0], D1[0]], [E1[1], D1[1]], [E1[2], D1[2]], color=ACCENT2, linewidth=6)
        self.ax.plot3D([D1[0], C1[0]], [D1[1], C1[1]], [D1[2], C1[2]], color="#8dffb8", linewidth=4)

        self.ax.scatter([A1[0], E1[0]], [A1[1], E1[1]], [A1[2], E1[2]], color=ACCENT4, s=65)
        self.ax.scatter([B1[0], D1[0]], [B1[1], D1[1]], [B1[2], D1[2]], color="#ffffff", s=40)
        self.ax.scatter([C1[0]], [C1[1]], [C1[2]], color="#ffffff", s=75)

        self.draw_wheel([x, y, z])
        return p1, p4

    def update(self):
        self.ax.clear()
        self.draw_body()
        left = self.draw_leg(HIP_X["L"], self.L_left.get(), self.phi_left.get())
        right = self.draw_leg(HIP_X["R"], self.L_right.get(), self.phi_right.get())

        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-240, 240)
        self.ax.set_zlim(-300, 90)
        self.ax.set_facecolor(BG)
        self.fig.patch.set_facecolor(BG)
        self.ax.tick_params(colors="#444444")

        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis.pane.set_facecolor(BG)
            axis.pane.set_edgecolor("#1a1a1a")
            axis.label.set_color("#555555")

        self.ax.set_title("WHEEL-LEG ROBOT // PID CORE", color=ACCENT1, fontfamily="monospace")
        self.ax.grid(True, alpha=0.08)
        self.ax.set_box_aspect((400, 480, 390))

        self.status.config(text=(
            f"MODE     : {self.mode}\n"
            f"ROLL     : {self.roll:7.2f}\n"
            f"PITCH    : {self.pitch:7.2f}\n"
            f"YAW      : {self.yaw:7.2f}\n"
            f"VEL      : {self.drive_velocity:7.2f}\n"
            f"PID-ROLL : {self.pid_roll.previous:7.2f}\n"
            f"PID-PITCH: {self.pid_pitch.previous:7.2f}\n"
            f"PID-DRV  : {self.pid_drive.previous:7.2f}"
        ))

        self.canvas.draw_idle()
        self.send_arduino(left, right)

    def send_arduino(self, left, right):
        if arduino is None or left is None or right is None:
            return
        p1l, p4l = left
        p1r, p4r = right
        values = [
            int(np.clip(90 + p1l, 0, 180)),
            int(np.clip(90 + p4l, 0, 180)),
            int(np.clip(90 + p1r, 0, 180)),
            int(np.clip(90 + p4r, 0, 180)),
        ]
        try:
            arduino.write((",".join(map(str, values)) + "\n").encode())
        except Exception:
            pass

    def control_loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0.1:
            dt = 0.1

        sm = self.speed_mult.get()

        if "w" in self.keys:
            self.target_velocity = min(40, self.target_velocity + 2.0 * sm)
        elif "s" in self.keys:
            self.target_velocity = max(-40, self.target_velocity - 2.0 * sm)
        else:
            self.target_velocity *= 0.9

        if "a" in self.keys:
            self.yaw -= 1.4 * sm
        if "d" in self.keys:
            self.yaw += 1.4 * sm
        if "q" in self.keys:
            self.yaw -= 1.6 * sm
        if "e" in self.keys:
            self.yaw += 1.6 * sm

        drive_error = self.target_velocity - self.drive_velocity
        drive_correction = self.pid_drive.update(drive_error, dt)
        self.drive_velocity += drive_correction * dt * 4
        self.wheel_angle += math.radians(self.drive_velocity) * dt * 10

        if "w" in self.keys or "s" in self.keys:
            lean = np.clip(self.drive_velocity * 0.35, -PHI_MAX, PHI_MAX)
            self.phi_left.set(lean)
            self.phi_right.set(lean)

        if self.mode == "PID":
            roll_error = self.target_roll - self.roll
            pitch_error = self.target_pitch - self.pitch
            height_error = self.target_height - (self.L_left.get() + self.L_right.get()) / 2

            roll_correction = self.pid_roll.update(roll_error, dt)
            pitch_correction = self.pid_pitch.update(pitch_error, dt)
            height_correction = self.pid_height.update(height_error, dt)

            self.L_left.set(np.clip(self.L_left.get() - roll_correction * 0.3 + height_correction * 0.2, L_MIN, L_MAX))
            self.L_right.set(np.clip(self.L_right.get() + roll_correction * 0.3 + height_correction * 0.2, L_MIN, L_MAX))
            self.phi_left.set(np.clip(self.phi_left.get() + pitch_correction * 0.1, -PHI_MAX, PHI_MAX))
            self.phi_right.set(np.clip(self.phi_right.get() + pitch_correction * 0.1, -PHI_MAX, PHI_MAX))

            self.roll *= 0.9
            self.pitch *= 0.9

        if abs(self.yaw) > 360:
            self.yaw %= 360

        if self.animating:
            steps = max(1, int(round(sm)))
            for _ in range(steps):
                if self.sequence_index >= len(self.sequence):
                    self.animating = False
                    self.mode = "IDLE"
                    break
                self.sequence[self.sequence_index]()
                self.sequence_index += 1

        self.update()
        self.root.after(FRAME_MS, self.control_loop)

    def key_down(self, event):
        key = event.keysym.lower()
        self.keys.add(key)
        if key == "space":
            self.jump()

    def key_up(self, event):
        key = event.keysym.lower()
        if key in self.keys:
            self.keys.remove(key)

    def stop(self):
        self.animating = False
        self.sequence = []
        self.mode = "STOP"
        self.keys.clear()
        self.target_velocity = 0

    def reset(self):
        self.stop()
        self.L_left.set(160)
        self.L_right.set(160)
        self.phi_left.set(0)
        self.phi_right.set(0)
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.body_z = 0
        self.drive_velocity = 0
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_height.reset()
        self.pid_drive.reset()
        self.mode = "IDLE"
        self.update()

    def run(self, frames):
        self.sequence = frames
        self.sequence_index = 0
        self.animating = True

    def jump(self):
        values = np.concatenate([np.linspace(160, 100, 14), np.linspace(100, L_MAX, 10), np.linspace(L_MAX, 160, 16)])
        frames = [lambda v=v: (self.L_left.set(v), self.L_right.set(v)) for v in values]
        self.mode = "JUMP"
        self.run(frames)

    def double_jump(self):
        cycle = np.concatenate([np.linspace(160, 100, 12), np.linspace(100, L_MAX, 8), np.linspace(L_MAX, 160, 12)])
        values = np.concatenate([cycle, cycle])
        frames = [lambda v=v: (self.L_left.set(v), self.L_right.set(v)) for v in values]
        self.mode = "DOUBLE JUMP"
        self.run(frames)

    def front_flip(self):
        frames = [lambda q=t: self.flip_frame(q, 1) for t in np.linspace(0, 1, 55)]
        self.mode = "FRONT FLIP"
        self.run(frames)

    def back_flip(self):
        frames = [lambda q=t: self.flip_frame(q, -1) for t in np.linspace(0, 1, 55)]
        self.mode = "BACK FLIP"
        self.run(frames)

    def double_front_flip(self):
        frames = [lambda q=t: self.flip_frame(q % 1.0, 1, turns=2) for t in np.linspace(0, 2, 100)]
        self.mode = "DOUBLE FRONT FLIP"
        self.run(frames)

    def flip_frame(self, t, direction, turns=1):
        self.pitch = direction * 360 * turns * math.sin(math.pi * t) / (2 if turns > 1 else 1) + direction * 180 * turns * t
        lift = 65 * math.sin(math.pi * t)
        self.L_left.set(np.clip(150 + lift, L_MIN, L_MAX))
        self.L_right.set(np.clip(150 + lift, L_MIN, L_MAX))
        self.body_z = lift

    def cartwheel(self):
        frames = [lambda q=t: self.cartwheel_frame(q) for t in np.linspace(0, 1, 65)]
        self.mode = "CARTWHEEL"
        self.run(frames)

    def cartwheel_frame(self, t):
        self.roll = 360 * t
        lift = 40 * math.sin(math.pi * t)
        self.L_left.set(np.clip(150 + lift, L_MIN, L_MAX))
        self.L_right.set(np.clip(150 + lift, L_MIN, L_MAX))
        self.phi_left.set(20 * math.sin(2 * math.pi * t))
        self.phi_right.set(-20 * math.sin(2 * math.pi * t))

    def spin(self):
        frames = [lambda q=t: self.spin_frame(q) for t in np.linspace(0, 1, 55)]
        self.mode = "360 SPIN"
        self.run(frames)

    def double_spin(self):
        frames = [lambda q=t: self.spin_frame(q % 1.0, turns=2) for t in np.linspace(0, 2, 100)]
        self.mode = "720 SPIN"
        self.run(frames)

    def spin_frame(self, t, turns=1):
        self.yaw = 360 * turns * t
        lift = 20 * math.sin(math.pi * t)
        self.L_left.set(155 + lift)
        self.L_right.set(155 + lift)

    def side_move(self):
        frames = [lambda q=t: self.side_frame(q) for t in np.linspace(0, 2 * math.pi, 70)]
        self.mode = "SIDE MOVE"
        self.run(frames)

    def side_frame(self, t):
        self.roll = 18 * math.sin(t)
        self.L_left.set(160 - 12 * math.sin(t))
        self.L_right.set(160 + 12 * math.sin(t))

    def wheelie(self):
        frames = [lambda q=t: self.wheelie_frame(q) for t in np.linspace(0, 1, 60)]
        self.mode = "WHEELIE"
        self.run(frames)

    def wheelie_frame(self, t):
        ramp = math.sin(math.pi * t)
        self.pitch = -35 * ramp
        self.L_left.set(np.clip(160 + 30 * ramp, L_MIN, L_MAX))
        self.L_right.set(np.clip(160 + 30 * ramp, L_MIN, L_MAX))
        self.wheel_angle += 0.4

    def moonwalk(self):
        frames = [lambda q=t: self.moonwalk_frame(q) for t in np.linspace(0, 2 * math.pi, 80)]
        self.mode = "MOONWALK"
        self.run(frames)

    def moonwalk_frame(self, t):
        self.pitch = -6 * math.sin(t)
        self.phi_left.set(-18 * math.sin(t))
        self.phi_right.set(-18 * math.sin(t + math.pi))
        self.wheel_angle -= 0.25

    def combo(self):
        frames = [lambda q=t: self.combo_frame(q) for t in np.linspace(0, 1, 130)]
        self.mode = "COMBO"
        self.run(frames)

    def combo_frame(self, t):
        self.yaw = 1080 * t
        self.roll = 25 * math.sin(6 * math.pi * t)
        self.pitch = 30 * math.sin(8 * math.pi * t)
        lift = 40 * abs(math.sin(2 * math.pi * t))
        self.L_left.set(np.clip(150 + lift, L_MIN, L_MAX))
        self.L_right.set(np.clip(150 + lift, L_MIN, L_MAX))
        self.wheel_angle += 0.5

    def drive_pid(self):
        self.mode = "DRIVE PID"
        self.target_velocity = 25
        self.pid_drive.reset()
        self.animating = False

    def pid_balance(self):
        self.mode = "PID"
        self.target_roll = 0
        self.target_pitch = 0
        self.target_height = 160
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_height.reset()
        self.roll = 14
        self.pitch = -9
        self.animating = False


if __name__ == "__main__":
    root = tk.Tk()
    app = Robot3D(root)
    root.mainloop()