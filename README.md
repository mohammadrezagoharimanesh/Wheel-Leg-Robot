# Wheel-Leg Robot

A 3D simulator and controller for a two-wheel robot with two parallel five-bar legs. It is built with Python, Tkinter and Matplotlib, and can drive real servos through an Arduino.

## Features

- Five-bar leg kinematics: virtual leg length and angle are converted to joint angles with inverse kinematics
- Live 3D view of the body, both legs and both wheels
- Sliders for left and right leg length and angle, plus a motion speed multiplier
- PID controllers for roll, pitch, height and drive velocity
- Built-in moves: jump, double jump, side move, PID drive
- Acrobatics: front flip, back flip, double front flip, cartwheel, 360 and 720 spins, wheelie, moonwalk and a final combo
- Optional Arduino output: four servo angles are sent over serial, and the program falls back to simulation mode if no board is found

## Requirements

- Python 3.9 or newer (Tkinter is included with most Python installs)
- `numpy`, `matplotlib`, `pyserial`

```bash
pip install numpy matplotlib pyserial
```

## Usage

```bash
python wheel_leg_robot_simulator.py
```

The program tries to open `COM5` at 9600 baud. If that fails it starts in simulation mode. To use another port, change the `serial.Serial(...)` line near the top of the script.

### Keyboard controls

| Key | Action |
| --- | --- |
| W / S | Forward / backward |
| A / D | Turn left / right |
| Q / E | Rotate body |
| Space | Jump |

## Serial protocol

Each update sends one line of four comma-separated servo angles in degrees (0 to 180):

```
left_proximal,left_distal,right_proximal,right_distal\n
```

## Geometry (mm)

| Parameter | Value |
| --- | --- |
| Hip spacing within a leg (`D`) | 71.7 |
| Proximal link | 88.5 |
| Distal link | 147.0 |
| Distance between legs | 149.0 |
| Wheel diameter | 75.0 |
