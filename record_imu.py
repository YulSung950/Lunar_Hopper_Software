#!/usr/bin/env python3
"""
imu_live.py - live view AND recording for imu_readings.ino (Lunar Hopper)

Shows four live graphs while it records:
  1. Acceleration x, y, z            (m/s^2, raw from the sensor)
  2. Gyro rate x, y, z               (deg/s, raw from the sensor)
  3. Orientation: roll, pitch, yaw   (deg) from adding up the gyro (solid lines),
     plus roll and pitch worked out from gravity (dashed lines) for comparison
  4. Relative position x, y, z       (m) from adding up acceleration twice

The CSV it saves is the same as imu_log_to_github.py's (raw readings + pc_time, same
folder), so you can push it later with:  python3 imu_log_to_github.py --push-only
The orientation and position are NOT saved: they can be recalculated from the raw data.

Setup (once):  pip3 install pyserial numpy matplotlib
Run:           python3 imu_live.py --name tilt_test
               python3 imu_live.py --name still_long --seconds 10800
Keys (click the graph window first):
  b   re-measure gyro bias (keep the IMU still for 2 seconds)
  z   zero: orientation back to the current tilt, position and speed back to 0
  q   stop and save (closing the window does the same)

Close the Arduino IDE's Serial Monitor / Plotter first: only one program can use the port.

WHAT TO EXPECT
  - Orientation from the gyro is good for seconds to minutes, then slowly drifts.
    Watch the solid lines move away from the dashed ones: that gap is gyro drift.
  - Position from acceleration drifts away within SECONDS, even when the IMU sits still.
    A tiny error is added up twice: 0.05 m/s^2 of error is ~2.5 m after 10 s.
    Treat it as a demonstration of why the Kalman filter is needed, not as a measurement.
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("pyserial is not installed. Run:  pip3 install pyserial")

LOG_FOLDER = os.path.expanduser("~/Downloads/imu_readings")   # same folder as imu_log_to_github.py
G = 9.80665              # standard gravity, m/s^2
WINDOW_S = 20            # seconds of history shown on the graphs
BIAS_S = 2.0             # seconds of stillness used to measure gyro bias
REDRAW_S = 0.1           # redraw the graphs 10 times a second
NEEDED = ["time_s", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
          "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]


# ----------------------------------------------------------------------------- math
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_from_rotvec(r):
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.array([1.0, r[0] / 2, r[1] / 2, r[2] / 2])
    axis = r / angle
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_rotate(q, v):
    """Rotate vector v from the sensor's frame into the world frame."""
    qv = np.array([0.0, *v])
    qc = q * np.array([1, -1, -1, -1])
    return quat_mul(quat_mul(q, qv), qc)[1:]


def quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def euler_from_quat(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x))))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def tilt_from_accel(a):
    """Roll and pitch (deg) from the direction of gravity. Only valid while not accelerating."""
    ax, ay, az = a
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return math.degrees(roll), math.degrees(pitch)


class Integrator:
    """Adds up gyro rate into orientation, and acceleration into speed and position."""

    def __init__(self):
        self.bias = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.last_t = None
        self.start_bias()

    def start_bias(self):
        self.calibrating = True
        self.cal_start = None
        self.cal_g, self.cal_a = [], []

    def zero(self, a_now):
        roll, pitch = tilt_from_accel(a_now)
        self.q = quat_from_euler(math.radians(roll), math.radians(pitch), 0.0)
        self.v = np.zeros(3)
        self.p = np.zeros(3)

    def update(self, t, a, g_dps):
        a = np.asarray(a, dtype=float)
        g = np.radians(np.asarray(g_dps, dtype=float))
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        if dt <= 0 or dt > 0.5:          # first sample, restart, or a big gap: don't integrate
            dt = 0.0

        if self.calibrating:
            if self.cal_start is None:
                self.cal_start = t
            self.cal_g.append(g)
            self.cal_a.append(a)
            if t - self.cal_start >= BIAS_S:
                self.bias = np.mean(self.cal_g, axis=0)
                self.zero(np.mean(self.cal_a, axis=0))
                self.calibrating = False
            return

        self.q = quat_mul(self.q, quat_from_rotvec((g - self.bias) * dt))
        self.q /= np.linalg.norm(self.q)
        a_world = quat_rotate(self.q, a) - np.array([0.0, 0.0, G])   # remove gravity
        self.v += a_world * dt
        self.p += self.v * dt


# ----------------------------------------------------------------------------- serial
def find_arduino_port():
    hints = ("usbserial", "usbmodem", "wchusbserial", "arduino", "ch340", "ch341", "ftdi", "ttyusb", "ttyacm")
    for p in serial.tools.list_ports.comports():
        text = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(h in text for h in hints) and "bluetooth" not in text:
            return p.device
    return None


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Live IMU graphs + CSV recording.")
    ap.add_argument("--port", help="serial port (found automatically if left out)")
    ap.add_argument("--baud", type=int, default=115200, help="must match Serial.begin() (default 115200)")
    ap.add_argument("--name", default="live", help="label for this run, used in the file name")
    ap.add_argument("--folder", default=LOG_FOLDER)
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until you close the window)")
    ap.add_argument("--snapshot", help="save a picture of the graphs to this .png file when done")
    a = ap.parse_args()

    port = a.port or find_arduino_port()
    if not port:
        sys.exit("No Arduino found. Plug it in, close the Serial Monitor, or pass --port.")
    try:
        ser = serial.Serial(port, a.baud, timeout=0)
    except serial.SerialException as e:
        sys.exit(f"Could not open {port}: {e}\nIs the Arduino IDE's Serial Monitor still open?")

    os.makedirs(a.folder, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in a.name)
    csv_path = os.path.join(a.folder, datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{safe_name}.csv")
    print(f"Connected to {port}. The Arduino restarts now: keep the IMU STILL for the first 2 seconds.")

    # ---- graphs
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig.canvas.manager.set_window_title(f"IMU live - {safe_name}")
    colors = ["tab:red", "tab:green", "tab:blue"]
    lines = {}
    for key, ax, title, unit in [("acc", axs[0, 0], "Acceleration", "m/s²"),
                                 ("gyr", axs[0, 1], "Gyro rate", "deg/s"),
                                 ("ori", axs[1, 0], "Orientation (solid: gyro, dashed: gravity)", "deg"),
                                 ("pos", axs[1, 1], "Relative position (drifts within seconds)", "m")]:
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        names = ["roll", "pitch", "yaw"] if key == "ori" else ["x", "y", "z"]
        lines[key] = [ax.plot([], [], color=c, lw=1.2, label=n)[0] for c, n in zip(colors, names)]
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    lines["tilt"] = [axs[1, 0].plot([], [], color=c, lw=1, ls="--")[0] for c in colors[:2]]
    for ax in axs[1]:
        ax.set_xlabel("time (s)")
    status = fig.suptitle("Waiting for the Arduino ...", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    hist = {k: deque(maxlen=int(WINDOW_S * 100) + 50) for k in
            ["t", "ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw", "troll", "tpitch", "px", "py", "pz"]}

    integ = Integrator()
    state = {"running": True, "last_a": None}

    def on_key(event):
        if event.key == "b":
            integ.start_bias()
            print("Re-measuring gyro bias: keep the IMU still for 2 s.")
        elif event.key == "z" and state["last_a"] is not None:
            integ.zero(state["last_a"])
            print("Zeroed orientation and position.")
        elif event.key == "q":
            state["running"] = False

    def on_close(_event):
        state["running"] = False

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    # ---- read, record, draw
    header, writer, f = None, None, None
    rows, bad = 0, 0
    buf = b""
    t_start = time.time()
    next_draw = t_start

    try:
        while state["running"]:
            if a.seconds and time.time() - t_start >= a.seconds:
                break
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("#"):
                    print("  " + line)
                    continue
                parts = [p.strip() for p in line.split(",")]
                if parts[0] == "time_s":
                    header = parts
                    continue
                if header is None or not all(is_number(p) for p in parts):
                    continue
                if len(parts) != len(header):
                    bad += 1
                    continue
                if writer is None:
                    missing = [c for c in NEEDED if c not in header]
                    if missing:
                        sys.exit(f"The sketch's header is missing columns: {missing}")
                    f = open(csv_path, "w", newline="")
                    writer = csv.writer(f)
                    writer.writerow(["pc_time"] + header)
                    idx = [header.index(c) for c in NEEDED]
                    print(f"Saving to {csv_path}")
                writer.writerow([datetime.now().isoformat(timespec="milliseconds")] + parts)
                rows += 1
                if rows % 100 == 0:
                    f.flush()

                v = [float(parts[i]) for i in idx]
                t, acc, gyr = v[0], v[1:4], v[4:7]
                state["last_a"] = acc
                if integ.last_t is not None and t < integ.last_t:     # Arduino restarted
                    for d in hist.values():
                        d.clear()
                    integ.__init__()
                    print("Arduino restarted: keep it still for 2 s while the gyro bias is re-measured.")
                integ.update(t, acc, gyr)
                r, p, y = euler_from_quat(integ.q)
                tr, tp = tilt_from_accel(acc)
                for k, val in zip(hist.keys(), [t, *acc, *gyr, r, p, y, tr, tp, *integ.p]):
                    hist[k].append(val)

            now = time.time()
            if now >= next_draw and hist["t"]:
                next_draw = now + REDRAW_S
                tt = np.array(hist["t"])
                for ln, k in zip(lines["acc"], ["ax", "ay", "az"]):
                    ln.set_data(tt, hist[k])
                for ln, k in zip(lines["gyr"], ["gx", "gy", "gz"]):
                    ln.set_data(tt, hist[k])
                for ln, k in zip(lines["ori"], ["roll", "pitch", "yaw"]):
                    ln.set_data(tt, hist[k])
                for ln, k in zip(lines["tilt"], ["troll", "tpitch"]):
                    ln.set_data(tt, hist[k])
                for ln, k in zip(lines["pos"], ["px", "py", "pz"]):
                    ln.set_data(tt, hist[k])
                for ax in axs.flat:
                    ax.relim()
                    ax.autoscale_view(scalex=False)
                    ax.set_xlim(max(0.0, tt[-1] - WINDOW_S), max(WINDOW_S, tt[-1]))
                if integ.calibrating:
                    status.set_text("Measuring gyro bias: keep the IMU STILL")
                else:
                    b = integ.bias * 180 / math.pi
                    status.set_text(f"t = {tt[-1]:.1f} s   rows {rows}   |   roll {hist['roll'][-1]:+.1f}°  "
                                    f"pitch {hist['pitch'][-1]:+.1f}°  yaw {hist['yaw'][-1]:+.1f}°   |   "
                                    f"gyro bias [{b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}] deg/s   "
                                    f"(keys: b = bias, z = zero, q = quit)")
                plt.pause(0.001)
            elif not chunk:
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    except serial.SerialException as e:
        print(f"\nConnection lost: {e}")
    finally:
        ser.close()
        if f:
            f.close()

    if a.snapshot:
        fig.savefig(a.snapshot, dpi=110)
        print(f"Saved a picture of the graphs to {a.snapshot}")
    if writer is None:
        sys.exit("No readings received. Check the baud rate and that imu_readings.ino is uploaded.")
    print(f"Saved {rows} rows to {csv_path}" + (f"  ({bad} cut-off lines skipped)" if bad else ""))
    if hist["t"]:
        print(f"Final orientation from gyro: roll {hist['roll'][-1]:+.1f}°, pitch {hist['pitch'][-1]:+.1f}°, "
              f"yaw {hist['yaw'][-1]:+.1f}°   |   from gravity: roll {hist['troll'][-1]:+.1f}°, "
              f"pitch {hist['tpitch'][-1]:+.1f}°")
        print(f"Final relative position (m): x {hist['px'][-1]:+.3f}, y {hist['py'][-1]:+.3f}, z {hist['pz'][-1]:+.3f}")
    print("Push it later with:  python3 imu_log_to_github.py --push-only")


if __name__ == "__main__":
    main()
