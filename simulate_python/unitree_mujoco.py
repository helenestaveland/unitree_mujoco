import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import sys
from pathlib import Path
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config

LIDAR_ROOT = Path(__file__).resolve().parents[2] / "MuJoCo-LiDAR"
sys.path.insert(0, str(LIDAR_ROOT))

from mujoco_lidar import MjLidarWrapper
from jt128_scan_test import generate_jt128
from examples.unitree_g1_jt128 import JT128Publisher
import rclpy


locker = threading.Lock()
_physics_stats = {
    "steps": 0,
    "last_step_wall": None,
    "max_period": 0.0,
    "max_lock_wait": 0.0,
}

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)

# LiDAR self-filter:
# Keep world/environment geoms in group 0, but move all robot geoms
# to group 2 so the JT128 (which ray-casts only against group 0)
# cannot see the G1 itself. This does not affect contact physics.
for geom_id in range(mj_model.ngeom):
    if mj_model.geom_bodyid[geom_id] != 0:
        mj_model.geom_group[geom_id] = 2

mj_data = mujoco.MjData(mj_model)

# Existing 128-channel JT128 scan pattern: 900 x 128 = 115200 rays.
rays_theta, rays_phi = generate_jt128(
    str(LIDAR_ROOT / "JT128_default_angle.csv")
)
rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)

geomgroup = np.zeros((mujoco.mjNGROUP,), dtype=np.ubyte)
geomgroup[0] = 1

lidar = MjLidarWrapper(
    mj_model,
    site_name="lidar",
    backend="warp",
    args={
        "device": "cuda:0",
        "bodyexclude": mj_model.body("torso_link").id,
        "geomgroup": geomgroup,
    },
)

# Separate MjData only for taking a snapshot of the RUNNING simulation.
# This is not a second simulation.
lidar_data = mujoco.MjData(mj_model)

rclpy.init()
ros_node = JT128Publisher()


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        acquired = time.perf_counter()
        previous = _physics_stats["last_step_wall"]
        if previous is not None:
            _physics_stats["max_period"] = max(
                _physics_stats["max_period"], acquired - previous
            )
        _physics_stats["last_step_wall"] = acquired
        _physics_stats["max_lock_wait"] = max(
            _physics_stats["max_lock_wait"], acquired - step_start
        )

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
            else:
                mj_data.xfrc_applied[band_attached_link, :3] = 0.0
        mujoco.mj_step(mj_model, mj_data)
        _physics_stats["steps"] += 1

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)



def LidarThread():
    global mj_data, mj_model

    while viewer.is_running():
        scan_start = time.perf_counter()

        # Take a quick snapshot of the current running simulation state.
        locker.acquire()
        lidar_data.qpos[:] = mj_data.qpos
        lidar_data.qvel[:] = mj_data.qvel
        lidar_data.time = mj_data.time
        locker.release()

        # Update geometry/site positions for that snapshot.
        t0 = time.perf_counter()
        mujoco.mj_forward(mj_model, lidar_data)
        t1 = time.perf_counter()

        # Existing JT128 LiDAR implementation.
        lidar.trace_rays(lidar_data, rays_theta, rays_phi)
        points = lidar.get_hit_points()
        t2 = time.perf_counter()

        valid = np.einsum("ij,ij->i", points, points) > 1e-12

        # Use the existing Hesai-compatible ROS2 publisher.
        ros_node.publish_points(points, lidar_data.time)
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        t3 = time.perf_counter()

        print(
            f"JT128: {valid.sum()}/{len(points)} | "
            f"forward {(t1-t0)*1000:.1f} ms | "
            f"raytrace {(t2-t1)*1000:.1f} ms | "
            f"publish {(t3-t2)*1000:.1f} ms"
        )

        # Approximately 10 Hz.
        remaining = 0.1 - (time.perf_counter() - scan_start)
        if remaining > 0:
            time.sleep(remaining)


def ImuThread():
    imu_dt = 0.1
    last_sim_time = -1.0
    snapshot = mujoco.MjData(mj_model)

    while viewer.is_running():
        imu_start = time.perf_counter()
        publish_sample = False

        # Copy the fields needed by the publishers.
        with locker:
            current_sim_time = float(mj_data.time)
            if current_sim_time > last_sim_time:
                snapshot.time = current_sim_time
                snapshot.sensordata[:] = mj_data.sensordata
                snapshot.site_xpos[:] = mj_data.site_xpos
                snapshot.site_xmat[:] = mj_data.site_xmat
                last_sim_time = current_sim_time
                publish_sample = True

        # Publish after releasing the physics lock.
        if publish_sample:
            ros_node.publish_imu(mj_model, snapshot)
            ros_node.publish_ground_truth(mj_model, snapshot)

        remaining = imu_dt - (time.perf_counter() - imu_start)
        if remaining > 0:
            time.sleep(remaining)

def PhysicsViewerThread():
    pause_file = Path("/tmp/g1_pause_viewer_sync")
    while viewer.is_running():
        if not pause_file.exists():
            with locker:
                viewer.sync()
        time.sleep(config.VIEWER_DT)

def TimingThread():
    previous = None
    while viewer.is_running():
        time.sleep(1.0)
        with locker:
            wall = time.perf_counter()
            sim = float(mj_data.time)
            steps = _physics_stats["steps"]
            worst = _physics_stats["max_period"]
            wait = _physics_stats["max_lock_wait"]
            _physics_stats["max_period"] = 0.0
            _physics_stats["max_lock_wait"] = 0.0

        if previous is not None:
            old_wall, old_sim, old_steps = previous
            elapsed = wall - old_wall
            print(
                f"[SIM TIMING] RTF={(sim-old_sim)/elapsed:.2f} "
                f"steps/s={(steps-old_steps)/elapsed:.0f} "
                f"worst_interval_ms={worst*1000:.1f} "
                f"lock_wait_ms={wait*1000:.1f}",
                flush=True,
            )
        previous = (wall, sim, steps)

if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    lidar_thread = Thread(target=LidarThread)
    imu_thread = Thread(target=ImuThread)

    viewer_thread.start()
    sim_thread.start()
    lidar_thread.start()  # Warp GPU JT128
    imu_thread.start()
    Thread(target=TimingThread, daemon=True).start()
