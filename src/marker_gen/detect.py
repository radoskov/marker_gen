#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aruco / Charuco diamond pose tool with RealSense or ROS input.

Changes:
- Open3D viewer: interactive (no per-frame view override), proper scale, optional grid, 3D labels.
- Marker mode: track poses of multiple markers; restrict to --marker-ids if given;
  maintain relative transforms to a main marker; fuse camera pose using all visible markers.
- Diamond mode: stable detection via detectCharucoDiamond; pose via solvePnP(IPPE_SQUARE fallback).
  Track multiple diamonds; restrict via --diamond-ids; fuse using learned relative transforms.
- Modular, concise processing; uniform docstrings.

Author: you
"""

import os
import sys
import math
import time
import yaml
import json
import queue
import argparse
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np

try:
    import cv2
    from cv2 import aruco
except Exception:
    sys.stderr.write("ERROR: OpenCV with aruco module required. Install: pip install opencv-contrib-python\n")
    raise

# ----------------------------
# Optional imports (loaded lazily)
# ----------------------------
_rs = None                 # pyrealsense2
_ros_mode = False
_ros2 = False
_ros = None                # rclpy or rospy
_ros_img = None            # sensor_msgs.msg.Image
_ros_caminfo = None        # sensor_msgs.msg.CameraInfo
_ros_tf2 = None            # tf2_ros
_ros_geom = None           # geometry_msgs.msg.TransformStamped
_ros_bridge = None         # cv_bridge (optional)


# ============================
# Utilities
# ============================

def now() -> float:
    return time.monotonic()


def rodrigues_to_quat(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rvec to quaternion [x,y,z,w]."""
    R, _ = cv2.Rodrigues(rvec)
    qw = math.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) * 0.5
    qx = (R[2,1] - R[1,2]) / (4.0 * qw + 1e-12)
    qy = (R[0,2] - R[2,0]) / (4.0 * qw + 1e-12)
    qz = (R[1,0] - R[0,1]) / (4.0 * qw + 1e-12)
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] to rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),       1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),       2*(yz+wx),     1-2*(xx+yy)]
    ], dtype=np.float64)
    return R


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation from q0 to q1 with factor t in [0,1]."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = (q0 + t*(q1-q0))
        return q / (np.linalg.norm(q) + 1e-12)
    theta_0 = math.acos(dot)
    sin_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / (sin_0 + 1e-12)
    s1 = math.sin(theta) / (sin_0 + 1e-12)
    return s0*q0 + s1*q1


# ============================
# Configuration
# ============================

@dataclass
class CameraConfig:
    source: str = "realsense"          # realsense | ros
    warmup_frames: int = 15
    serial: Optional[str] = None
    rgb_width: int = 1280
    rgb_height: int = 720
    rgb_fps: int = 30
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    dist: Optional[List[float]] = None


@dataclass
class ArucoConfig:
    dictionary: str = "DICT_6X6_250"
    detect_mode: str = "marker"  # marker | diamond
    marker_id: int = 0           # used in 'marker' mode (if you want to restrict to single ID, else detect any)
    marker_main_id: int = 0           # used in 'marker' mode (if you want to restrict to single ID, else detect any)
    marker_size_mm: float = 50.0 # physical size for pose
    diamond_square_mm: float = 40.0 # base square for diamond (used for annotation length and ID print)
    border_bits: int = 2
    corner_refine: bool = True
    # Optional explicit diamond definitions: list of [id0, id1, id2, id3]
    diamond_ids: Optional[List[List[int]]] = None


@dataclass
class RuntimeConfig:
    mode: str = "continuous"     # single | continuous
    throttle_sec: float = 0.1
    filter_enable: bool = True
    filter_strength: float = 0.5
    stale_tf_sec: float = 2.0
    warn_every_sec: float = 1.0


@dataclass
class OutputConfig:
    show_window: bool = True
    window_name: str = "Aruco Pose"
    draw_axes_length_m: float = 0.05
    ros_enable: bool = False
    ros_version: Optional[str] = None # auto | ros2 | ros1
    ros_in_image_topic: str = "/camera/color/image_raw"
    ros_in_camera_info: str = "/camera/color/camera_info"
    ros_out_image_topic: Optional[str] = "/aruco/annotated"
    tf_enable: bool = False
    tf_parent: str = "camera"
    tf_child: str = "marker"
    tf_broadcast_hz: float = 20.0
    draw_rejected: bool = False


@dataclass
class Config:
    camera: CameraConfig
    aruco: ArucoConfig
    runtime: RuntimeConfig
    output: OutputConfig


def ensure_dict(d):
    return {} if d is None else dict(d)


def load_yaml(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict, upd: Dict) -> Dict:
    """Deep merge dicts."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base


def build_config(args) -> Config:
    cfg = {"camera": {}, "aruco": {}, "runtime": {}, "output": {}}
    cfg = deep_update(cfg, ensure_dict(load_yaml(args.config)))

    # Camera
    if args.source:        cfg["camera"]["source"] = args.source
    if args.rs_serial:     cfg["camera"]["serial"] = args.rs_serial
    if args.rs_size:       cfg["camera"]["rgb_width"], cfg["camera"]["rgb_height"] = args.rs_size
    if args.rs_fps:        cfg["camera"]["rgb_fps"] = args.rs_fps
    if args.warmup_frames is not None: cfg["camera"]["warmup_frames"] = args.warmup_frames
    for k in ("fx","fy","cx","cy"):
        val = getattr(args, k)
        if val is not None: cfg["camera"][k] = val
    if args.dist is not None: cfg["camera"]["dist"] = args.dist

    # Aruco
    if args.dictionary:    cfg["aruco"]["dictionary"] = args.dictionary
    if args.detect_mode:   cfg["aruco"]["detect_mode"] = args.detect_mode
    if args.marker_id is not None:            cfg["aruco"]["marker_id"] = args.marker_id
    if args.marker_ids is not None:           cfg["aruco"]["marker_ids"] = args.marker_ids
    if args.marker_main_id is not None:       cfg["aruco"]["marker_main_id"] = args.marker_main_id
    if args.marker_mm:     cfg["aruco"]["marker_size_mm"] = args.marker_mm
    if args.diamond_mm:    cfg["aruco"]["diamond_square_mm"] = args.diamond_mm
    if args.diamond_ids is not None:
        if len(args.diamond_ids) % 4 != 0:
            raise ValueError("--diamond-ids must be a multiple of 4 IDs.")
        groups = [args.diamond_ids[i:i + 4] for i in range(0, len(args.diamond_ids), 4)]
        cfg["aruco"]["diamond_ids"] = groups
    if args.border_bits is not None:          cfg["aruco"]["border_bits"] = args.border_bits
    if args.no_corner_refine:                 cfg["aruco"]["corner_refine"] = False

    # Runtime
    if args.mode:          cfg["runtime"]["mode"] = args.mode
    if args.throttle is not None: cfg["runtime"]["throttle_sec"] = args.throttle
    if args.no_filter:     cfg["runtime"]["filter_enable"] = False
    if args.filter_strength is not None: cfg["runtime"]["filter_strength"] = args.filter_strength
    if args.stale_tf is not None: cfg["runtime"]["stale_tf_sec"] = args.stale_tf

    # Output
    if args.no_window:     cfg["output"]["show_window"] = False
    if args.window_name:   cfg["output"]["window_name"] = args.window_name
    if args.axis_m is not None: cfg["output"]["draw_axes_length_m"] = args.axis_m
    if args.ros:           cfg["output"]["ros_enable"] = True
    if args.ros_version:   cfg["output"]["ros_version"] = args.ros_version
    if args.ros_in:        cfg["output"]["ros_in_image_topic"] = args.ros_in
    if args.ros_info:      cfg["output"]["ros_in_camera_info"] = args.ros_info
    if args.ros_out:       cfg["output"]["ros_out_image_topic"] = args.ros_out
    if args.tf:            cfg["output"]["tf_enable"] = True
    if args.tf_parent:     cfg["output"]["tf_parent"] = args.tf_parent
    if args.tf_child:      cfg["output"]["tf_child"] = args.tf_child
    if args.tf_hz:         cfg["output"]["tf_broadcast_hz"] = args.tf_hz
    if args.draw_rejected: cfg["output"]["draw_rejected"] = True

    cam = CameraConfig(**deep_update(asdict(CameraConfig()), cfg.get("camera", {})))
    aru = ArucoConfig(**deep_update(asdict(ArucoConfig()), cfg.get("aruco", {})))
    # backward compat: marker_id acts as marker_main_id if main not given
    if aru.marker_main_id is None and aru.marker_id is not None:
        aru.marker_main_id = aru.marker_id
    run = RuntimeConfig(**deep_update(asdict(RuntimeConfig()), cfg.get("runtime", {})))
    out = OutputConfig(**deep_update(asdict(OutputConfig()), cfg.get("output", {})))
    return Config(camera=cam, aruco=aru, runtime=run, output=out)


# ============================
# Camera Sources
# ============================

class RealSenseSource:
    """RGB frames from Intel RealSense; color stream aligned."""
    def __init__(self, cam_cfg: CameraConfig):
        global _rs
        import importlib
        if _rs is None:
            _rs = importlib.import_module('pyrealsense2')
        self.cfg = cam_cfg
        self.pipe = _rs.pipeline()
        self.align = _rs.align(_rs.stream.color)
        self.profile = None

    def start(self):
        config = _rs.config()
        if self.cfg.serial:
            config.enable_device(self.cfg.serial)
        config.enable_stream(_rs.stream.color, self.cfg.rgb_width, self.cfg.rgb_height,
                             _rs.format.bgr8, self.cfg.rgb_fps)
        self.profile = self.pipe.start(config)
        for _ in range(int(self.cfg.warmup_frames)):
            self.pipe.wait_for_frames()

    def read(self) -> Tuple[np.ndarray, float]:
        frames = self.pipe.wait_for_frames()
        frames = self.align.process(frames)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame.")
        img = np.asanyarray(color.get_data())
        ts = frames.get_timestamp() * 1e-3
        return img, ts

    def intrinsics(self):
        if self.profile is None:
            return None, None
        color_stream = self.profile.get_stream(_rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.array(intr.coeffs, dtype=np.float64)
        return K, dist

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass

def init_ros(requested_version):
    global _ros_mode, _ros2, _ros, _ros_img, _ros_caminfo, _ros_bridge, ROS_NODE
    _ros_mode = True
    # try ROS2 then ROS1
    try:
        import rclpy as rclpy_mod
        _ros = rclpy_mod
        _ros2 = True
    except Exception:
        import rospy as rospy_mod
        _ros = rospy_mod
        _ros2 = False

    if _ros2:
        from sensor_msgs.msg import Image, CameraInfo
        _ros_img = Image
        _ros_caminfo = CameraInfo
        from rclpy.node import Node
        class _Node(Node): pass
        _ros.init(args=None)
        ROS_NODE = _Node("aruco_pose_node")
    else:
        from sensor_msgs.msg import Image, CameraInfo
        _ros_img = Image
        _ros_caminfo = CameraInfo
        _ros.init_node("aruco_pose_node", anonymous=True)
    try:
        import cv_bridge as cv_bridge_mod
        _ros_bridge = cv_bridge_mod.CvBridge()
    except Exception:
        _ros_bridge = None


class RosImageSource:
    """RGB frames and CameraInfo from ROS1/ROS2, with auto version detect."""
    def __init__(self, out_cfg: OutputConfig, cam_cfg: CameraConfig):
        self.out = out_cfg
        self.cam = cam_cfg
        self.latest = None  # (image ndarray, stamp_sec)
        self.K = None
        self.D = None
        self.lock = threading.Lock()
        self._running = False

    def start(self):
        if _ros2:
            self._start_ros2()
        else:
            self._start_ros1()

    def _start_ros2(self):
        self.node = ROS_NODE
        self.sub_img = self.node.create_subscription(_ros_img, self.out.ros_in_image_topic, self._cb_img_ros2, 10)
        self.sub_info = self.node.create_subscription(_ros_caminfo, self.out.ros_in_camera_info, self._cb_info_ros2, 10)
        self._running = True
        self.spin_thread = threading.Thread(target=_ros.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

    def _start_ros1(self):
        self.sub_img = _ros.Subscriber(self.out.ros_in_image_topic, _ros_img, self._cb_img_ros1, queue_size=1)
        self.sub_info = _ros.Subscriber(self.out.ros_in_camera_info, _ros_caminfo, self._cb_info_ros1, queue_size=1)
        self._running = True

    def _cb_info_ros2(self, msg):
        self._set_info(msg)

    def _cb_info_ros1(self, msg):
        self._set_info(msg)

    def _set_info(self, msg):
        K = np.array(msg.k, dtype=np.float64).reshape(3,3)
        D = np.array(msg.d, dtype=np.float64).reshape(-1)
        with self.lock:
            self.K, self.D = K, D

    def _cb_img_ros2(self, msg):
        img = self._img_from_msg(msg)
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)*1e-9
        with self.lock:
            self.latest = (img, stamp)

    def _cb_img_ros1(self, msg):
        img = self._img_from_msg(msg)
        stamp = msg.header.stamp.to_sec()
        with self.lock:
            self.latest = (img, stamp)

    def _img_from_msg(self, msg):
        if _ros_bridge is not None:
            try:
                return _ros_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                pass
        if msg.encoding not in ('bgr8', 'rgb8'):
            raise RuntimeError(f"Unsupported encoding: {msg.encoding}")
        dtype = np.uint8
        img = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        with self.lock:
            return None if self.latest is None else (self.latest[0].copy(), self.latest[1])

    def intrinsics(self):
        with self.lock:
            return (None if self.K is None else self.K.copy(),
                    None if self.D is None else self.D.copy())

    def stop(self):
        self._running = False
        if _ros2:
            try:
                self.node.destroy_node()
                _ros.shutdown()
            except Exception:
                pass
        else:
            try:
                _ros.signal_shutdown("done")
            except Exception:
                pass


# ============================
# ROS Publishers / TF
# ============================

class RosPublisher:
    """Optional ROS image publisher and TF broadcaster."""
    def __init__(self, out_cfg: OutputConfig):
        if not out_cfg.ros_enable:
            self.enabled = False
            return
        self.enabled = True
        self.out = out_cfg
        self._init_ros_pub()

    def _init_ros_pub(self):
        global _ros_tf2, _ros_geom
        if _ros2:
            from rclpy.node import Node
            from geometry_msgs.msg import TransformStamped
            import tf2_ros
            self.node = None
            _ros_geom = TransformStamped
            _ros_tf2 = tf2_ros
        else:
            from geometry_msgs.msg import TransformStamped
            import tf2_ros
            _ros_geom = TransformStamped
            _ros_tf2 = tf2_ros

        # Image publisher
        self.pub_image = None
        if self.out.ros_out_image_topic:
            if _ros2:
                if not hasattr(self, "node") or self.node is None:
                    self.node = ROS_NODE
                self.pub_image = self.node.create_publisher(_ros_img, self.out.ros_out_image_topic, 10)
            else:
                self.pub_image = _ros.Publisher(self.out.ros_out_image_topic, _ros_img, queue_size=1)

        # TF broadcaster
        self.br = None
        if self.out.tf_enable:
            try:
                self.br = _ros_tf2.TransformBroadcaster(self.node) if _ros2 else _ros_tf2.TransformBroadcaster()
            except Exception as e:
                print(f"[WARN] TF broadcaster not available: {e}")
                self.br = None

    def publish_image(self, img_bgr: np.ndarray, frame_id: str = "camera", stamp_sec: Optional[float] = None):
        if not self.enabled or self.pub_image is None:
            return
        msg = _ros_img()
        h, w = img_bgr.shape[:2]
        msg.height = h
        msg.width = w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = img_bgr.tobytes()
        if _ros2:
            msg.header.frame_id = frame_id
            if stamp_sec is None:
                stamp_sec = time.time()
            sec = int(stamp_sec)
            nsec = int((stamp_sec - sec) * 1e9)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
        else:
            from std_msgs.msg import Header
            msg.header = Header()
            msg.header.frame_id = frame_id
            if stamp_sec is None:
                stamp_sec = time.time()
            import rospy
            msg.header.stamp = rospy.Time.from_sec(stamp_sec)
        try:
            self.pub_image.publish(msg)
        except Exception as e:
            print(f"[WARN] failed to publish image: {e}")

    def broadcast_tf(self, parent: str, child: str,
                     tvec: np.ndarray, q_xyzw: np.ndarray,
                     stamp_sec: Optional[float] = None):
        if not self.enabled or self.br is None:
            return
        msg = _ros_geom()
        if _ros2:
            msg.header.frame_id = parent
            if stamp_sec is None:
                stamp_sec = time.time()
            sec = int(stamp_sec)
            nsec = int((stamp_sec - sec) * 1e9)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
        else:
            import rospy
            msg.header.frame_id = parent
            if stamp_sec is None:
                stamp_sec = time.time()
            msg.header.stamp = rospy.Time.from_sec(stamp_sec)
        msg.child_frame_id = child
        msg.transform.translation.x = float(tvec[0])
        msg.transform.translation.y = float(tvec[1])
        msg.transform.translation.z = float(tvec[2])
        msg.transform.rotation.x = float(q_xyzw[0])
        msg.transform.rotation.y = float(q_xyzw[1])
        msg.transform.rotation.z = float(q_xyzw[2])
        msg.transform.rotation.w = float(q_xyzw[3])
        try:
            self.br.sendTransform(msg)
        except Exception as e:
            print(f"[WARN] TF send failed: {e}")


# ============================
# Pose filter
# ============================

class PoseFilter:
    """Simple low-pass on t and q. Strength 0..1: 1 is heavy smoothing."""
    def __init__(self, enabled: bool, strength: float):
        self.enabled = enabled
        self.s = float(np.clip(strength, 0.0, 1.0))
        self.have = False
        self.t = np.zeros(3, np.float64)
        self.q = np.array([0,0,0,1], np.float64)

    def reset(self):
        self.have = False

    def update(self, t_meas: np.ndarray, q_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled or not self.have:
            self.t = t_meas.astype(np.float64)
            self.q = q_meas.astype(np.float64)
            self.have = True
            return self.t, self.q
        alpha = 1.0 - self.s
        self.t = alpha * t_meas + (1.0 - alpha) * self.t
        self.q = slerp(self.q, q_meas, alpha)
        return self.t, self.q


# ============================
# Small SE3 helpers
# ============================

class SE3:
    """Tiny helpers for 4x4 transforms and conversions."""
    @staticmethod
    def T_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(rvec.reshape(3,1))
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R
        T[:3, 3] = tvec.reshape(3)
        return T

    @staticmethod
    def rt_from_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rvec, _ = cv2.Rodrigues(T[:3,:3])
        tvec = T[:3,3].copy()
        return rvec.reshape(3), tvec.reshape(3)

    @staticmethod
    def invert(T: np.ndarray) -> np.ndarray:
        R = T[:3,:3]; t = T[:3,3]
        Ti = np.eye(4, dtype=np.float64)
        Ti[:3,:3] = R.T
        Ti[:3, 3] = -R.T @ t
        return Ti

    @staticmethod
    def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B

    @staticmethod
    def avg_transforms(Ts: List[np.ndarray]) -> np.ndarray:
        """Average SE3 by averaging translation and quaternion (simple, good enough here)."""
        if len(Ts) == 1:
            return Ts[0]
        ts = np.stack([T[:3,3] for T in Ts], 0)
        t_avg = ts.mean(0)
        qs = []
        for T in Ts:
            R = T[:3,:3]
            rvec, _ = cv2.Rodrigues(R)
            qs.append(rodrigues_to_quat(rvec.reshape(3)))
        q = qs[0]
        for i in range(1, len(qs)):
            q = slerp(q, qs[i], 1.0/(i+1))
        R_avg = quat_to_R(q)
        Tout = np.eye(4, dtype=np.float64)
        Tout[:3,:3] = R_avg
        Tout[:3, 3] = t_avg
        return Tout


# ============================
# Aruco detector
# ============================

class ArucoDetector:
    """Version-adaptive detector wrapper."""
    def __init__(self, dic_name: str, refine: bool):
        self.dict, self.dict_name = self._resolve_dictionary(dic_name)
        self.detector = self._make_detector(refine)

    @staticmethod
    def _resolve_dictionary(name: str):
        raw = name.strip().upper()
        if not raw.startswith("DICT_"):
            if raw in ("ARUCO_ORIGINAL", "ARUCO-ORIGINAL"):
                raw = "DICT_ARUCO_ORIGINAL"
            else:
                raw = "DICT_" + raw
        if not hasattr(aruco, raw):
            raise ValueError(f"Unknown dictionary '{name}'.")
        return aruco.getPredefinedDictionary(getattr(aruco, raw)), raw

    def _make_detector(self, refine: bool):
        try:
            params = aruco.DetectorParameters()
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX if refine else aruco.CORNER_REFINE_NONE
            params.aprilTagQuadDecimate = 1.0
            params.minCornerDistanceRate = 0.02
            params.minMarkerPerimeterRate = 0.03
            params.maxMarkerPerimeterRate = 4.0
            return aruco.ArucoDetector(self.dict, params)
        except Exception:
            return None

    def detect(self, gray: np.ndarray):
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, self.dict)
        return corners, ids, rejected


# ============================
# Marker tracker (multi-ID)
# ============================

class MarkerTracker:
    """
    Track multiple Aruco markers. Optionally restrict to a set of IDs.
    Maintain relative transforms to a main marker and fuse camera pose accordingly.
    """

    def __init__(self, cfg: ArucoConfig, K: np.ndarray, D: np.ndarray,
                 filter_enable: bool, filter_strength: float):
        self.cfg = cfg
        self.K, self.D = K, D
        self.allowed = set(cfg.marker_ids) if cfg.marker_ids else None
        self.main_id = cfg.marker_main_id
        self.border_bits = cfg.border_bits
        self.marker_size_m = cfg.marker_size_mm / 1000.0
        self.flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        # state
        self.tracked: Dict[int, Dict] = {}          # id -> {T_c_m (4x4), t,q, filt}
        self.rel_to_main: Dict[int, np.ndarray] = {}# id -> T_main_from_id
        self.last_fused_T_c_main: Optional[np.ndarray] = None

    def _estimate_marker_pose(self, img_corners: np.ndarray) -> Optional[np.ndarray]:
        img_pts = np.squeeze(img_corners).astype(np.float64).reshape(4,2)
        s = 0.5 * self.marker_size_m
        obj_pts = np.array([[-s,  s, 0.0],
                            [ s,  s, 0.0],
                            [ s, -s, 0.0],
                            [-s, -s, 0.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D, flags=self.flag)
        if not ok:
            return None
        return SE3.T_from_rt(rvec.reshape(3), tvec.reshape(3))

    def _get_or_make_filter(self, id_: int) -> PoseFilter:
        if id_ in self.tracked and "filt" in self.tracked[id_]:
            return self.tracked[id_]["filt"]
        pf = PoseFilter(enabled=self.cfg.corner_refine,  # use same switch as refine? or runtime later
                        strength=0.5)  # local smoothing for markers, fixed mild
        return pf

    def update(self, ids: Optional[np.ndarray], corners: Optional[List[np.ndarray]]):
        """Update per-frame from detections. Learn relations to main when co-observed."""
        if ids is None or len(ids) == 0:
            # keep last fused pose if any
            return

        ids = ids.reshape(-1).astype(int)
        # choose main if not set: pick first in allowed set or first seen
        if self.main_id is None:
            if self.allowed:
                for _id in ids:
                    if _id in self.allowed:
                        self.main_id = _id
                        break
            else:
                self.main_id = int(ids[0])

        # iterate all detections
        visible = {}
        for i, mid in enumerate(ids):
            if self.allowed and mid not in self.allowed:
                continue
            T_c_m = self._estimate_marker_pose(corners[i])
            if T_c_m is None:
                continue
            visible[mid] = T_c_m

            # filter per-marker
            rvec, tvec = SE3.rt_from_T(T_c_m)
            q = rodrigues_to_quat(rvec)
            pf = self._get_or_make_filter(mid)
            t_s, q_s = pf.update(tvec, q) if pf.enabled else (tvec, q)
            R_s = quat_to_R(q_s)
            T_c_m_s = np.eye(4, dtype=np.float64); T_c_m_s[:3,:3] = R_s; T_c_m_s[:3,3] = t_s
            self.tracked[mid] = {"T_c_m": T_c_m_s, "t": t_s, "q": q_s, "filt": pf}

        # learn relations to main if both present
        if self.main_id is not None and self.main_id in visible:
            T_c_main = visible[self.main_id]
            T_main_c = SE3.invert(T_c_main)
            for mid, T_c_m in visible.items():
                if mid == self.main_id:
                    continue
                T_main_m = SE3.compose(T_main_c, T_c_m)  # T_main_from_id
                if mid in self.rel_to_main:
                    # simple running average
                    self.rel_to_main[mid] = SE3.avg_transforms([self.rel_to_main[mid], T_main_m])
                else:
                    self.rel_to_main[mid] = T_main_m

        # fuse camera pose relative to main
        self._fuse_camera_pose(visible)

    def _fuse_camera_pose(self, visible: Dict[int, np.ndarray]):
        """Fuse T_c_main using all visible markers that have a relation to main."""
        if self.main_id is None:
            self.last_fused_T_c_main = None
            return

        candidates = []

        # if main visible, use it
        if self.main_id in visible:
            candidates.append(visible[self.main_id])

        # others with relation learned
        for mid, T_c_m in visible.items():
            if mid == self.main_id:
                continue
            if mid in self.rel_to_main:
                # T_c_main_i = T_c_m * T_m_main
                T_m_main = SE3.invert(self.rel_to_main[mid])
                T_c_main_i = SE3.compose(T_c_m, T_m_main)
                candidates.append(T_c_main_i)

        if len(candidates) == 0:
            # try tracked ones not visible by using their last T and relation
            for mid, rec in self.tracked.items():
                if mid == self.main_id:
                    continue
                if mid in self.rel_to_main:
                    T_c_m = rec["T_c_m"]
                    T_m_main = SE3.invert(self.rel_to_main[mid])
                    candidates.append(SE3.compose(T_c_m, T_m_main))

        if len(candidates) == 0:
            return

        self.last_fused_T_c_main = SE3.avg_transforms(candidates)

    def get_fused_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return fused (t,q) of camera in main marker frame (object->camera)."""
        if self.last_fused_T_c_main is None:
            return None, None
        rvec, tvec = SE3.rt_from_T(self.last_fused_T_c_main)
        q = rodrigues_to_quat(rvec)
        return tvec, q


# ============================
# Diamond tracker (multi-diamond)
# ============================

class DiamondTracker:
    """
    Track Charuco diamonds via detectCharucoDiamond and solvePnP on the outer square.
    Optionally restrict to specific diamonds (ids in groups of 4).
    Learn relative transforms between diamonds to fuse pose when multiple are visible.
    """

    def __init__(self, cfg: ArucoConfig, K: np.ndarray, D: np.ndarray,
                 filter_enable: bool, filter_strength: float, aruco_detector: ArucoDetector):
        self.cfg = cfg
        self.K, self.D = K, D
        self.marker_len_m = cfg.marker_size_mm / 1000.0
        self.square_len_m = cfg.diamond_square_mm / 1000.0
        self.outer_side_m = 2.0 * self.square_len_m
        self.flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        self.allowed = None
        if cfg.diamond_ids:
            # normalize each group as sorted tuple
            self.allowed = {tuple(sorted(map(int, g))) for g in cfg.diamond_ids}
        self.detector = aruco_detector
        # state
        self.tracked: Dict[Tuple[int,int,int,int], Dict] = {}   # key -> {T_c_d, filt}
        self.rel_to_main: Dict[Tuple[int,int,int,int], np.ndarray] = {}
        self.main_key: Optional[Tuple[int,int,int,int]] = None
        self.last_fused_T_c_main: Optional[np.ndarray] = None

    def _estimate_diamond_pose_from_square(self, diamond_corners: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate pose from diamond outer square corners (4x2) with solvePnP.
        """
        img_pts = np.squeeze(diamond_corners).astype(np.float64).reshape(4,2)
        s = 0.5 * self.outer_side_m
        obj_pts = np.array([[-s,  s, 0.0],
                            [ s,  s, 0.0],
                            [ s, -s, 0.0],
                            [-s, -s, 0.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D, flags=self.flag)
        if not ok:
            return None
        return SE3.T_from_rt(rvec.reshape(3), tvec.reshape(3))

    def update(self, gray: np.ndarray, marker_corners, marker_ids):
        """Detect diamonds, update tracking and fused pose."""
        if marker_ids is None or len(marker_ids) == 0:
            return

        # Detect diamonds (Python API)
        try:
            ratio = self.marker_len_m / self.square_len_m
            diamonds, diamond_ids = cv2.aruco.detectCharucoDiamond(
                gray, marker_corners, marker_ids, ratio, cameraMatrix=self.K, distCoeffs=self.D
            )
        except Exception:
            diamonds, diamond_ids = [], None

        if diamond_ids is None or len(diamond_ids) == 0:
            return

        visible = {}
        keys_in_frame = []

        for dc, dids in zip(diamonds, diamond_ids):
            ids4 = tuple(sorted(int(x) for x in dids.reshape(-1).tolist()))
            if self.allowed and ids4 not in self.allowed:
                continue
            T_c_d = self._estimate_diamond_pose_from_square(dc)
            if T_c_d is None:
                continue
            visible[ids4] = T_c_d
            keys_in_frame.append(ids4)

            # record filtered state
            rec = self.tracked.get(ids4, None)
            if rec is None:
                pf = PoseFilter(enabled=True, strength=0.5)
            else:
                pf = rec["filt"]
            rvec, tvec = SE3.rt_from_T(T_c_d)
            q = rodrigues_to_quat(rvec)
            t_s, q_s = pf.update(tvec, q) if pf.enabled else (tvec, q)
            R_s = quat_to_R(q_s)
            T_s = np.eye(4, dtype=np.float64); T_s[:3,:3] = R_s; T_s[:3,3] = t_s
            self.tracked[ids4] = {"T_c_d": T_s, "filt": pf}

        # choose main key
        if self.main_key is None:
            if self.allowed:
                # pick the first allowed that is visible
                for k in list(self.allowed):
                    if k in visible:
                        self.main_key = k
                        break
            else:
                if len(keys_in_frame) > 0:
                    self.main_key = keys_in_frame[0]

        # learn relative transforms to main
        if self.main_key in visible:
            T_c_main = visible[self.main_key]
            T_main_c = SE3.invert(T_c_main)
            for k, T_c_d in visible.items():
                if k == self.main_key:
                    continue
                T_main_d = SE3.compose(T_main_c, T_c_d)
                if k in self.rel_to_main:
                    self.rel_to_main[k] = SE3.avg_transforms([self.rel_to_main[k], T_main_d])
                else:
                    self.rel_to_main[k] = T_main_d

        # fuse pose
        self._fuse_camera_pose(visible)

    def _fuse_camera_pose(self, visible: Dict[Tuple[int,int,int,int], np.ndarray]):
        if self.main_key is None:
            return
        candidates = []
        if self.main_key in visible:
            candidates.append(visible[self.main_key])
        for k, T_c_d in visible.items():
            if k == self.main_key:
                continue
            if k in self.rel_to_main:
                T_d_main = SE3.invert(self.rel_to_main[k])
                candidates.append(SE3.compose(T_c_d, T_d_main))
        if len(candidates) == 0:
            for k, rec in self.tracked.items():
                if k == self.main_key:
                    continue
                if k in self.rel_to_main:
                    T_c_d = rec["T_c_d"]
                    T_d_main = SE3.invert(self.rel_to_main[k])
                    candidates.append(SE3.compose(T_c_d, T_d_main))
        if len(candidates) == 0:
            return
        self.last_fused_T_c_main = SE3.avg_transforms(candidates)

    def get_fused_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.last_fused_T_c_main is None:
            return None, None
        rvec, tvec = SE3.rt_from_T(self.last_fused_T_c_main)
        q = rodrigues_to_quat(rvec)
        return tvec, q


# ============================
# Open3D visualizer
# ============================

# ============================
# Simple ROS-less 3D Pose Visualizer (Open3D O3DVisualizer-based)
# ============================

class SimplePoseVisualizer3D:
    """
    Interactive 3D visualization using Open3D O3DVisualizer.

    Coordinate frames:
      - World frame == camera frame.
      - Camera is at the origin (0,0,0) with its own axes.
      - The main object (marker or diamond) is drawn at its pose relative to the camera.
        (So when the camera moves in the real world, the object shifts in this view.)
      - A 3D label ("MARKER" or "DIAMOND") is drawn at the object position.

    This class is "ROS-less": it only visualizes the 3D relationship between camera and
    detected marker/diamond. It does not know TF or ROS frames.

    Usage pattern (unchanged from previous version):
        viz = SimplePoseVisualizer3D(...)
        ...
        viz.set_detections(ids, corners, marker_size_m, main_id=...)
        viz.set_image_size(w, h)
        viz.update_and_show(t_obj_in_cam, q_obj_in_cam, obj_side_m, title=...)

    Where:
        - t_obj_in_cam: translation of object in camera frame [meters], from solvePnP
        - q_obj_in_cam: quaternion [x,y,z,w] of object in camera frame
        - obj_side_m:   physical side of the object:
            * marker mode: marker_size_m
            * diamond mode: 2 * square_size_m (outer diamond size)
        - title: "marker" or "diamond"

    Internals:
      - Uses open3d.visualization.gui.Application + open3d.visualization.O3DVisualizer.
      - Runs the GUI main loop in a background thread.
      - All geometry updates are posted to the GUI thread via post_to_main_thread.
      - Camera view is set ONCE (reset_camera_to_default) on first valid detection,
        so you can freely rotate and zoom afterwards without it being overridden.
    """

    def __init__(self, window: str = "Pose3D", fov_deg: float = 55.0,
                 img_size_hint: Tuple[int, int] = (1280, 720),
                 K: Optional[np.ndarray] = None, dist: Optional[np.ndarray] = None,
                 show_grid: bool = False,
                 **kwargs):
        try:
            import open3d as o3d
            import open3d.visualization.gui as gui
            import open3d.visualization.rendering as rendering
        except Exception as e:
            raise RuntimeError("Open3D with GUI support is required for --viz3d. "
                               "Install: pip install open3d") from e

        self.o3d = o3d
        self.gui = gui
        self.render = rendering

        self.window_title = window
        self.width, self.height = 960, 720
        self.fov = float(fov_deg)
        self.img_w, self.img_h = int(img_size_hint[0]), int(img_size_hint[1])
        self.K = None if K is None else K.astype(np.float64).copy()
        self.D = None if dist is None else dist.astype(np.float64).copy()

        # Stored 2D detections if needed in future
        self._det_ids: Optional[np.ndarray] = None
        self._det_corners: Optional[List[np.ndarray]] = None
        self._det_marker_size_m: Optional[float] = None
        self._det_main_id: Optional[int] = None

        # GUI / scene state
        self._app = self.gui.Application.instance
        try:
            # initialize() will throw if already initialized; ignore that
            self._app.initialize()
        except Exception:
            pass

        self._vis = self.o3d.visualization.O3DVisualizer(self.window_title,
                                                         self.width, self.height)
        # Basic UI toggles
        self._vis.show_settings = False
        self._vis.show_axes = True
        self._vis.show_ground = bool(show_grid)
        try:
            self._vis.show_skybox(False)
        except Exception:
            pass

        # Background color (RGBA)
        try:
            self._vis.set_background([0.97, 0.97, 0.97, 1.0])
        except Exception:
            pass

        # Default material for line / mesh geometry
        self._mat = self.render.MaterialRecord()
        self._mat.shader = "defaultUnlit"
        self._mat.base_color = (0.05, 0.05, 0.05, 1.0)
        self._mat.line_width = 2.0

        self._app.add_window(self._vis)

        # Geometry registry: name -> (kind, size_m)
        # kind is a simple string so we can re-create geometry easily if needed.
        self._geom_kinds: Dict[str, Tuple[str, float]] = {}
        self._camera_initialized = False
        self._scene_created = False

        # Start GUI main loop in background thread
        import threading
        self._gui_thread = threading.Thread(target=self._app.run, daemon=True)
        self._gui_thread.start()

    # ------------------------------------------------------------------
    # Public API (same names as previous version)
    # ------------------------------------------------------------------

    def set_image_size(self, w: int, h: int):
        """Record image size; currently used only for potential frustum scaling."""
        self.img_w, self.img_h = int(w), int(h)

    def set_grid_visible(self, enabled: bool):
        """Toggle ground grid."""
        enabled = bool(enabled)

        def _cb():
            self._vis.show_ground = enabled
            self._vis.post_redraw()

        self.gui.Application.instance.post_to_main_thread(self._vis, _cb)

    def set_detections(self, ids: np.ndarray, corners: List[np.ndarray],
                       marker_size_m: float, main_id: Optional[int] = None):
        """
        Store detected markers for potential visual hints.

        Current implementation uses only the main object's pose for visualization,
        so this method just stores the inputs without drawing per-marker geometry yet.
        (It is kept for API compatibility and possible future extensions.)
        """
        if ids is None or len(ids) == 0:
            self._det_ids = None
            self._det_corners = None
            self._det_marker_size_m = None
            self._det_main_id = None
            return
        ids = np.array(ids).reshape(-1).astype(int)
        norm_corners = []
        for c in corners:
            cc = np.squeeze(c)
            if cc.shape != (4, 2):
                cc = cc.reshape(4, 2)
            norm_corners.append(cc.astype(np.float64))
        self._det_ids = ids
        self._det_corners = norm_corners
        self._det_marker_size_m = float(marker_size_m)
        self._det_main_id = int(main_id) if main_id is not None else None

    def set_tf_frame(self, tf_frame_name: str, camera_frame: str = "camera"):
        """
        Kept for API compatibility. This visualizer is ROS-less and does not
        display TF frames, so this is a no-op.
        """
        # If you want TF visualization in the future, this is the natural hook.
        return

    def update_and_show(self, t_obj_in_cam: np.ndarray, q_obj_in_cam: np.ndarray,
                        obj_square_side_m: float, title: str = "marker"):
        """
        Update the visualizer with a new object pose.

        t_obj_in_cam, q_obj_in_cam:
            Pose of the main object in camera frame (output of solvePnP etc.).
        obj_square_side_m:
            For marker: marker size in meters.
            For diamond: outer square size (2 * square length) in meters.
        title:
            "marker" or "diamond" (controls geometry and label).
        """
        if t_obj_in_cam is None or q_obj_in_cam is None:
            return

        # Normalize pose
        t = np.asarray(t_obj_in_cam, dtype=np.float64).reshape(3)
        q = np.asarray(q_obj_in_cam, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return
        q = q / n
        R_co = quat_to_R(q)      # object -> camera
        t_co = t

        # We use world = camera frame.
        # So we want object in world frame: camera->object transform.
        # If solvePnP returns object->camera, the inverse gives camera->object:
        R_oc = R_co.T
        t_oc = -R_oc @ t_co

        # Schedule GUI update on main GUI thread
        def _cb():
            self._ensure_scene_created(obj_square_side_m, title)
            self._update_object_pose(R_oc, t_oc, obj_square_side_m, title)
            if not self._camera_initialized:
                # Let Open3D compute a nice initial camera for current geometry.
                try:
                    self._vis.reset_camera_to_default()
                except Exception:
                    pass
                self._camera_initialized = True
            self._vis.post_redraw()

        self.gui.Application.instance.post_to_main_thread(self._vis, _cb)

    # ------------------------------------------------------------------
    # Internal helpers (GUI thread only)
    # ------------------------------------------------------------------

    def _ensure_scene_created(self, obj_side_m: float, title: str):
        """Create static scene elements (camera axes, initial label) once."""
        if self._scene_created:
            return

        # Camera axes at origin (world = camera)
        cam_axes = self.o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(0.3 * obj_side_m, 0.05)
        )
        self._add_or_replace_geometry("camera_axes", cam_axes, kind=("axes", obj_side_m))

        # Optional: a tiny origin square for reference (under the camera)
        # (Can be commented out if not needed.)
        origin_sq = self._make_square_lines(0.25 * obj_side_m)
        self._add_or_replace_geometry("origin_square", origin_sq, kind=("origin_sq", obj_side_m))

        # Initial label (will be repositioned each frame)
        label_text = "MARKER" if title.lower().startswith("marker") else "DIAMOND"
        try:
            self._vis.clear_3d_labels()
        except Exception:
            pass
        try:
            # start with label near origin; will move to object later
            self._vis.add_3d_label([0.0, 0.0, 0.0], label_text)
        except Exception:
            pass

        self._scene_created = True

    def _update_object_pose(self, R_oc: np.ndarray, t_oc: np.ndarray,
                            obj_side_m: float, title: str):
        """Rebuild the main object geometry at the new pose."""
        # Build object geometry centered at origin in its own frame
        if title.lower().startswith("diamond"):
            geom = self._make_diamond_wire(obj_side_m)
            label_text = "DIAMOND"
        else:
            geom = self._make_square_lines(obj_side_m)
            label_text = "MARKER"

        # Transform into world (camera) frame
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_oc
        T[:3, 3] = t_oc.reshape(3)
        geom.transform(T)

        self._add_or_replace_geometry("main_object", geom, kind=("object", obj_side_m))

        # Update label at object position
        try:
            self._vis.clear_3d_labels()
        except Exception:
            pass
        try:
            self._vis.add_3d_label(t_oc.tolist(), label_text)
        except Exception:
            pass

    def _add_or_replace_geometry(self, name: str, geom, kind: Tuple[str, float]):
        """Add or replace a named geometry in the O3DVisualizer scene."""
        if name in self._geom_kinds:
            try:
                self._vis.remove_geometry(name)
            except Exception:
                pass
        try:
            self._vis.add_geometry(name, geom, self._mat)
        except Exception as e:
            # If something fails, do not leave a half-registered name
            print(f"[viz3d] Failed to add geometry '{name}': {e}")
            if name in self._geom_kinds:
                del self._geom_kinds[name]
            return
        self._geom_kinds[name] = kind

    # ------------------------------------------------------------------
    # Geometry constructors (object frame)
    # ------------------------------------------------------------------

    def _make_square_lines(self, side: float):
        """Return a LineSet for a square centered at the origin on the Z=0 plane."""
        s = float(side) * 0.5
        pts = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)
        lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        ls = self.o3d.geometry.LineSet()
        ls.points = self.o3d.utility.Vector3dVector(pts)
        ls.lines = self.o3d.utility.Vector2iVector(lines)
        ls.colors = self.o3d.utility.Vector3dVector(
            np.tile(np.array([[0.10, 0.10, 0.10]], dtype=np.float64), (len(lines), 1))
        )
        return ls

    def _make_diamond_wire(self, outer_side: float):
        """
        Return a LineSet that approximates a ChArUco diamond:
          - outer square of side 'outer_side'
          - 2x2 grid inside (so you see four small squares)
        """
        s = float(outer_side) * 0.5
        # Outer points
        pts = [
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ]
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0]  # outer
        ]

        # Internal grid (2x2): vertical lines and horizontal lines
        # squares are of size outer_side / 2
        step = outer_side / 2.0
        xs = [-s + step, -s + 2.0 * step]  # but second is +s so we only want 1 internal vertical
        ys = [-s + step, -s + 2.0 * step]

        # internal vertical
        pts.append([0.0, -s, 0.0])    # index 4
        pts.append([0.0,  s, 0.0])    # index 5
        lines.append([4, 5])

        # internal horizontal
        pts.append([-s, 0.0, 0.0])    # index 6
        pts.append([ s, 0.0, 0.0])    # index 7
        lines.append([6, 7])

        pts_arr = np.array(pts, dtype=np.float64)
        lines_arr = np.array(lines, dtype=np.int32)

        ls = self.o3d.geometry.LineSet()
        ls.points = self.o3d.utility.Vector3dVector(pts_arr)
        ls.lines = self.o3d.utility.Vector2iVector(lines_arr)
        ls.colors = self.o3d.utility.Vector3dVector(
            np.tile(np.array([[0.10, 0.10, 0.10]], dtype=np.float64), (len(lines_arr), 1))
        )
        return ls


# ============================
# Processor
# ============================

class Processor:
    def __init__(self, cfg: Config, cam_src, ros_pub: Optional[RosPublisher]):
        self.cfg = cfg
        self.cam = cam_src
        self.aru = ArucoDetector(cfg.aruco.dictionary, refine=cfg.aruco.corner_refine)
        self.pub = ros_pub
        self.K, self.D = self._init_intrinsics()
        self.filter = PoseFilter(cfg.runtime.filter_enable, cfg.runtime.filter_strength)
        self.last_detect_ts = None
        self.last_pose = None  # (t(3), q(4))
        self._last_warn = 0.0
        # ChArUco diamond detector (OpenCV 4.12+)
        self.charuco_detector = self._init_charuco_detector()

    def _init_intrinsics(self):
        # From camera/ROS if available
        K, D = (None, None)
        try:
            K, D = self.cam.intrinsics()
        except Exception as e:
            print(f"[processor init] Failed to get camera intrinsics: {e}")
        if K is None:
            c = self.cfg.camera
            if c.fx and c.fy and c.cx and c.cy:
                K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], dtype=np.float64)
        if D is None:
            c = self.cfg.camera
            if c.dist:
                D = np.array(c.dist, dtype=np.float64)
        if K is None:
            raise RuntimeError("Camera intrinsics unknown. Provide via RealSense, ROS CameraInfo, or YAML/CLI (fx,fy,cx,cy,dist).")
        if D is None:
            D = np.zeros(5, np.float64)
        return K, D

    def _init_charuco_detector(self):
        """
        Create a CharucoDetector for diamond detection (OpenCV 4.12+).

        If this fails (older OpenCV), diamond mode will report a clear error.
        """
        if self.cfg.aruco.detect_mode != "diamond":
            return None
        try:
            square_len_m = self.cfg.aruco.diamond_square_mm / 1000.0
            marker_len_m = self.cfg.aruco.marker_size_mm / 1000.0
            board = aruco.CharucoBoard((3, 3), square_len_m, marker_len_m, self.aru.dict)
            charuco_params = aruco.CharucoParameters()
            charuco_params.cameraMatrix = self.K
            charuco_params.distCoeffs = self.D
            det_params = aruco.DetectorParameters()
            det_params.cornerRefinementMethod = (
                aruco.CORNER_REFINE_SUBPIX if self.cfg.aruco.corner_refine else aruco.CORNER_REFINE_NONE
            )
            return aruco.CharucoDetector(board, charuco_params, det_params)
        except Exception as e:
            print(f"[WARN] CharucoDetector not available in this OpenCV build: {e}")
            return None

    def _estimate_marker_pose(self, corners: np.ndarray, marker_size_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        OpenCV >= 4.7 compliant single-marker pose:
        Use cv.solvePnP (prefer IPPE_SQUARE if available) instead of deprecated
        cv2.aruco.estimatePoseSingleMarkers.
        """
        # corners can be (1,4,2), (4,1,2) or (4,2): squeeze to (4,2)
        img_pts = np.squeeze(corners).astype(np.float64)
        if img_pts.shape != (4, 2):
            img_pts = img_pts.reshape(4, 2)

        # Marker model (Z=0 plane), origin at marker center (same convention as aruco)
        s = float(marker_size_m) * 0.5
        obj_pts = np.array([[-s,  s, 0.0],
                            [ s,  s, 0.0],
                            [ s, -s, 0.0],
                            [-s, -s, 0.0]], dtype=np.float64)

        # Prefer IPPE_SQUARE (stable for a planar square with 4 points), else fallback
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D, flags=flag)
        if not ok:
            raise RuntimeError("solvePnP failed for single marker")
        return rvec.reshape(3), tvec.reshape(3)

    def _detect_diamonds(self, img_gray: np.ndarray, marker_corners, marker_ids):
        """
        Detect ChArUco diamonds using cv2.aruco.CharucoDetector.detectDiamonds.

        Returns (diamondCorners, diamondIds). Both may be empty on failure.
        """
        if self.charuco_detector is None:
            raise RuntimeError(
                "CharucoDetector is not available; diamond detection requires an OpenCV build "
                "with cv2.aruco.CharucoDetector (e.g. 4.7+ / 4.12)."
            )

        if marker_ids is None or len(marker_ids) == 0:
            return [], None

        # Normalize inputs for the detector
        # mc = [np.squeeze(c).astype(np.float32) for c in marker_corners]
        # ids = np.array(marker_ids, dtype=np.uint16).reshape(-1)

        square_len_m = self.cfg.aruco.diamond_square_mm / 1000.0
        marker_len_m = self.cfg.aruco.marker_size_mm / 1000.0
        # According to the Charuco diamond tutorial:
        # squareMarkerLengthRate = squareLength / markerLength
        # square_marker_rate = square_len_m / max(marker_len_m, 1e-9)

        diamond_corners, diamond_ids, _, _ = self.charuco_detector.detectDiamonds(
            image=img_gray, markerCorners=marker_corners, markerIds=marker_ids.astype(np.uint16),
        )
        return diamond_corners, diamond_ids

    def _estimate_diamond_pose_from_corners(self, diamond_corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pose of a ChArUco diamond from its outer square corners using cv2.solvePnP.

        diamond_corners: (4,2) or (1,4,2) image points.
        Returns (rvec, tvec) in camera frame.
        """
        img_pts = np.squeeze(diamond_corners).astype(np.float64)
        if img_pts.shape != (4, 2):
            img_pts = img_pts.reshape(4, 2)

        square_len_m = self.cfg.aruco.diamond_square_mm / 1000.0
        outer_side = 2.0 * square_len_m  # outer square spans two ChArUco squares
        s = 0.5 * outer_side
        obj_pts = np.array([[-s,  s, 0.0],
                            [ s,  s, 0.0],
                            [ s, -s, 0.0],
                            [-s, -s, 0.0]], dtype=np.float64)

        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D, flags=flag)
        if not ok:
            raise RuntimeError("solvePnP failed for diamond.")
        return rvec.reshape(3), tvec.reshape(3)

    def _avg_poses(self, rvecs: List[np.ndarray], tvecs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # Average in camera frame
        if len(rvecs) == 1:
            return rvecs[0], tvecs[0]
        # average translations
        t = np.mean(np.stack(tvecs, axis=0), axis=0)
        # average rotations via quaternions
        qs = [rodrigues_to_quat(r) for r in rvecs]
        q = qs[0]
        for i in range(1, len(qs)):
            q = slerp(q, qs[i], 1.0/(i+1))
        # Convert back to rvec for axes drawing if needed
        R = quat_to_R(q)
        rvec, _ = cv2.Rodrigues(R)
        return rvec.reshape(3), t.reshape(3)

    def _draw_axes(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, length_m: float):
        try:
            cv2.drawFrameAxes(img, self.K, self.D, rvec, tvec, length_m)
        except Exception:
            pass

    def _annotate(self, img: np.ndarray, corners, ids, color=(0, 255, 0)):
        cv2.aruco.drawDetectedMarkers(img, corners, ids, color)

    def process_frame(self, frame_bgr: np.ndarray, stamp_sec: Optional[float]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Returns annotated image and current pose (t, q) if any.
        """
        img = frame_bgr.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img.copy()
        corners, ids, rejected = self.aru.detect(gray)
        pose_t, pose_q = None, None

        # draw rejected corners
        if self.cfg.output.draw_rejected and rejected is not None:
            cv2.aruco.drawDetectedMarkers(img, rejected, None, (0, 0, 255))

        if ids is not None and len(ids) > 0:
            ids = ids.flatten().astype(int)
            self._annotate(img, corners, ids.reshape(-1, 1))

            if self.cfg.aruco.detect_mode == "marker":
                # If a specific ID is requested, pick it; else use first
                chosen_idx = None
                if self.cfg.aruco.marker_id is not None:
                    for i, _id in enumerate(ids):
                        if _id == self.cfg.aruco.marker_id:
                            chosen_idx = i
                            break
                else:
                    chosen_idx = 0
                if chosen_idx is not None:
                    rvec, tvec = self._estimate_marker_pose(
                        corners[chosen_idx],
                        self.cfg.aruco.marker_size_mm / 1000.0
                    )
                    self._draw_axes(img, rvec, tvec, self.cfg.output.draw_axes_length_m)
                    pose_q = rodrigues_to_quat(rvec)
                    pose_t = tvec
            else:
                # diamond mode: detect diamonds, optionally filter by requested ids, then solvePnP on each
                try:
                    diamond_corners, diamond_ids = self._detect_diamonds(gray, corners, ids)
                except RuntimeError as e:
                    print(f"[ERR] Diamond detection failed: {e}")
                    diamond_corners, diamond_ids = [], None

                rvecs, tvecs = [], []
                if diamond_ids is not None and len(diamond_ids) > 0:
                    # Optional filtering by explicit diamond id sets
                    allowed = None
                    if self.cfg.aruco.diamond_ids:
                        allowed = {tuple(sorted(map(int, g))) for g in self.cfg.aruco.diamond_ids}

                    for d_idx, dids in enumerate(diamond_ids):
                        id_list = [int(x) for x in np.array(dids).reshape(-1).tolist()]
                        key = tuple(sorted(id_list))
                        if allowed is not None and key not in allowed:
                            continue
                        try:
                            rvec, tvec = self._estimate_diamond_pose_from_corners(diamond_corners[d_idx])
                        except RuntimeError:
                            continue
                        rvecs.append(rvec)
                        tvecs.append(tvec)
                        self._draw_axes(img, rvec, tvec, self.cfg.output.draw_axes_length_m)

                    # Visualize all detected diamonds (allowed or not)
                    try:
                        if len(diamond_corners) > 0 and diamond_ids is not None:
                            cv2.aruco.drawDetectedDiamonds(img, diamond_corners, diamond_ids)
                    except Exception:
                        pass

                    if len(rvecs) > 0:
                        rr, tt = self._avg_poses(rvecs, tvecs)  # average if multiple diamonds
                        pose_q = rodrigues_to_quat(rr)
                        pose_t = tt

        # Pose smoothing
        if pose_t is not None and pose_q is not None:
            t_s, q_s = self.filter.update(pose_t, pose_q)
            self.last_pose = (t_s.copy(), q_s.copy())
            self.last_detect_ts = stamp_sec if stamp_sec is not None else time.time()
        else:
            # no detection; keep last pose (stale)
            t_s, q_s = (None, None)
            if self.last_pose is not None:
                t_s, q_s = self.last_pose

        # Warn if stale
        if self.cfg.output.tf_enable and self.last_pose is not None:
            gap = (time.time() - (self.last_detect_ts or time.time()))
            if gap > self.cfg.runtime.stale_tf_sec:
                if time.time() - self._last_warn > self.cfg.runtime.warn_every_sec:
                    print(f"[WARN] No detections for {gap:.2f}s; re-broadcasting last TF (assuming occlusion, no motion).")
                    self._last_warn = time.time()

        return img, (ids if ids is not None else None), (corners if ids is not None else None), ((t_s, q_s) if (t_s is not None and q_s is not None) else (None, None))


# ============================
# Main
# ============================

def main():
    parser = argparse.ArgumentParser(description="RealSense/ROS Aruco marker or Charuco diamond pose tool.")
    # Config
    parser.add_argument("--config", "--cfg", "-c", type=str, default=None, help="YAML config file.")

    # Camera
    parser.add_argument("--source", choices=["realsense", "ros"], default=None)
    parser.add_argument("--rs-serial", type=str, default=None)
    parser.add_argument("--rs-size", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--rs-fps", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)

    # Intrinsics override
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--dist", type=float, nargs="+", default=None)

    # Aruco
    parser.add_argument("--dictionary", type=str, default=None)
    parser.add_argument("--detect-mode", choices=["marker", "diamond"], default=None)
    parser.add_argument("--marker-id", type=int, default=None, help="Backward compat: main marker id.")
    parser.add_argument("--marker-ids", type=int, nargs="+", default=None, help="Restrict tracking to these ids.")
    parser.add_argument("--marker-main-id", type=int, default=None, help="Explicit main marker id (overrides --marker-id).")
    parser.add_argument("--marker-mm", type=float, default=None)
    parser.add_argument("--diamond-mm", type=float, default=None, help="Square size of diamond in mm.")
    parser.add_argument("--diamond-ids", type=int, nargs="+", default=None,
                        help="Diamond ids in groups of 4 (e.g. --diamond-ids 10 11 12 13 20 21 22 23).")
    parser.add_argument("--border-bits", type=int, default=None)
    parser.add_argument("--no-corner-refine", action="store_true")

    # Runtime
    parser.add_argument("--mode", choices=["single", "continuous", "cont"], default=None)
    parser.add_argument("--throttle", type=float, default=None)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--filter-strength", type=float, default=None)
    parser.add_argument("--stale-tf", type=float, default=None)

    # Output
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--window-name", type=str, default=None)
    parser.add_argument("--axis-m", type=float, default=None)
    parser.add_argument("--draw-rejected", "--rejected", action="store_true", help="Draw rejected markers.")

    # ROS
    parser.add_argument("--ros", action="store_true", help="Enable ROS mode.")
    parser.add_argument("--ros-version", choices=["auto", "ros2", "ros1"], default=None)
    parser.add_argument("--ros-in", type=str, default=None, help="Input image topic.")
    parser.add_argument("--ros-info", type=str, default=None, help="CameraInfo topic.")
    parser.add_argument("--ros-out", type=str, default=None, help="Annotated image topic.")
    parser.add_argument("--tf", action="store_true", help="Enable TF broadcaster.")
    parser.add_argument("--tf-parent", type=str, default=None)
    parser.add_argument("--tf-child", type=str, default=None)
    parser.add_argument("--tf-hz", type=float, default=None)

    # Viewer
    parser.add_argument("--viz3d", action="store_true", help="Open Open3D 3D window for pose check.")
    parser.add_argument("--viz3d-fov", type=float, default=55.0, help="Viewer field of view (deg).")
    parser.add_argument("--viz3d-grid", action="store_true", help="Show ground grid in the 3D viewer.")

    args = parser.parse_args()
    cfg = build_config(args)

    print("=== Effective Configuration ===")
    print(json.dumps({
        "camera": asdict(cfg.camera),
        "aruco": asdict(cfg.aruco),
        "runtime": asdict(cfg.runtime),
        "output": asdict(cfg.output),
    }, indent=2))
    print("===============================")

    if cfg.output.ros_enable:
        init_ros(cfg.output.ros_version)

    # camera source
    if cfg.camera.source == "realsense":
        cam = RealSenseSource(cfg.camera); cam.start(); stop_cam = cam.stop
    elif cfg.camera.source == "ros" and cfg.output.ros_enable:
        cam = RosImageSource(cfg.output, cfg.camera); cam.start(); stop_cam = cam.stop
    else:
        print("[ERR] Invalid camera source."); return

    # ROS pub
    ros_pub = RosPublisher(cfg.output) if cfg.output.ros_enable else None

    # processor
    proc = Processor(cfg, cam, ros_pub)

    # 3D viewer
    viz = None
    if getattr(args, "viz3d", False):
        viz = SimplePoseVisualizer3D(window="Pose3D",
                                     fov_deg=getattr(args, "viz3d_fov", 55.0),
                                     img_size_hint=(cfg.camera.rgb_width, cfg.camera.rgb_height),
                                     K=proc.K)
        # For marker: side = marker_size; for diamond: side = 2*square_length
        marker_side_m = cfg.aruco.marker_size_mm / 1000.0
        obj_side_m = (marker_side_m
                      if cfg.aruco.detect_mode == "marker"
                      else 2.0 * cfg.aruco.diamond_square_mm / 1000.0)

    # TF rebroadcast loop
    def tf_loop():
        if not (cfg.output.tf_enable and ros_pub is not None):
            return
        period = 1.0 / max(1e-6, cfg.output.tf_broadcast_hz)
        parent, child = cfg.output.tf_parent, cfg.output.tf_child
        while True:
            if proc.last_pose is not None:
                t, q = proc.last_pose
                ros_pub.broadcast_tf(parent, child, t, q, stamp_sec=time.time())
            time.sleep(period)

    if cfg.output.tf_enable and ros_pub is not None:
        threading.Thread(target=tf_loop, daemon=True).start()

    throttle = max(0.0, cfg.runtime.throttle_sec)
    last_proc_time = 0.0

    try:
        if cfg.runtime.mode == "single":
            # read frames until we get one (or in RealSense we already warmed up)
            # If ROS: wait for first image then process once
            while True:
                rec = cam.read()
                if rec is None:
                    time.sleep(0.01); continue
                frame, stamp = rec
                last_frame_stamp = stamp
                if frame is None:
                    time.sleep(0.01); continue
                annotated, found_ids, found_corners, (t, q) = proc.process_frame(frame, stamp)
                if cfg.output.show_window:
                    cv2.imshow(cfg.output.window_name, annotated)
                    cv2.waitKey(10)
                # Publish image if ROS
                if ros_pub is not None and cfg.output.ros_out_image_topic:
                    ros_pub.publish_image(annotated, frame_id=cfg.output.tf_parent, stamp_sec=stamp)
                break

            # 3D viewer update
            if viz is not None and t is not None and q is not None:
                viz.set_detections(found_ids, found_corners, marker_side_m, main_id=cfg.aruco.marker_id)
                # First frame may define actual image size for frustum
                viz.set_image_size(annotated.shape[1], annotated.shape[0])
                viz.update_and_show(t, q, obj_side_m, title=cfg.aruco.detect_mode)

            print("[INFO] Single snapshot processed.")
            if cfg.output.show_window:
                print("[INFO] Press any key in the window to quit.")
                cv2.waitKey(0)

        else:  # continuous
            while True:
                rec = cam.read()
                if rec is None: time.sleep(0.005); continue
                frame, stamp = rec
                if (now() - last_proc_time) < throttle:
                    continue
                last_proc_time = now()

                annotated, found_ids, found_corners, (t, q) = proc.process_frame(frame, stamp)

                if cfg.output.show_window:
                    # 3D viewer update
                    if viz is not None and t is not None and q is not None:
                        viz.set_detections(found_ids, found_corners, marker_side_m)  # marker size, not obj_side_m
                        viz.set_image_size(annotated.shape[1], annotated.shape[0])
                        viz.update_and_show(t, q, obj_side_m, title=cfg.aruco.detect_mode)
                    cv2.imshow(cfg.output.window_name, annotated)
                    key = cv2.waitKey(1)
                    if key == 27 or key == ord('q'):
                        break

                if ros_pub is not None and cfg.output.ros_out_image_topic:
                    ros_pub.publish_image(annotated, frame_id=cfg.output.tf_parent, stamp_sec=stamp)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if cfg.output.show_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
