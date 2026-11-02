#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================
#changelogs

#25.10.31
# **"직선-코너 비례계수(Factor)"** 를 도입하여, 코너진입/직진 상황 등에 맞추어 동적으로 max_speed를 조절하는 기능 추가
# forza_final 이미지에 맞게 리팩토링 작업 진행

#25.11.1
# factor 로깅 추가
# 최종 속도 계산된 속도 로깅 추가
# factor_raw 대신 factor를 그냥 사용. 이미 정규화가 된거로 추정

#25.11.2
# 동적 스티어링 상한 로직 도입

#25.11.3
# 선형 보간 -> 초월 함수 형 보간법 사용 (감소 지수 함수를 적용)

# =======================================================================================

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import time
import math
import rclpy
from rclpy.node import Node

import numpy as np
import tensorflow as tf

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

# ===================== 기본 튜닝값 =====================
DOWN_SAMPLE = 2
EMA_ALPHA = 0.7
STEERING_DEADZONE_RAD = 0.0

CURVE_SPEED_WEIGHT = 0.65 # 
STEER_ABS_MAX = 0.36 

# =========================
# for dyanamic max speed caculations [need param tuning]

# =========================
#param changelog

#25.10.31 [firstday]
# fov 40 -> 20 : make it see more frontside :I THINK IT IS WORKING BETTER
# v_min 2.0 -> 3.0 -> 0.5 : lab std's advice (3.0 was doing good)
# v_max 6.0 -> 8.0 -> 10.0 -> 12.0 -> 15.0 -> 20.0
# factor_zero_dist 3.0 -> 4.0 : it is too fast now. so early braking
# one_dist 10.0

#25.11.1
# factor_zero_dist 4.0 -> 3.0 -> 2.0 -> 3.0 -> 4.0 -> 5.0 (멀리있는데도 factor 0되버림) -> 3.0 -> 4.0 -> 3.5 -> 4.0
# fov : 20 -> 30 -> 25
# one_dist : 10.0 -> 8.0 -> 10.0 -> 12.0 -> 15.0 -> 12.0 -> 7.0 -> 9.0 -> 12.0 -> 10.0
# v_min 0.5 -> 1.0 -> 2.0 -> 0.5 -> 2.0 -> 3.0 -> 0.5 -> 2.0 -> 2.5 -> 3.0(직선 구간에서 털림) -> 2.8(털림) -> 2.5
# v_max 20.0 -> 30.0

#25.11.2
# ONE_STEER_MIN_V 5.0 -> 4.0(직선, 장애물 구간 다 돔) -> 4.5(time_trial 최적) -> 4.3 -> 4.5
# ZERO_STEER_MAX_V 20.0 -> 15.0(time_trial 최적) -> 20 -> 18.0 -> 23.0 -> 21.0
# =========================

V_MIN = 2.5
V_MAX = 30.0

FACTOR_ZERO_DIST = 4.0 # factor will be zero if the mean_dist falls under this one
FACTOR_ONE_DIST = 10.0 # factor will be one if the mean_dist falls over this one

FACTOR_FOV_DEG_ = 25.0 # lidar scan angle to use on mean_distance of all laser beams.
# big angle will affect aside wall to take effect on "factor" (not desired)
# hopeful to use on front-side laser beams

ONE_STEER_MIN_V = 4.5 # 속도가 이거보다 작으면, 100퍼센트 조향 그대로 허용 (조향팩터==1)
ZERO_STEER_MAX_V = 21.0 # 속도가 이거보다 크면, 조향 미허용 (조향팩터==0)

K_FOR_EXP_DECAY = 0.35  # 조향 디케이 지수함수 계수

# =========================

# =====================================================

class TLNInference(Node):
    def __init__(self):
        super().__init__('tln_inference')

        # ---------- TensorFlow 런타임 튜닝 ----------
        # 과도한 CPU 스레드 사용 억제 (장비 코어 수에 맞춰 조절)
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except Exception:
            pass
        # GPU 사용 시 메모리 증가 허용
        try:
            for d in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(d, True)
        except Exception:
            pass
        # -------------------------------------------

        # 모델 로드
        pkg_share = get_package_share_directory('ofc')

        #use orig model
        #self.model_path = str(Path(pkg_share) / 'models' / 'f2_f4_silverstone_7lap.keras')

        #use finetune model
        self.model_path = str(Path(pkg_share) / 'models' / 'tinylidarnet_finetuned_1103.keras')

        self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)
        self.get_logger().info("모델 로드 완료 - 백엔드 초기화")

        # ---------- 파라미터 선언 ----------
        self.declare_parameter("drive_publish_enabled", False)

        self.declare_parameters('', [
            ('scan_topic', '/scan'),
            ('drive_topic', '/drive'),

            ('down_sample', DOWN_SAMPLE),
            ('ema_alpha', EMA_ALPHA),
            ('steering_deadzone_rad', STEERING_DEADZONE_RAD),

            # 런타임 좌우반전 스위치(입력 뒤집고, 조향 부호 반전)
            ('reverse_direction', False),

            ('curve_speed_weight', CURVE_SPEED_WEIGHT),
            ('v_min', V_MIN),
            ('v_max', V_MAX),
            ('steer_abs_max', STEER_ABS_MAX),

            # 매핑 모드 스위치
            ('mapping', False),

            # 라이다 값 클램핑 상한 (기본 10.0 m)
            ('clamp_range_max_m', 10.0),
        ])

        # 값 로드
        self.drive_publish_enabled = bool(self.get_parameter("drive_publish_enabled").value)
        self.scan_topic            = str(self.get_parameter('scan_topic').value)
        self.drive_topic           = str(self.get_parameter('drive_topic').value)
        self.down_sample           = int(self.get_parameter('down_sample').value)
        self.ema_alpha             = float(self.get_parameter('ema_alpha').value)
        self.steering_deadzone_rad = float(self.get_parameter('steering_deadzone_rad').value)

        self.reverse_direction     = bool(self.get_parameter('reverse_direction').value)

        self.curve_speed_weight    = float(self.get_parameter('curve_speed_weight').value)
        self.v_min                 = float(self.get_parameter('v_min').value)
        self.v_max                 = float(self.get_parameter('v_max').value)
        self.steer_abs_max         = float(self.get_parameter('steer_abs_max').value)

        self.mapping               = bool(self.get_parameter('mapping').value)
        self.clamp_range_max_m     = float(self.get_parameter('clamp_range_max_m').value)

        # --- [ 이 부분 추가 ] ---------------------------------------------------------------------------
        # "비례계수" 튜닝을 위한 파라미터 3개 로드
        self.FACTOR_ZERO_DIST_ = FACTOR_ZERO_DIST
        self.FACTOR_ONE_DIST_ = FACTOR_ONE_DIST
        self.FACTOR_FOV_DEG_ = FACTOR_FOV_DEG_
        self.FACTOR_FOV_RAD_ = np.deg2rad(self.FACTOR_FOV_DEG_).astype(np.float32)

        # 매 순간 갱신될 동적 속도 변수 (v_min으로 초기화)
        self.dynamic_v_max_ = self.v_min
        # 비례계수 계산에 사용할 각도 배열 (미리 생성)
        self.ANGLES_1081_ = np.linspace(-np.deg2rad(135), np.deg2rad(135), 1081, dtype=np.float32)
        # --- [ 추가 끝 ] ---

        #----------------------------------------------------------
        # for 동적 조향 상한

        self.ONE_STEER_MIN_V = ONE_STEER_MIN_V # 속도가 이거보다 작으면, 100퍼센트 조향 그대로 허용 (조향팩터==1)
        self.ZERO_STEER_MAX_V = ZERO_STEER_MAX_V # 속도가 이거보다 크면, 조향 미허용 (조향팩터==0)

        #for exp decay
        self.k_for_exp_dacay = K_FOR_EXP_DECAY

        self.dynamic_steer_max = self.steer_abs_max

        #----------------------------------------------------------

        # 변경 콜백 등록
        self.add_on_set_parameters_callback(self._on_params)
        # ----------------------------------------------------------------------------------------------

        # ===== 최적화를 위한 사전 버퍼/그래프 준비 =====
        # 1081 길이 버퍼 (float32 고정)
        self._scan_buf = np.empty((1081,), dtype=np.float32)

        # downsample 입력 길이에 맞춰 TF 함수 빌드
        self._build_tf_function(self.down_sample)

        # EMA 초기값, 로그 스로틀용 타임스탬프
        self.steer_ema = 0.0
        self._last_log = 0.0
        # ============================================

        # 구독/퍼블리셔
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 5)
        self.pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 5)

        # 1회 워밍업(옵션: 그래프/커널 예열)
        try:
            dummy = np.zeros((1, self._in_len, 1), dtype=np.float32)
            _ = self._infer(tf.convert_to_tensor(dummy, dtype=tf.float32))
        except Exception:
            pass

    # ====== 그래프 고정/재빌드 ======
    def _build_tf_function(self, ds: int):
        ds = max(1, int(ds))
        # 1081 포인트를 ds 간격으로 샘플링했을 때 길이
        self._in_len = math.ceil(1081 / ds)
        self._ds = ds

        # 입력 시그니처 고정
        self._tf_spec = tf.TensorSpec(shape=(1, self._in_len, 1), dtype=tf.float32)

        # tf.function으로 경계/서명 고정
        @tf.function(input_signature=[self._tf_spec])
        def _infer(x):
            return self.model(x, training=False)

        self._infer = _infer
        self.get_logger().info(f"[tln_inference] tf.function build: in_len={self._in_len}, ds={self._ds}")
    # =================================

    # ====== 유틸 ======
    @staticmethod
    def linear_map(x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def _fill_scan_buf_1081(self, rng: np.ndarray):
        """
        rng: float32, 길이 1080 또는 1081
        self._scan_buf(float32,1081)에 채워서 반환 (새 할당 없음)
        """
        if rng.shape[0] == 1081:
            # 그대로 복사(사전 버퍼로)
            self._scan_buf[:] = rng
        else:
            # 1080 → 1081 보정
            self._scan_buf[:-1] = rng
            self._scan_buf[-1] = rng[-1]
        return self._scan_buf
    # ===============

    #--------------------------------------------------------------------------------
    # 동적 조향 상한 계산용 함수 (지수 보간법)
    def caculate_dynamic_maxium_steer_amount(self, speed):
        """
        속도에 따라 조향 허용량을 감소 지수 함수로 보간.
        - ONE_STEER_MIN_V 이하: steering_factor = 1.0
        - ZERO_STEER_MAX_V 이상: steering_factor = 0.0
        - 그 사이는 exp() 기반의 감소 곡선 적용
        """

        # 속도가 한계 밖이면 단순 클립
        if speed <= self.ONE_STEER_MIN_V:
            steering_factor = 1.0
        elif speed >= self.ZERO_STEER_MAX_V:
            steering_factor = 0.0

        else:
            # 속도를 0~1로 정규화
            norm = (speed - self.ONE_STEER_MIN_V) / (self.ZERO_STEER_MAX_V - self.ONE_STEER_MIN_V)
            norm = np.clip(norm, 0.0, 1.0)

            # 감소 지수 함수 적용 (감쇠율 k는 튜닝 파라미터)
            k = self.k_for_exp_dacay # <- 조향 감쇠 민감도 (값이 클수록 초반 급격히 줄어듦)
            steering_factor = np.exp(-k * norm)

        # 동적 최대 조향각 계산
        return (self.steer_abs_max * steering_factor)

    #---------------------------------------------------------------------------------

    def scan_cb(self, msg: LaserScan):
        t0 = time.time()

        # ranges → float32 뷰/카피
        rng = np.asarray(msg.ranges, dtype=np.float32)

        # in-place NaN/Inf 처리 + 클램프
        cap = np.float32(self.clamp_range_max_m if self.clamp_range_max_m > 0.0 else 10.0)
        np.nan_to_num(rng, nan=cap, posinf=cap, neginf=np.float32(0.0), copy=False)
        np.clip(rng, np.float32(0.0), cap, out=rng)

        # 1081 채우기(사전 버퍼)
        scan1081 = self._fill_scan_buf_1081(rng)

        # --- [ 이 부분 추가 ] -----------------------------------------------------------------------
        # "연속적 비례계수 (Factor)" 계산 로직
        try:
            # 1. 전방 시야각(FOV) 내의 유효한 인덱스 필터링
            half_fov = self.FACTOR_FOV_RAD_ / 2.0
            front_indices = np.where(
                (self.ANGLES_1081_ > -half_fov) & 
                (self.ANGLES_1081_ < half_fov)
            )[0]

            if len(front_indices) == 0:
                avg_dist_ahead = 0.0 # 전방에 유효 스캔 없음
            else:
                # 2. 전방 FOV 내의 '평균' 거리 계산 (scan1081 버퍼 사용)
                avg_dist_ahead = np.mean(scan1081[front_indices])

            #debugg..
            #self.get_logger().info(f'[inference_node.py] avg_distance_ahead caculated : {avg_dist_ahead} m')

            # 3. '비례계수' 계산 (선형 보간) [factor_raw]
            factor = ((avg_dist_ahead - self.FACTOR_ZERO_DIST_) / 
                          (self.FACTOR_ONE_DIST_ - self.FACTOR_ZERO_DIST_))
            
            # 4. 0.0 ~ 1.0 사이의 연속적인 값으로 클리핑
            # factor = np.clip(factor_raw, 0.0, 1.0)

            #debugg..
            #self.get_logger().info(f'[inference_node.py] factor caculated : {factor}')
            
            # 5. '동적 최대 속도' 갱신
            self.dynamic_v_max_ = max(self.v_max * factor, self.v_min)
            
        except Exception as e:
            self.get_logger().error(f'Factor calculation error: {e}')
            self.dynamic_v_max_ = self.v_min # 오류 시 즉시 최저 속도로 제한
        # --- [ 추가 로직 끝 ] --------------------------------------------------------------------------

        # 좌우 반전(in-place)
        if self.reverse_direction:
            np.flip(scan1081, out=scan1081)

        # downsample 뷰 → (1, L, 1) reshape (뷰 기반, 새 할당 없음)
        arr = scan1081[::self._ds].reshape(1, self._in_len, 1)

        # 추론 (tf.function 경로, dtype/shape 고정)
        y = self._infer(tf.convert_to_tensor(arr, dtype=tf.float32))
        # 모델이 [steer, speed] 1x2 형태로 준다고 가정
        y_np = y.numpy()[0]  # 1D
        steering_raw = float(y_np[0])
        speed_raw    = float(y_np[1])

        # 역방향이면 조향만 부호 반전
        if self.reverse_direction:
            steering_raw = -steering_raw

        # ====== 보정(EMA/데드존/커브시 속도 감쇠/클립) ======
        alpha = self.ema_alpha
        self.steer_ema = (1.0 - alpha) * self.steer_ema + alpha * steering_raw
        steering = self.steer_ema

        if abs(steering) < self.steering_deadzone_rad:
            steering = 0.0

        # [ 1번 수정 ]
        model_speed = self.linear_map(speed_raw, -1.0, 1.0, self.v_min, self.dynamic_v_max_)

        turn_factor = min(abs(steering) / self.steer_abs_max, 1.0)
        curve_damping_ratio = 1.0 - turn_factor
        final_speed = model_speed * curve_damping_ratio
        speed = (1.0 - self.curve_speed_weight) * model_speed + self.curve_speed_weight * final_speed

        # [ 동적 속도 상한 도입 ]---------------------------------------------------------
        speed    = float(np.clip(speed,    self.v_min,          self.dynamic_v_max_)) #FINAL SPEED
        #----------------------------------------------------------------------------

        # 동적 조향 상한 도입-------------------------------------------------------------
        self.dynamic_steer_max = self.caculate_dynamic_maxium_steer_amount(speed)
        steering = float(np.clip(steering, -self.dynamic_steer_max, self.dynamic_steer_max)) #FINAL STEER
        #-----------------------------------------------------------------------------

        # 동적 조향 상한 미사용 시
        #steering = float(np.clip(steering, -self.steer_abs_max, self.steer_abs_max)) #FINAL STEER <= 활성화

        #로깅
        self.get_logger().info(f'[inference_node.py] FINAL SPEED : {speed} m/s')
        self.get_logger().info(f'[inference_node.py] FINAL STEER : {steering}')

        # 매핑 모드 시 속도 제한
        if self.mapping:
            if speed < 1.0: speed = 1.0
            if speed > 3.0: speed = 3.0
        # ===================================================

        # 퍼블리시
        if self.drive_publish_enabled:
            out = AckermannDriveStamped()
            out.header.stamp = msg.header.stamp
            out.drive.steering_angle = steering
            out.drive.speed = speed
            self.pub.publish(out)

        # 스로틀 로그(1초에 한 번)
        now = time.time()
        if now - self._last_log > 1.0:
            self._last_log = now
            took_ms = (now - t0) * 1000.0
            self.get_logger().info(f"Servo: {steering:.4f}, Speed: {speed:.3f} m/s | Took: {took_ms:.2f} ms")

    # 파라미터 변경 콜백
    def _on_params(self, params):
        rebuild_tf = False
        for p in params:
            if p.name == "drive_publish_enabled" and p.type_ == Parameter.Type.BOOL:
                self.drive_publish_enabled = bool(p.value)

            elif p.name == 'scan_topic':
                self.scan_topic = str(p.value)  # 런타임 교체는 재시작 권장

            elif p.name == 'drive_topic':
                self.drive_topic = str(p.value) # 런타임 교체는 재시작 권장

            elif p.name == 'down_sample':
                new_ds = max(1, int(p.value))
                if new_ds != self._ds:
                    self.down_sample = new_ds
                    rebuild_tf = True

            elif p.name == 'ema_alpha':
                self.ema_alpha = float(p.value)

            elif p.name == 'steering_deadzone_rad':
                self.steering_deadzone_rad = float(p.value)

            elif p.name == 'reverse_direction':
                self.reverse_direction = bool(p.value)

            elif p.name == 'curve_speed_weight':
                self.curve_speed_weight = float(p.value)

            elif p.name == 'v_min':
                self.v_min = float(p.value)

            elif p.name == 'v_max':
                self.v_max = float(p.value)

            elif p.name == 'steer_abs_max':
                self.steer_abs_max = float(p.value)

            elif p.name == 'mapping':
                self.mapping = bool(p.value)
                self.get_logger().info(f"[tln_inference] mapping = {self.mapping}")

            elif p.name == 'clamp_range_max_m':
                self.clamp_range_max_m = float(p.value)
                self.get_logger().info(f"[tln_inference] clamp_range_max_m = {self.clamp_range_max_m:.2f} m")

        if rebuild_tf:
            # downsample 변경 시 입력 길이가 달라지므로 tf.function 재빌드
            self._build_tf_function(self.down_sample)
            # 재빌드 직후 짧은 워밍업(옵션)
            try:
                dummy = np.zeros((1, self._in_len, 1), dtype=np.float32)
                _ = self._infer(tf.convert_to_tensor(dummy, dtype=tf.float32))
            except Exception:
                pass

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = TLNInference()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
