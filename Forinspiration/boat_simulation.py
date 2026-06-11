from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


def angle_difference(angle1: float, angle2: float) -> float:
    angle1 = float(angle1) % (2 * np.pi)
    angle2 = float(angle2) % (2 * np.pi)
    diff = angle1 - angle2
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return float(diff)


class PIDController:
    def __init__(self, Kp: float, Kd: float, Ki: float, error_fun: Callable[[float, float], float] = angle_difference):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_leakage = 0.8
        self.set_points = deque(maxlen=1)
        self.error_fun = error_fun

    def step(self, input_value: float, set_point: float, dt: float, output_min: float, output_max: float) -> float:
        if dt <= 0:
            dt = 1e-6

        self.set_points.append(set_point)
        averaged_set_point = float(np.mean(self.set_points))
        error = self.error_fun(averaged_set_point, input_value)
        self.integral = self.integral * self.integral_leakage + error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        pid_output = self.Kp * error + self.Kd * derivative + self.Ki * self.integral
        return float(np.clip(pid_output, output_min, output_max))

    def reset(self) -> None:
        self.prev_error = 0.0
        self.integral = 0.0


KNOT_TO_MS = 0.514444
DEFAULT_TIME_STEP = 0.1
DEFAULT_DEAD_ZONE_ANGLE_RAD = np.pi / 6

# ILCA 7 target speeds parameterized by true wind speed and true wind angle.
ILCA7_TWA_DEG = np.array([0, 30, 35, 45, 60, 75, 90, 110, 120, 135, 150, 160, 170, 180], dtype=float)
ILCA7_TWS_KTS = np.array([4, 6, 8, 10, 12, 14, 16, 20], dtype=float)
ILCA7_POLAR_SPEEDS_MS = np.array(
    [
        [0.0, 0.0, 1.8, 2.6, 3.2, 3.6, 3.8, 3.7, 3.6, 3.4, 3.1, 2.9, 2.8, 2.7],
        [0.0, 0.0, 2.3, 3.2, 3.8, 4.3, 4.6, 4.7, 4.8, 4.9, 4.8, 4.6, 4.3, 4.0],
        [0.0, 0.0, 2.6, 3.6, 4.4, 5.0, 5.5, 5.8, 6.0, 6.4, 6.2, 5.9, 5.6, 5.2],
        [0.0, 0.0, 2.8, 3.9, 4.8, 5.5, 6.0, 6.4, 6.8, 7.4, 7.2, 6.9, 6.5, 6.0],
        [0.0, 0.0, 3.0, 4.1, 5.0, 5.8, 6.3, 6.8, 7.2, 8.0, 8.0, 7.7, 7.2, 6.6],
        [0.0, 0.0, 3.1, 4.2, 5.1, 5.9, 6.4, 7.0, 7.5, 8.5, 8.6, 8.3, 7.8, 7.1],
        [0.0, 0.0, 3.2, 4.3, 5.2, 6.0, 6.6, 7.2, 7.8, 9.0, 9.2, 8.8, 8.3, 7.6],
        [0.0, 0.0, 3.3, 4.4, 5.3, 6.1, 6.8, 7.5, 8.2, 9.5, 9.8, 9.3, 8.7, 8.0],
    ],
    dtype=float,
) * KNOT_TO_MS


def wrap_phase(angles: float | np.ndarray) -> float | np.ndarray:
    return np.remainder(np.remainder(angles, 2 * np.pi) + 2 * np.pi, 2 * np.pi)


def mirror_angle_to_half_circle(angle: float | np.ndarray) -> float | np.ndarray:
    angle = wrap_phase(angle)
    return np.where(angle > np.pi, 2 * np.pi - angle, angle)


def ilca7_speed_from_true_wind(true_wind_speed: float, true_wind_angle: float) -> float:
    true_wind_speed_kts = float(true_wind_speed) / KNOT_TO_MS
    twa_deg = float(np.rad2deg(mirror_angle_to_half_circle(true_wind_angle)))
    speeds_at_angle = np.array(
        [np.interp(twa_deg, ILCA7_TWA_DEG, row) for row in ILCA7_POLAR_SPEEDS_MS],
        dtype=float,
    )
    base_speed = float(
        np.interp(
            true_wind_speed_kts,
            ILCA7_TWS_KTS,
            speeds_at_angle,
            left=speeds_at_angle[0],
            right=speeds_at_angle[-1],
        )
    )
    return base_speed


def best_upwind_true_wind_angle(true_wind_speed: float) -> float:
    candidate_angles_deg = np.linspace(35.0, 90.0, 112)
    candidate_angles_rad = np.deg2rad(candidate_angles_deg)
    speeds = np.array([ilca7_speed_from_true_wind(true_wind_speed, angle) for angle in candidate_angles_rad], dtype=float)
    vmg = speeds * np.cos(candidate_angles_rad)
    best_index = int(np.argmax(vmg))
    return float(candidate_angles_rad[best_index])


@dataclass
class BoatConfig:
    mass: float = 83.0
    length: float = 4.23
    beam: float = 1.37
    drag_coefficient: float = 0.003
    sail_area: float = 7.06
    water_density: float = 1025.0
    reference_area: float = 2.5
    max_rudder_angle_deg: float = 55.0
    rudder_rate_deg_s: float = 180.0
    speed_up_time_constant: float = 2.2
    speed_down_time_constant: float = 1.2
    yaw_damping: float = 10.0
    rudder_turn_coefficient: float = 28.0
    min_steerage_speed: float = 1.0
    dead_zone_angle_rad: float = DEFAULT_DEAD_ZONE_ANGLE_RAD

    @property
    def max_rudder_angle_rad(self) -> float:
        return float(np.deg2rad(self.max_rudder_angle_deg))

    @property
    def rudder_rate_rad_s(self) -> float:
        return float(np.deg2rad(self.rudder_rate_deg_s))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> "BoatConfig":
        if not data:
            return cls()
        return cls(**data)


@dataclass
class BoatState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    angular_velocity: float = 0.0
    rudder_angle: float = 0.0

    def copy(self) -> "BoatState":
        return BoatState(**asdict(self))


class Boat:
    def __init__(self, config: BoatConfig | None = None, initial_state: BoatState | None = None):
        self.config = config or BoatConfig()
        self.moment_of_inertia = self.estimate_moment_of_inertia()
        self.state = initial_state.copy() if initial_state is not None else BoatState()

    @property
    def x(self) -> float:
        return self.state.x

    @property
    def y(self) -> float:
        return self.state.y

    @property
    def heading(self) -> float:
        return self.state.heading

    @property
    def speed(self) -> float:
        return self.state.speed

    @property
    def angular_velocity(self) -> float:
        return self.state.angular_velocity

    @property
    def rudder_angle(self) -> float:
        return self.state.rudder_angle

    def estimate_moment_of_inertia(self) -> float:
        return float(self.config.mass * (self.config.length**2 + self.config.beam**2) / 12.0)

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        speed: float = 0.0,
        angular_velocity: float = 0.0,
        rudder_angle: float = 0.0,
    ) -> BoatState:
        self.state = BoatState(
            x=float(x),
            y=float(y),
            heading=float(wrap_phase(heading)),
            speed=max(0.0, float(speed)),
            angular_velocity=float(angular_velocity),
            rudder_angle=float(np.clip(rudder_angle, -self.config.max_rudder_angle_rad, self.config.max_rudder_angle_rad)),
        )
        return self.get_state_snapshot()

    def set_state(self, state: BoatState) -> None:
        self.state = state.copy()
        self.state.heading = float(wrap_phase(self.state.heading))
        self.state.speed = max(0.0, float(self.state.speed))
        self.state.rudder_angle = float(
            np.clip(self.state.rudder_angle, -self.config.max_rudder_angle_rad, self.config.max_rudder_angle_rad)
        )

    def step(self, wind_speed: float, wind_angle: float, rudder_delta_command: float, dt: float = DEFAULT_TIME_STEP) -> BoatState:
        dt = max(float(dt), 1e-6)
        self.apply_rudder_delta(rudder_delta_command, dt)
        self.update_speed(wind_speed, wind_angle, dt)
        self.update_heading(dt)
        self.update_position(dt)
        return self.get_state_snapshot()

    def apply_rudder_delta(self, rudder_delta_command: float, dt: float) -> float:
        command = float(np.clip(rudder_delta_command, -1.0, 1.0))
        delta = command * self.config.rudder_rate_rad_s * dt
        self.state.rudder_angle = float(
            np.clip(
                self.state.rudder_angle + delta,
                -self.config.max_rudder_angle_rad,
                self.config.max_rudder_angle_rad,
            )
        )
        return self.state.rudder_angle

    def update_speed(self, wind_speed: float, wind_angle: float, dt: float) -> float:
        true_wind_angle = wrap_phase(wind_angle - self.state.heading)
        target_speed = self.get_speed_from_polar_chart(wind_speed, true_wind_angle, self.config.dead_zone_angle_rad)
        time_constant = self.config.speed_up_time_constant if target_speed >= self.state.speed else self.config.speed_down_time_constant
        acceleration = (target_speed - self.state.speed) / max(time_constant, 1e-6)
        self.state.speed = max(0.0, float(self.state.speed + acceleration * dt))
        return self.state.speed

    def update_heading(self, dt: float) -> float:
        turning_torque = self.calculate_turning_torque(self.state.rudder_angle, self.state.speed)
        angular_acceleration = (turning_torque - self.config.yaw_damping * self.state.angular_velocity) / max(
            self.moment_of_inertia,
            1e-6,
        )
        self.state.angular_velocity += float(angular_acceleration * dt)
        self.state.heading = float(wrap_phase(self.state.heading + self.state.angular_velocity * dt))
        return self.state.heading

    def update_position(self, dt: float) -> tuple[float, float]:
        self.state.x += float(self.state.speed * np.sin(self.state.heading) * dt)
        self.state.y += float(self.state.speed * np.cos(self.state.heading) * dt)
        return self.state.x, self.state.y

    def get_speed_from_polar_chart(self, true_wind_speed: float, true_wind_angle: float, dead_zone_angle: float) -> float:
        twa = float(mirror_angle_to_half_circle(true_wind_angle))
        if twa < dead_zone_angle:
            return 0.0
        return ilca7_speed_from_true_wind(true_wind_speed, twa)

    def calculate_propulsive_force(self, apparent_wind_speed: float) -> float:
        return float(self.config.sail_area * apparent_wind_speed**2)

    def calculate_drag_force(self, boat_speed: float) -> float:
        return float(
            0.5
            * self.config.water_density
            * self.config.drag_coefficient
            * self.config.reference_area
            * boat_speed**2
        )

    def calculate_turning_torque(self, rudder_angle: float, speed: float) -> float:
        effective_speed = max(float(speed), float(self.config.min_steerage_speed))
        return float(self.config.rudder_turn_coefficient * rudder_angle * effective_speed)

    def calculate_apparent_wind(
        self,
        true_wind_speed: float,
        true_wind_direction: float,
        boat_speed: float | None = None,
        boat_heading: float | None = None,
    ) -> tuple[float, float]:
        speed = self.state.speed if boat_speed is None else float(boat_speed)
        heading = self.state.heading if boat_heading is None else float(boat_heading)
        wind_vector = true_wind_speed * np.array([np.sin(true_wind_direction), np.cos(true_wind_direction)])
        boat_vector = speed * np.array([np.sin(heading), np.cos(heading)])
        apparent_wind_vector = wind_vector + boat_vector
        apparent_wind_speed = float(np.linalg.norm(apparent_wind_vector))
        apparent_wind_angle_world = float(np.arctan2(apparent_wind_vector[0], apparent_wind_vector[1]))
        apparent_wind_angle = float(wrap_phase(apparent_wind_angle_world - heading))
        return apparent_wind_speed, apparent_wind_angle

    def get_state_snapshot(self) -> BoatState:
        return self.state.copy()
