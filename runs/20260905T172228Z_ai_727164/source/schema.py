from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Position(StrictModel):
    x_m: float = Field(ge=-0.20, le=0.20, description="Forward from frame center, meters")
    y_m: float = Field(ge=-0.15, le=0.15, description="Left from frame center, meters")


class Design(StrictModel):
    frame_width_m: float = Field(ge=0.18, le=0.32, description="Outer frame edges, lateral; wheels outside")
    frame_length_m: float = Field(ge=0.25, le=0.42, description="Outer frame edges, longitudinal")
    wheelbase_m: float = Field(ge=0.14, le=0.32, description="Front to rear wheel center distance")
    track_width_m: float = Field(ge=0.23, le=0.40, description="Left to right wheel center distance")
    wheel_diameter_m: Literal[0.08255, 0.1016, 0.104775]
    gearing_rpm: Literal[100, 200, 600]
    battery: Position
    ballast: Position | None = Field(description="Null omits ballast; otherwise fixed 0.30 kg block")

    @model_validator(mode="after")
    def geometry(self):
        if self.track_width_m < self.frame_width_m + 0.035:
            raise ValueError("track_width_m must be >= frame_width_m + 0.035 (25 mm wheel + 5 mm clearance per side)")
        if self.wheelbase_m < self.wheel_diameter_m + 0.02:
            raise ValueError("wheelbase must leave 20 mm between front and rear tires")
        if self.wheelbase_m > self.frame_length_m - 0.04:
            raise ValueError("wheel centers must lie >=20 mm inside frame ends")
        if max(self.frame_length_m, self.wheelbase_m + self.wheel_diameter_m) > 0.4572 or self.track_width_m + 0.025 > 0.4572:
            raise ValueError("overall tire/frame footprint exceeds 0.4572 m square engineering envelope")
        blocks = [("battery", self.battery, 0.12, 0.065)]
        if self.ballast is not None:
            blocks.append(("ballast", self.ballast, 0.05, 0.05))
        for name, p, lx, ly in blocks:
            if abs(p.x_m) + lx/2 + 0.005 > self.frame_length_m/2 or abs(p.y_m) + ly/2 + 0.005 > self.frame_width_m/2:
                raise ValueError(f"{name} must fit on frame with 5 mm edge clearance")
        if self.ballast is not None:
            if abs(self.battery.x_m-self.ballast.x_m) < 0.09 and abs(self.battery.y_m-self.ballast.y_m) < 0.0625:
                raise ValueError("battery and ballast require 5 mm separation; move ballast or use null")
        return self

    @property
    def frame_mass_kg(self):
        return 0.65 + 2.0 * (self.frame_length_m + self.frame_width_m)

    @property
    def total_mass_kg(self):
        return self.frame_mass_kg + 4*0.12 + 4*0.28 + 0.35 + (0.30 if self.ballast else 0)

    @property
    def stall_torque_nm(self):
        return 1.05 * 200 / self.gearing_rpm


class Proposal(StrictModel):
    design: Design
    rationale: str = Field(min_length=1, max_length=1200)
    predicted_benefit: str = Field(min_length=1, max_length=800)
    hypothesis: str = Field(min_length=1, max_length=800)


class Trial(StrictModel):
    seed: int
    friction: float
    initial_pose: list[float]
    completed: bool
    completion_time_s: float | None
    elapsed_s: float
    waypoint_progress: float
    waypoints_reached: int
    collision_episodes: int
    rms_tracking_error_m: float
    parking_position_error_m: float
    parking_heading_error_rad: float
    unstable: bool
    failures: list[str]
    wheel_travel_m: list[float]
    saturation_fraction: float
    distance_traveled_m: float
    peak_speed_m_s: float
    braking_stop_time_s: float | None


def sample_design(revised=False):
    """Handwritten fixtures, never AI-generated."""
    return Design(frame_width_m=0.25, frame_length_m=0.34,
                  wheelbase_m=0.20 if revised else 0.27,
                  track_width_m=0.30, wheel_diameter_m=0.1016,
                  gearing_rpm=200, battery=Position(x_m=0, y_m=0), ballast=None)
