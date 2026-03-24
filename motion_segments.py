from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(
        int(window), center=True, min_periods=1
    ).mean().to_numpy()


def _first_motion_rotation(
    vel_xy_mps: np.ndarray,
    speed_mps: np.ndarray,
    speed_threshold_mps: float,
) -> np.ndarray:
    speed_mps = np.asarray(speed_mps, dtype=float)
    vel_xy_mps = np.asarray(vel_xy_mps, dtype=float)

    active_idx = np.flatnonzero(speed_mps > float(speed_threshold_mps))
    if len(active_idx) == 0:
        return np.eye(2)

    center_idx = int(active_idx[0])
    start_idx = max(0, center_idx - 5)
    end_idx = min(len(speed_mps), center_idx + 6)
    first_vec = np.nanmedian(vel_xy_mps[start_idx:end_idx], axis=0)

    norm = np.linalg.norm(first_vec)
    if not np.isfinite(norm) or norm < 1e-8:
        return np.eye(2)

    axis_1 = first_vec / norm
    axis_2 = np.array([-axis_1[1], axis_1[0]], dtype=float)
    return np.column_stack([axis_1, axis_2])


def add_motion_plane_kinematics(
    opti_df: pd.DataFrame,
    time_col: str = "time_synced_to_imu",
    pos_smooth_samples: int = 241,
    vel_smooth_samples: int = 121,
    speed_threshold_mps: float = 0.08,
):
    out = opti_df.copy()

    xyz = out[["X", "Y", "Z"]].to_numpy(dtype=float)
    center = np.nanmean(xyz, axis=0)
    xyz0 = xyz - center

    _, _, vt = np.linalg.svd(xyz0, full_matrices=False)
    plane_axes = vt[:2].copy()
    plane_xy_mm = xyz0 @ plane_axes.T

    time_s = out[time_col].to_numpy(dtype=float)
    plane_u_smooth_mm = _rolling_mean(plane_xy_mm[:, 0], pos_smooth_samples)
    plane_v_smooth_mm = _rolling_mean(plane_xy_mm[:, 1], pos_smooth_samples)
    vel_u_mps = np.gradient(plane_u_smooth_mm, time_s) / 1000.0
    vel_v_mps = np.gradient(plane_v_smooth_mm, time_s) / 1000.0
    vel_xy_mps = np.column_stack([vel_u_mps, vel_v_mps])
    speed_mps = np.linalg.norm(vel_xy_mps, axis=1)

    rotation = _first_motion_rotation(
        vel_xy_mps=vel_xy_mps,
        speed_mps=speed_mps,
        speed_threshold_mps=speed_threshold_mps,
    )
    plane_xy_mm = plane_xy_mm @ rotation
    plane_axes = rotation.T @ plane_axes

    plane_u_mm = plane_xy_mm[:, 0]
    plane_v_mm = plane_xy_mm[:, 1]
    plane_u_smooth_mm = _rolling_mean(plane_u_mm, pos_smooth_samples)
    plane_v_smooth_mm = _rolling_mean(plane_v_mm, pos_smooth_samples)

    vel_u_mps = np.gradient(plane_u_smooth_mm, time_s) / 1000.0
    vel_v_mps = np.gradient(plane_v_smooth_mm, time_s) / 1000.0
    vel_u_smooth_mps = _rolling_mean(vel_u_mps, vel_smooth_samples)
    vel_v_smooth_mps = _rolling_mean(vel_v_mps, vel_smooth_samples)
    speed_smooth_mps = np.sqrt(vel_u_smooth_mps ** 2 + vel_v_smooth_mps ** 2)
    heading_deg = np.degrees(np.arctan2(vel_v_smooth_mps, vel_u_smooth_mps))

    plane_normal = np.cross(plane_axes[0], plane_axes[1])

    out["motion_plane_u_mm"] = plane_u_mm
    out["motion_plane_v_mm"] = plane_v_mm
    out["motion_plane_u_smooth_mm"] = plane_u_smooth_mm
    out["motion_plane_v_smooth_mm"] = plane_v_smooth_mm
    out["motion_plane_vel_u_mps"] = vel_u_smooth_mps
    out["motion_plane_vel_v_mps"] = vel_v_smooth_mps
    out["motion_plane_speed_mps"] = np.sqrt(vel_u_mps ** 2 + vel_v_mps ** 2)
    out["motion_plane_speed_smooth_mps"] = speed_smooth_mps
    out["motion_heading_deg"] = heading_deg

    out["motion_plane_axis_1_x"] = plane_axes[0, 0]
    out["motion_plane_axis_1_y"] = plane_axes[0, 1]
    out["motion_plane_axis_1_z"] = plane_axes[0, 2]
    out["motion_plane_axis_2_x"] = plane_axes[1, 0]
    out["motion_plane_axis_2_y"] = plane_axes[1, 1]
    out["motion_plane_axis_2_z"] = plane_axes[1, 2]
    out["motion_plane_normal_x"] = plane_normal[0]
    out["motion_plane_normal_y"] = plane_normal[1]
    out["motion_plane_normal_z"] = plane_normal[2]
    out["motion_plane_center_x"] = center[0]
    out["motion_plane_center_y"] = center[1]
    out["motion_plane_center_z"] = center[2]

    return out, plane_axes, plane_normal, center


def extract_motion_segments(
    opti_df: pd.DataFrame,
    time_col: str = "time_synced_to_imu",
    speed_col: str = "motion_plane_speed_smooth_mps",
    pos_cols: tuple[str, str] = ("motion_plane_u_smooth_mm", "motion_plane_v_smooth_mm"),
    min_speed_mps: float = 0.08,
    min_duration_sec: float = 3.0,
) -> pd.DataFrame:
    active = opti_df[speed_col].to_numpy(dtype=float) > float(min_speed_mps)
    change = np.diff(np.r_[False, active, False].astype(int))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    rows: list[dict[str, float | int | str]] = []

    for start_idx, end_idx in zip(starts, ends):
        start_time_s = float(opti_df.iloc[start_idx][time_col])
        end_time_s = float(opti_df.iloc[end_idx][time_col])
        duration_sec = end_time_s - start_time_s
        if duration_sec < float(min_duration_sec):
            continue

        du_mm = float(opti_df.iloc[end_idx][pos_cols[0]] - opti_df.iloc[start_idx][pos_cols[0]])
        dv_mm = float(opti_df.iloc[end_idx][pos_cols[1]] - opti_df.iloc[start_idx][pos_cols[1]])
        disp_xyz_mm = (
            opti_df.iloc[end_idx][["X", "Y", "Z"]].to_numpy(dtype=float)
            - opti_df.iloc[start_idx][["X", "Y", "Z"]].to_numpy(dtype=float)
        )
        segment_distance_m = float(np.hypot(du_mm, dv_mm) / 1000.0)
        heading_deg = float(np.degrees(np.arctan2(dv_mm, du_mm)))

        primary_axis_id = 1 if abs(du_mm) >= abs(dv_mm) else 2
        primary_axis_disp_mm = du_mm if primary_axis_id == 1 else dv_mm
        direction_sign = 1 if primary_axis_disp_mm >= 0 else -1
        direction_label = (
            f"axis_{primary_axis_id}_positive"
            if direction_sign > 0
            else f"axis_{primary_axis_id}_negative"
        )

        rows.append(
            {
                "segment_id": len(rows) + 1,
                "direction_sign": direction_sign,
                "direction_label": direction_label,
                "primary_axis_id": primary_axis_id,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_frame": int(opti_df.iloc[start_idx]["frame"]),
                "end_frame": int(opti_df.iloc[end_idx]["frame"]),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_sec": duration_sec,
                "heading_deg": heading_deg,
                "segment_distance_m": segment_distance_m,
                "segment_displacement_u_m": du_mm / 1000.0,
                "segment_displacement_v_m": dv_mm / 1000.0,
                "segment_displacement_x_m": float(disp_xyz_mm[0] / 1000.0),
                "segment_displacement_y_m": float(disp_xyz_mm[1] / 1000.0),
                "segment_displacement_z_m": float(disp_xyz_mm[2] / 1000.0),
                "median_motion_speed_mps": float(
                    np.nanmedian(opti_df.iloc[start_idx : end_idx + 1][speed_col])
                ),
                "max_motion_speed_mps": float(
                    np.nanmax(opti_df.iloc[start_idx : end_idx + 1][speed_col])
                ),
            }
        )

    return pd.DataFrame(rows)
