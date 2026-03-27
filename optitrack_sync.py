from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import correlate

from motion_segments import add_motion_plane_kinematics


HORIZONTAL_AXES = (0, 2)


def safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def mad_threshold(
    signal: pd.Series | np.ndarray,
    baseline_mask: pd.Series | np.ndarray,
    n_mad: float = 8.0,
) -> float:
    baseline = pd.Series(signal, dtype=float).loc[np.asarray(baseline_mask, dtype=bool)].dropna()
    if baseline.empty:
        raise RuntimeError("Baseline window is empty. Check the input signal.")

    center = float(baseline.median())
    mad = float((baseline - center).abs().median())
    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale == 0:
        robust_scale = float(baseline.std())
    if not np.isfinite(robust_scale) or robust_scale == 0:
        robust_scale = 1e-6
    return center + float(n_mad) * robust_scale


def first_contiguous_crossing(
    time_s: np.ndarray,
    signal: pd.Series | np.ndarray,
    baseline_end_s: float,
    search_end_s: float,
    n_mad: float = 8.0,
    min_run: int = 3,
) -> tuple[int, float]:
    time_s = np.asarray(time_s, dtype=float)
    signal = pd.Series(signal, dtype=float)
    threshold = mad_threshold(signal, time_s <= float(baseline_end_s), n_mad=n_mad)
    mask = (time_s <= float(search_end_s)) & signal.gt(threshold).fillna(False).to_numpy()
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise RuntimeError("No onset was found. Lower threshold_n_mad or increase search_end_s.")

    run_start = idx[0]
    prev = idx[0]
    for current in idx[1:]:
        if current == prev + 1:
            prev = current
            continue
        if prev - run_start + 1 >= int(min_run):
            return int(run_start), float(threshold)
        run_start = current
        prev = current

    if prev - run_start + 1 >= int(min_run):
        return int(run_start), float(threshold)
    return int(idx[0]), float(threshold)


def extract_marker_xyz(opti_raw: pd.DataFrame, marker_id: str) -> pd.DataFrame:
    marker_cols = [
        col
        for col in opti_raw.columns
        if len(col) >= 5 and col[2] == marker_id and col[-1] in ["X", "Y", "Z"]
    ]
    if len(marker_cols) != 3:
        raise ValueError(f"Could not find X/Y/Z columns for marker ID {marker_id}.")

    xyz = opti_raw[marker_cols].apply(pd.to_numeric, errors="coerce").copy()
    xyz.columns = ["X", "Y", "Z"]
    return xyz


def estimate_translation(prev_track: pd.DataFrame, next_track: pd.DataFrame) -> tuple[np.ndarray, int]:
    prev_valid = prev_track.notna().all(axis=1)
    next_valid = next_track.notna().all(axis=1)
    overlap = prev_valid & next_valid

    if overlap.any():
        delta = (
            prev_track.loc[overlap, ["X", "Y", "Z"]].to_numpy()
            - next_track.loc[overlap, ["X", "Y", "Z"]].to_numpy()
        )
        return np.nanmedian(delta, axis=0), int(overlap.sum())

    prev_idx = np.flatnonzero(prev_valid.to_numpy())
    next_idx = np.flatnonzero(next_valid.to_numpy())
    if len(prev_idx) == 0 or len(next_idx) == 0:
        return np.zeros(3, dtype=float), 0

    delta = (
        prev_track.loc[prev_idx[-1], ["X", "Y", "Z"]].to_numpy()
        - next_track.loc[next_idx[0], ["X", "Y", "Z"]].to_numpy()
    )
    return np.asarray(delta, dtype=float), 0


def load_imu(
    path: Path | str,
    baseline_sec: float = 0.5,
    rolling_samples: int = 11,
) -> pd.DataFrame:
    imu = pd.read_csv(path)
    imu["imu_time"] = imu["timestamp"] - imu["timestamp"].iloc[0]
    imu["acc_norm"] = np.sqrt((imu[["AccX", "AccY", "AccZ"]] ** 2).sum(axis=1))
    imu["gyro_norm"] = np.sqrt((imu[["GyroX", "GyroY", "GyroZ"]] ** 2).sum(axis=1))

    baseline = imu.loc[imu["imu_time"] <= float(baseline_sec), "acc_norm"]
    gravity_level = float(baseline.median())
    imu["imu_sync_signal"] = (imu["acc_norm"] - gravity_level).abs().rolling(
        int(rolling_samples), center=True, min_periods=1
    ).mean()
    return imu


def load_optitrack_ankle(
    path: Path | str,
    marker_ids: Iterable[str],
    rolling_samples: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, object]]]:
    marker_ids = list(marker_ids)
    opti_raw = pd.read_csv(path, header=[1, 2, 3, 4, 5])
    frame_col = opti_raw.columns[0]
    time_col = opti_raw.columns[1]

    merged = pd.DataFrame(
        {
            "frame": pd.to_numeric(opti_raw[frame_col], errors="coerce"),
            "opti_time": pd.to_numeric(opti_raw[time_col], errors="coerce"),
        }
    )

    raw_tracks = {marker_id: extract_marker_xyz(opti_raw, marker_id) for marker_id in marker_ids}
    shifted_tracks: dict[str, pd.DataFrame] = {}
    switch_log: list[dict[str, object]] = []
    cumulative_offset = np.zeros(3, dtype=float)
    prev_shifted: pd.DataFrame | None = None

    for i, marker_id in enumerate(marker_ids):
        raw_track = raw_tracks[marker_id]
        if i == 0:
            shifted_track = raw_track.copy()
        else:
            if prev_shifted is None:
                raise RuntimeError("Previous marker track is missing during marker stitching.")
            delta_mm, overlap_count = estimate_translation(prev_shifted, raw_track)
            cumulative_offset = cumulative_offset + delta_mm
            shifted_track = raw_track.add(pd.Series(cumulative_offset, index=["X", "Y", "Z"]), axis=1)

            first_valid = np.flatnonzero(raw_track.notna().all(axis=1).to_numpy())
            first_frame = int(merged.loc[first_valid[0], "frame"]) if len(first_valid) else None
            switch_log.append(
                {
                    "from_marker": marker_ids[i - 1],
                    "to_marker": marker_id,
                    "first_new_frame": first_frame,
                    "overlap_frames": int(overlap_count),
                    "applied_offset_mm": cumulative_offset.copy(),
                }
            )

        shifted_tracks[marker_id] = shifted_track
        prev_shifted = shifted_track

        for axis in ["X", "Y", "Z"]:
            merged[f"{marker_id}_{axis}"] = shifted_track[axis]

    valid_by_marker = pd.DataFrame(
        {marker_id: raw_tracks[marker_id].notna().all(axis=1) for marker_id in marker_ids}
    )

    def pick_source(row: pd.Series) -> str | None:
        for marker_id in marker_ids:
            if bool(row[marker_id]):
                return marker_id
        return None

    merged["marker_source"] = valid_by_marker.apply(pick_source, axis=1)

    for axis in ["X", "Y", "Z"]:
        axis_stack = pd.concat([shifted_tracks[marker_id][axis] for marker_id in marker_ids], axis=1)
        merged[axis] = axis_stack.bfill(axis=1).iloc[:, 0]

    merged[["X", "Y", "Z"]] = merged[["X", "Y", "Z"]].interpolate(limit_direction="both")
    dt = merged["opti_time"].diff().replace(0, np.nan)
    velocity = merged[["X", "Y", "Z"]].diff().div(1000.0).div(dt, axis=0)
    merged["ankle_speed_mps"] = np.sqrt((velocity**2).sum(axis=1)).rolling(
        int(rolling_samples), center=True, min_periods=1
    ).mean()
    merged["opti_sync_signal"] = merged["ankle_speed_mps"].diff().div(dt).abs().rolling(
        int(rolling_samples), center=True, min_periods=1
    ).mean()

    return merged, raw_tracks, shifted_tracks, switch_log


def refine_offset_seconds(
    imu_time: np.ndarray,
    imu_signal: np.ndarray,
    opti_time: np.ndarray,
    opti_signal: np.ndarray,
    coarse_offset_sec: float,
    center_time_sec: float,
    window_sec: tuple[float, float] = (-0.6, 2.0),
    max_shift_sec: float = 1.0,
) -> float:
    imu_time = np.asarray(imu_time, dtype=float)
    imu_signal = np.asarray(imu_signal, dtype=float)
    opti_time = np.asarray(opti_time, dtype=float) + float(coarse_offset_sec)
    opti_signal = np.asarray(opti_signal, dtype=float)

    interpolated = np.interp(imu_time, opti_time, opti_signal, left=np.nan, right=np.nan)
    mask = (
        (imu_time >= float(center_time_sec) + float(window_sec[0]))
        & (imu_time <= float(center_time_sec) + float(window_sec[1]))
        & np.isfinite(interpolated)
    )
    if mask.sum() < 10:
        return 0.0

    x = safe_zscore(imu_signal[mask])
    y = safe_zscore(interpolated[mask])
    cc = correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))
    dt = np.nanmedian(np.diff(imu_time[mask]))
    lag_sec = lags * dt
    valid = np.abs(lag_sec) <= float(max_shift_sec)
    return float(lag_sec[valid][np.argmax(cc[valid])])


def overlap_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def estimate_local_lags(
    imu_time: np.ndarray,
    imu_signal: np.ndarray,
    opti_time: np.ndarray,
    opti_signal: np.ndarray,
    window_sec: float = 12.0,
    step_sec: float = 10.0,
    max_shift_sec: float = 1.5,
) -> pd.DataFrame:
    imu_time = np.asarray(imu_time, dtype=float)
    imu_signal = np.asarray(imu_signal, dtype=float)
    opti_time = np.asarray(opti_time, dtype=float)
    opti_signal = np.asarray(opti_signal, dtype=float)

    common_start = max(float(np.nanmin(imu_time)), float(np.nanmin(opti_time)))
    common_end = min(float(np.nanmax(imu_time)), float(np.nanmax(opti_time)))
    rows = []
    current_start = common_start

    while current_start + float(window_sec) <= common_end:
        current_end = current_start + float(window_sec)
        center = current_start + 0.5 * float(window_sec)
        imu_mask = (imu_time >= current_start) & (imu_time <= current_end)
        t = imu_time[imu_mask]
        x = imu_signal[imu_mask]
        y = np.interp(t, opti_time, opti_signal, left=np.nan, right=np.nan)
        valid = np.isfinite(y)
        t = t[valid]
        x = x[valid]
        y = y[valid]

        if len(t) < 20:
            current_start += float(step_sec)
            continue

        xz = safe_zscore(x)
        yz = safe_zscore(y)
        cc = correlate(xz, yz, mode="full")
        lags = np.arange(-len(yz) + 1, len(xz))
        dt = np.nanmedian(np.diff(t))
        lag_sec = lags * dt
        keep = np.abs(lag_sec) <= float(max_shift_sec)
        if not np.any(keep):
            current_start += float(step_sec)
            continue

        best_lag_sec = float(lag_sec[keep][np.argmax(cc[keep])])
        rows.append(
            {
                "window_start_s": float(current_start),
                "window_end_s": float(current_end),
                "center_time_s": float(center),
                "residual_lag_sec": best_lag_sec,
                "corr": overlap_corr(xz, yz),
                "n_samples": int(len(t)),
            }
        )
        current_start += float(step_sec)

    return pd.DataFrame(rows)


def fit_linear_lag_model(local_lag_df: pd.DataFrame, method: str = "global") -> tuple[float, float]:
    if local_lag_df.empty:
        return 0.0, 0.0

    x = local_lag_df["center_time_s"].to_numpy(dtype=float)
    y = local_lag_df["residual_lag_sec"].to_numpy(dtype=float)

    if method in [None, "none"]:
        return 0.0, 0.0
    if method == "first_last":
        slope, intercept = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
    elif method == "weighted":
        weights = np.clip(local_lag_df["corr"].fillna(0).to_numpy(dtype=float), 0.01, None)
        slope, intercept = np.polyfit(x, y, 1, w=weights)
    elif method == "global":
        slope, intercept = np.polyfit(x, y, 1)
    else:
        raise ValueError(f"Unknown lag fit method: {method}")

    return float(slope), float(intercept)


def apply_linear_lag_model(
    opti_time: np.ndarray,
    slope_sec_per_sec: float,
    intercept_sec: float,
) -> np.ndarray:
    opti_time = np.asarray(opti_time, dtype=float)
    lag_correction = float(slope_sec_per_sec) * opti_time + float(intercept_sec)
    return opti_time + lag_correction


def sync_imu_optitrack_trial(
    imu_path: Path | str,
    opti_path: Path | str,
    marker_ids: Iterable[str],
    imu_baseline_sec: float = 0.5,
    opti_baseline_sec: float = 1.0,
    search_window_sec: float = 8.0,
    imu_rolling_samples: int = 11,
    opti_rolling_samples: int = 7,
    threshold_n_mad: float = 8.0,
    refine_window_sec: tuple[float, float] = (-0.6, 2.0),
    max_refine_shift_sec: float = 1.0,
    lag_window_sec: float = 12.0,
    lag_step_sec: float = 10.0,
    lag_max_shift_sec: float = 1.5,
    linear_fit_method: str = "global",
) -> dict[str, object]:
    imu = load_imu(
        path=imu_path,
        baseline_sec=imu_baseline_sec,
        rolling_samples=imu_rolling_samples,
    )
    opti_ankle, raw_tracks, shifted_tracks, switch_log = load_optitrack_ankle(
        path=opti_path,
        marker_ids=marker_ids,
        rolling_samples=opti_rolling_samples,
    )

    imu_onset_idx, imu_threshold = first_contiguous_crossing(
        imu["imu_time"].to_numpy(),
        imu["imu_sync_signal"],
        baseline_end_s=imu_baseline_sec,
        search_end_s=search_window_sec,
        n_mad=threshold_n_mad,
    )
    opti_onset_idx, opti_threshold = first_contiguous_crossing(
        opti_ankle["opti_time"].to_numpy(),
        opti_ankle["opti_sync_signal"],
        baseline_end_s=opti_baseline_sec,
        search_end_s=search_window_sec,
        n_mad=threshold_n_mad,
    )

    coarse_offset_sec = float(imu.loc[imu_onset_idx, "imu_time"] - opti_ankle.loc[opti_onset_idx, "opti_time"])
    refine_offset_sec = refine_offset_seconds(
        imu["imu_time"].to_numpy(),
        imu["imu_sync_signal"].to_numpy(),
        opti_ankle["opti_time"].to_numpy(),
        opti_ankle["opti_sync_signal"].fillna(0).to_numpy(),
        coarse_offset_sec=coarse_offset_sec,
        center_time_sec=float(imu.loc[imu_onset_idx, "imu_time"]),
        window_sec=refine_window_sec,
        max_shift_sec=max_refine_shift_sec,
    )

    offset_opti_to_imu_sec = coarse_offset_sec + refine_offset_sec
    opti_ankle["time_synced_const"] = opti_ankle["opti_time"] + offset_opti_to_imu_sec

    local_lag_df = estimate_local_lags(
        imu["imu_time"].to_numpy(),
        imu["imu_sync_signal"].to_numpy(),
        opti_ankle["time_synced_const"].to_numpy(),
        opti_ankle["opti_sync_signal"].fillna(0).to_numpy(),
        window_sec=lag_window_sec,
        step_sec=lag_step_sec,
        max_shift_sec=lag_max_shift_sec,
    )
    drift_slope_sec_per_sec, drift_intercept_sec = fit_linear_lag_model(
        local_lag_df,
        method=linear_fit_method,
    )

    opti_ankle["time_synced_linear"] = apply_linear_lag_model(
        opti_ankle["time_synced_const"].to_numpy(),
        drift_slope_sec_per_sec,
        drift_intercept_sec,
    )
    opti_ankle["time_synced_to_imu"] = opti_ankle["time_synced_linear"]

    local_lag_df_linear = estimate_local_lags(
        imu["imu_time"].to_numpy(),
        imu["imu_sync_signal"].to_numpy(),
        opti_ankle["time_synced_linear"].to_numpy(),
        opti_ankle["opti_sync_signal"].fillna(0).to_numpy(),
        window_sec=lag_window_sec,
        step_sec=lag_step_sec,
        max_shift_sec=lag_max_shift_sec,
    )

    imu_time = imu["imu_time"].to_numpy(dtype=float)
    imu_sync_z = safe_zscore(imu["imu_sync_signal"].to_numpy(dtype=float))
    opti_sync_z = safe_zscore(opti_ankle["opti_sync_signal"].fillna(0).to_numpy(dtype=float))

    common_start = max(float(imu["imu_time"].min()), float(opti_ankle["time_synced_linear"].min()))
    common_end = min(float(imu["imu_time"].max()), float(opti_ankle["time_synced_linear"].max()))
    overlap_mask = (imu_time >= common_start) & (imu_time <= common_end)

    opti_before_on_imu = np.interp(
        imu_time,
        opti_ankle["opti_time"].to_numpy(dtype=float),
        opti_sync_z,
        left=np.nan,
        right=np.nan,
    )
    opti_const_on_imu = np.interp(
        imu_time,
        opti_ankle["time_synced_const"].to_numpy(dtype=float),
        opti_sync_z,
        left=np.nan,
        right=np.nan,
    )
    opti_linear_on_imu = np.interp(
        imu_time,
        opti_ankle["time_synced_linear"].to_numpy(dtype=float),
        opti_sync_z,
        left=np.nan,
        right=np.nan,
    )

    summary = {
        "imu_onset_time_s": float(imu.loc[imu_onset_idx, "imu_time"]),
        "opti_onset_time_s": float(opti_ankle.loc[opti_onset_idx, "opti_time"]),
        "imu_onset_threshold": float(imu_threshold),
        "opti_onset_threshold": float(opti_threshold),
        "coarse_offset_opti_to_imu_s": float(coarse_offset_sec),
        "refine_offset_s": float(refine_offset_sec),
        "constant_offset_opti_to_imu_s": float(offset_opti_to_imu_sec),
        "linear_fit_method": linear_fit_method,
        "linear_drift_slope_sec_per_sec": float(drift_slope_sec_per_sec),
        "linear_drift_intercept_sec": float(drift_intercept_sec),
        "linear_time_scale": float(1.0 + drift_slope_sec_per_sec),
        "residual_lag_first_window_before_s": float(local_lag_df["residual_lag_sec"].iloc[0])
        if not local_lag_df.empty
        else np.nan,
        "residual_lag_last_window_before_s": float(local_lag_df["residual_lag_sec"].iloc[-1])
        if not local_lag_df.empty
        else np.nan,
        "residual_lag_first_window_after_s": float(local_lag_df_linear["residual_lag_sec"].iloc[0])
        if not local_lag_df_linear.empty
        else np.nan,
        "residual_lag_last_window_after_s": float(local_lag_df_linear["residual_lag_sec"].iloc[-1])
        if not local_lag_df_linear.empty
        else np.nan,
        "corr_before_sync": float(overlap_corr(imu_sync_z[overlap_mask], opti_before_on_imu[overlap_mask])),
        "corr_after_const_sync": float(overlap_corr(imu_sync_z[overlap_mask], opti_const_on_imu[overlap_mask])),
        "corr_after_linear_sync": float(overlap_corr(imu_sync_z[overlap_mask], opti_linear_on_imu[overlap_mask])),
        "n_imu_samples": int(len(imu)),
        "n_opti_samples": int(len(opti_ankle)),
    }

    return {
        "imu": imu,
        "opti_ankle": opti_ankle,
        "raw_tracks": raw_tracks,
        "shifted_tracks": shifted_tracks,
        "switch_log": switch_log,
        "summary": summary,
        "local_lag_df": local_lag_df,
        "local_lag_df_linear": local_lag_df_linear,
        "imu_onset_idx": int(imu_onset_idx),
        "opti_onset_idx": int(opti_onset_idx),
    }


def export_sync_results(
    imu_df: pd.DataFrame,
    opti_df: pd.DataFrame,
    summary: dict[str, object],
    local_lag_df: pd.DataFrame,
    local_lag_df_linear: pd.DataFrame,
    imu_output_path: Path | str,
    opti_output_path: Path | str,
    summary_output_path: Path | str,
    lag_output_path: Path | str,
) -> dict[str, Path]:
    imu_output_path = Path(imu_output_path)
    opti_output_path = Path(opti_output_path)
    summary_output_path = Path(summary_output_path)
    lag_output_path = Path(lag_output_path)

    const_lag_df = local_lag_df.copy()
    linear_lag_df = local_lag_df_linear.copy()
    if const_lag_df.empty:
        const_lag_df = pd.DataFrame(
            columns=["window_start_s", "window_end_s", "center_time_s", "residual_lag_sec", "corr", "n_samples"]
        )
    if linear_lag_df.empty:
        linear_lag_df = pd.DataFrame(columns=["center_time_s", "residual_lag_sec", "corr", "n_samples"])

    lag_export = const_lag_df.rename(
        columns={
            "residual_lag_sec": "residual_lag_const_sec",
            "corr": "corr_const",
            "n_samples": "n_samples_const",
        }
    )
    lag_export = lag_export.merge(
        linear_lag_df[["center_time_s", "residual_lag_sec", "corr", "n_samples"]].rename(
            columns={
                "residual_lag_sec": "residual_lag_linear_sec",
                "corr": "corr_linear",
                "n_samples": "n_samples_linear",
            }
        ),
        on="center_time_s",
        how="outer",
    ).sort_values("center_time_s")

    imu_output_path.parent.mkdir(parents=True, exist_ok=True)
    opti_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    lag_output_path.parent.mkdir(parents=True, exist_ok=True)

    imu_df.to_csv(imu_output_path, index=False)
    opti_df.to_csv(opti_output_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_output_path, index=False)
    lag_export.to_csv(lag_output_path, index=False)

    return {
        "imu_output_path": imu_output_path,
        "opti_output_path": opti_output_path,
        "summary_output_path": summary_output_path,
        "lag_output_path": lag_output_path,
    }


def _cumulative_distance(traj: np.ndarray, dims: int = 2) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    if traj.shape[0] <= 1:
        return np.zeros(traj.shape[0], dtype=float)

    axes = HORIZONTAL_AXES if int(dims) == 2 else (0, 1, 2)
    step = np.linalg.norm(np.diff(traj[:, axes], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def _path_length(traj: np.ndarray, dims: int = 2) -> float:
    return float(_cumulative_distance(traj, dims=dims)[-1])


def _endpoint_displacement(traj: np.ndarray, dims: int = 2) -> float:
    traj = np.asarray(traj, dtype=float)
    if traj.shape[0] == 0:
        return 0.0
    axes = HORIZONTAL_AXES if int(dims) == 2 else (0, 1, 2)
    return float(np.linalg.norm(traj[-1, axes] - traj[0, axes]))


def make_full_trial_segment(
    imu_df: pd.DataFrame,
    opti_df: pd.DataFrame,
    time_col: str = "time_synced_to_imu",
    pos_smooth_samples: int = 241,
    vel_smooth_samples: int = 121,
    min_speed_mps: float = 0.08,
    segment_id: int = 1,
    direction_label: str = "full_trial",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opti_with_motion, _, _, _ = add_motion_plane_kinematics(
        opti_df,
        time_col=time_col,
        pos_smooth_samples=pos_smooth_samples,
        vel_smooth_samples=vel_smooth_samples,
        speed_threshold_mps=min_speed_mps,
    )
    active = opti_with_motion["motion_plane_speed_smooth_mps"].to_numpy(dtype=float) > float(min_speed_mps)
    active_idx = np.flatnonzero(active)
    if len(active_idx) == 0:
        raise RuntimeError("No active motion interval was found for the requested full-trial segment.")

    start_idx = int(active_idx[0])
    end_idx = int(active_idx[-1])
    start_time_s = float(opti_with_motion.iloc[start_idx][time_col])
    end_time_s = float(opti_with_motion.iloc[end_idx][time_col])

    opti_segment_df = opti_with_motion.loc[
        (opti_with_motion[time_col] >= start_time_s) & (opti_with_motion[time_col] <= end_time_s)
    ].copy()
    imu_segment_df = imu_df.loc[
        (imu_df["imu_time"] >= start_time_s) & (imu_df["imu_time"] <= end_time_s)
    ].copy()

    gt_xyz_m = opti_segment_df[["X", "Y", "Z"]].to_numpy(dtype=float) / 1000.0
    gt_xyz_m = gt_xyz_m - gt_xyz_m[0]

    segment_summary_df = pd.DataFrame(
        [
            {
                "segment_id": int(segment_id),
                "direction_label": direction_label,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_frame": int(opti_with_motion.iloc[start_idx]["frame"]),
                "end_frame": int(opti_with_motion.iloc[end_idx]["frame"]),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_sec": float(end_time_s - start_time_s),
                "min_speed_mps": float(min_speed_mps),
                "n_opti_samples": int(len(opti_segment_df)),
                "n_imu_samples": int(len(imu_segment_df)),
                "median_motion_speed_mps": float(opti_segment_df["motion_plane_speed_smooth_mps"].median()),
                "max_motion_speed_mps": float(opti_segment_df["motion_plane_speed_smooth_mps"].max()),
                "gt_path_xz_m": _path_length(gt_xyz_m, dims=2),
                "gt_path_3d_m": _path_length(gt_xyz_m, dims=3),
                "gt_disp_xz_m": _endpoint_displacement(gt_xyz_m, dims=2),
                "gt_disp_3d_m": _endpoint_displacement(gt_xyz_m, dims=3),
            }
        ]
    )

    return opti_with_motion, segment_summary_df, imu_segment_df, opti_segment_df
