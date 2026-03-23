from __future__ import annotations

from math import ceil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ins_tools.INS import INS
from ins_tools.geometry_helpers import quat2mat
from ins_tools.util import align_plots, compute_error


IMU_COLUMNS = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
QUAT_COLUMNS = ["q0", "q1", "q2", "q3"]
OPTI_COLUMNS = ["X", "Y", "Z"]


def make_segment_name(segment_id: int, direction_label: str) -> str:
    return f"segment_{int(segment_id):02d}_{direction_label}"


def discover_segments(summary_path: Path | str, segments_dir: Path | str) -> pd.DataFrame:
    summary_path = Path(summary_path)
    segments_dir = Path(segments_dir)
    summary_df = pd.read_csv(summary_path).sort_values("segment_id").reset_index(drop=True)
    summary_df["segment_name"] = summary_df.apply(
        lambda row: make_segment_name(row["segment_id"], row["direction_label"]),
        axis=1,
    )
    summary_df["imu_path"] = summary_df["segment_name"].map(lambda name: str(segments_dir / f"{name}_imu.csv"))
    summary_df["opti_path"] = summary_df["segment_name"].map(lambda name: str(segments_dir / f"{name}_opti.csv"))
    return summary_df


def _positive_dt(dt: np.ndarray, fallback: float = 0.01) -> np.ndarray:
    dt = np.asarray(dt, dtype=float)
    valid = dt[np.isfinite(dt) & (dt > 0)]
    fill = float(np.median(valid)) if valid.size else fallback
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, fill)
    return dt


def _interp_series(time_src: np.ndarray, values: np.ndarray, time_dst: np.ndarray) -> np.ndarray:
    time_src = np.asarray(time_src, dtype=float)
    values = np.asarray(values, dtype=float)
    time_dst = np.asarray(time_dst, dtype=float)

    valid = np.isfinite(time_src) & np.isfinite(values)
    if not np.any(valid):
        return np.full(time_dst.shape, np.nan, dtype=float)

    time_src = time_src[valid]
    values = values[valid]
    order = np.argsort(time_src)
    time_src = time_src[order]
    values = values[order]
    time_src, unique_idx = np.unique(time_src, return_index=True)
    values = values[unique_idx]
    return np.interp(time_dst, time_src, values)


def _safe_align_plots(traj_est: np.ndarray, traj_gt: np.ndarray, dist: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    traj_est = np.asarray(traj_est, dtype=float)
    traj_gt = np.asarray(traj_gt, dtype=float)
    traj_est = traj_est - traj_est[0]
    traj_gt = traj_gt - traj_gt[0]

    try:
        est_aligned, gt_aligned = align_plots(traj_est.copy(), traj_gt.copy(), dist=dist)
        if not np.isfinite(est_aligned).all() or not np.isfinite(gt_aligned).all():
            raise ValueError("alignment produced non-finite values")
        return est_aligned, gt_aligned
    except Exception:
        return traj_est, traj_gt


def cumulative_distance(traj: np.ndarray, dims: int = 2) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    if traj.shape[0] <= 1:
        return np.zeros(traj.shape[0], dtype=float)
    step = np.linalg.norm(np.diff(traj[:, :dims], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def path_length(traj: np.ndarray, dims: int = 2) -> float:
    return float(cumulative_distance(traj, dims=dims)[-1])


def endpoint_error(traj_est: np.ndarray, traj_gt: np.ndarray, dims: int = 2) -> float:
    return float(np.linalg.norm(traj_est[-1, :dims] - traj_gt[-1, :dims]))


def _trajectory_label(attitude_source: str) -> str:
    if attitude_source == "measured_quat":
        return "Pyshoe + quat"
    return "Pyshoe gyro"


def _integrate_with_measured_quat(
    imu_array: np.ndarray,
    quat_array: np.ndarray,
    dt: np.ndarray,
    zv: np.ndarray,
    gravity_mps2: float = 9.8029,
    quat_mode: str = "direct",
) -> np.ndarray:
    quat_array = np.asarray(quat_array, dtype=float)
    if quat_array.ndim != 2 or quat_array.shape[1] != 4:
        raise ValueError("quat_array must have shape (N, 4)")
    if quat_mode not in {"direct", "transpose"}:
        raise ValueError("quat_mode must be 'direct' or 'transpose'")

    x = np.zeros((imu_array.shape[0], 3), dtype=float)
    v = np.zeros((imu_array.shape[0], 3), dtype=float)
    gravity = np.array([0.0, 0.0, gravity_mps2], dtype=float)

    for k in range(1, imu_array.shape[0]):
        q = quat_array[k]
        q_norm = np.linalg.norm(q)
        if not np.isfinite(q_norm) or q_norm <= 0:
            q = quat_array[k - 1]
            q_norm = np.linalg.norm(q)
        q = q / q_norm

        rot = quat2mat(q)
        if quat_mode == "transpose":
            rot = rot.T

        acc_n = rot.dot(imu_array[k, :3]) + gravity
        v[k] = v[k - 1] + dt[k - 1] * acc_n
        if zv[k]:
            v[k] = 0.0
        x[k] = x[k - 1] + dt[k - 1] * v[k] + 0.5 * (dt[k - 1] ** 2) * acc_n

    return x


def load_segment_pair(
    imu_path: Path | str,
    opti_path: Path | str,
    time_column: str = "time_synced_to_imu",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    imu_df = pd.read_csv(imu_path)
    opti_df = pd.read_csv(opti_path)

    imu_time = imu_df["imu_time"].to_numpy(dtype=float)
    opti_time = opti_df[time_column].to_numpy(dtype=float)
    opti_xyz_m = opti_df[OPTI_COLUMNS].to_numpy(dtype=float) / 1000.0

    gt_xyz_m = np.column_stack(
        [_interp_series(opti_time, opti_xyz_m[:, axis], imu_time) for axis in range(3)]
    )
    gt_xyz_m = gt_xyz_m - gt_xyz_m[0]

    return imu_df, opti_df, imu_time, gt_xyz_m


def analyze_segment(
    imu_path: Path | str,
    opti_path: Path | str,
    segment_id: int,
    direction_label: str,
    detector: str,
    threshold: float,
    window_size: int = 5,
    sigma_a: float = 0.01,
    sigma_w_dps: float = 0.1,
    align_dist_m: float = 0.8,
    time_column: str = "time_synced_to_imu",
    attitude_source: str = "gyro_integrated",
    quat_mode: str = "direct",
) -> tuple[dict, pd.DataFrame]:
    imu_df, opti_df, imu_time, gt_xyz_m = load_segment_pair(
        imu_path=imu_path,
        opti_path=opti_path,
        time_column=time_column,
    )
    imu_time_rel = imu_time - imu_time[0]
    imu_array = imu_df[IMU_COLUMNS].to_numpy(dtype=float)

    dt = _positive_dt(np.diff(imu_time), fallback=0.01)
    ins = INS(
        imu_array,
        sigma_a=float(sigma_a),
        sigma_w=np.deg2rad(float(sigma_w_dps)),
        T=float(np.median(dt)),
        dt=dt,
    )

    lrt = ins.Localizer.compute_zv_lrt(
        W=int(window_size),
        G=float(threshold),
        detector=detector,
        return_zv=False,
    )
    zv = lrt < float(threshold)

    if attitude_source == "gyro_integrated":
        x = ins.baseline(zv=zv)
        est_xyz_m = x[:, :3]
    elif attitude_source == "measured_quat":
        if not set(QUAT_COLUMNS).issubset(imu_df.columns):
            raise ValueError(f"Quaternion columns {QUAT_COLUMNS} are required for attitude_source='measured_quat'")
        quat_array = imu_df[QUAT_COLUMNS].to_numpy(dtype=float)
        est_xyz_m = _integrate_with_measured_quat(
            imu_array=imu_array,
            quat_array=quat_array,
            dt=dt,
            zv=zv,
            gravity_mps2=ins.g,
            quat_mode=quat_mode,
        )
    else:
        raise ValueError("attitude_source must be 'gyro_integrated' or 'measured_quat'")

    est_aligned_m, gt_aligned_m = _safe_align_plots(est_xyz_m, gt_xyz_m, dist=align_dist_m)

    opti_time = opti_df[time_column].to_numpy(dtype=float)
    opti_speed = (
        _interp_series(opti_time, opti_df["ankle_speed_mps"].to_numpy(dtype=float), imu_time)
        if "ankle_speed_mps" in opti_df.columns
        else np.concatenate(
            [
                [0.0],
                np.linalg.norm(np.diff(gt_xyz_m[:, :3], axis=0), axis=1) / np.maximum(dt, 1e-6),
            ]
        )
    )

    est_cumdist_xy = cumulative_distance(est_aligned_m, dims=2)
    gt_cumdist_xy = cumulative_distance(gt_aligned_m, dims=2)
    est_cumdist_3d = cumulative_distance(est_aligned_m, dims=3)
    gt_cumdist_3d = cumulative_distance(gt_aligned_m, dims=3)

    metrics = {
        "segment_id": int(segment_id),
        "direction_label": direction_label,
        "segment_name": make_segment_name(segment_id, direction_label),
        "detector": detector,
        "threshold": float(threshold),
        "attitude_source": attitude_source,
        "quat_mode": quat_mode if attitude_source == "measured_quat" else "",
        "num_samples": int(len(imu_df)),
        "duration_s": float(imu_time_rel[-1]) if len(imu_time_rel) else 0.0,
        "zv_count": int(np.sum(zv)),
        "zv_ratio": float(np.mean(zv)),
        "armse_2d_m": float(compute_error(est_aligned_m, gt_aligned_m, dim="2d")),
        "armse_3d_m": float(compute_error(est_aligned_m, gt_aligned_m, dim="3d")),
        "endpoint_xy_error_m": endpoint_error(est_aligned_m, gt_aligned_m, dims=2),
        "endpoint_3d_error_m": endpoint_error(est_aligned_m, gt_aligned_m, dims=3),
        "est_path_xy_m": path_length(est_aligned_m, dims=2),
        "gt_path_xy_m": path_length(gt_aligned_m, dims=2),
        "path_xy_error_m": path_length(est_aligned_m, dims=2) - path_length(gt_aligned_m, dims=2),
        "est_disp_xy_m": float(np.linalg.norm(est_aligned_m[-1, :2] - est_aligned_m[0, :2])),
        "gt_disp_xy_m": float(np.linalg.norm(gt_aligned_m[-1, :2] - gt_aligned_m[0, :2])),
        "disp_xy_error_m": float(
            np.linalg.norm(est_aligned_m[-1, :2] - est_aligned_m[0, :2])
            - np.linalg.norm(gt_aligned_m[-1, :2] - gt_aligned_m[0, :2])
        ),
        "est_path_3d_m": path_length(est_aligned_m, dims=3),
        "gt_path_3d_m": path_length(gt_aligned_m, dims=3),
        "path_3d_error_m": path_length(est_aligned_m, dims=3) - path_length(gt_aligned_m, dims=3),
    }

    detail_df = pd.DataFrame(
        {
            "segment_time_s": imu_time_rel,
            "imu_time": imu_time,
            "lrt": lrt.astype(float),
            "zv_detected": zv.astype(int),
            "opti_speed_mps": opti_speed.astype(float),
            "est_x_m": est_xyz_m[:, 0],
            "est_y_m": est_xyz_m[:, 1],
            "est_z_m": est_xyz_m[:, 2],
            "est_x_aligned_m": est_aligned_m[:, 0],
            "est_y_aligned_m": est_aligned_m[:, 1],
            "est_z_aligned_m": est_aligned_m[:, 2],
            "gt_x_m": gt_aligned_m[:, 0],
            "gt_y_m": gt_aligned_m[:, 1],
            "gt_z_m": gt_aligned_m[:, 2],
            "est_cumdist_xy_m": est_cumdist_xy,
            "gt_cumdist_xy_m": gt_cumdist_xy,
            "est_cumdist_3d_m": est_cumdist_3d,
            "gt_cumdist_3d_m": gt_cumdist_3d,
        }
    )

    return metrics, detail_df


def sweep_all_segments(
    segment_table: pd.DataFrame,
    detector_grids: dict[str, np.ndarray | list[float]],
    window_size: int = 5,
    sigma_a: float = 0.01,
    sigma_w_dps: float = 0.1,
    align_dist_m: float = 0.8,
    time_column: str = "time_synced_to_imu",
    attitude_source: str = "gyro_integrated",
    quat_mode: str = "direct",
) -> pd.DataFrame:
    rows = []
    for segment in segment_table.itertuples(index=False):
        for detector, thresholds in detector_grids.items():
            for threshold in thresholds:
                metrics, _ = analyze_segment(
                    imu_path=segment.imu_path,
                    opti_path=segment.opti_path,
                    segment_id=segment.segment_id,
                    direction_label=segment.direction_label,
                    detector=detector,
                    threshold=float(threshold),
                    window_size=window_size,
                    sigma_a=sigma_a,
                    sigma_w_dps=sigma_w_dps,
                    align_dist_m=align_dist_m,
                    time_column=time_column,
                    attitude_source=attitude_source,
                    quat_mode=quat_mode,
                )
                rows.append(metrics)
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    return metrics_df.sort_values(["segment_id", "detector", "threshold"]).reset_index(drop=True)


def select_best_per_segment(metrics_df: pd.DataFrame, metric: str = "armse_2d_m") -> pd.DataFrame:
    work = metrics_df.copy()
    work["abs_path_xy_error_m"] = work["path_xy_error_m"].abs()
    work["abs_disp_xy_error_m"] = work["disp_xy_error_m"].abs()
    ranked = work.sort_values(
        ["segment_id", metric, "abs_path_xy_error_m", "abs_disp_xy_error_m", "zv_ratio"]
    )
    return ranked.groupby("segment_id", as_index=False).first().sort_values("segment_id").reset_index(drop=True)


def summarize_best_results(best_df: pd.DataFrame) -> pd.DataFrame:
    detector_counts = best_df["detector"].value_counts().to_dict()
    summary = {
        "num_segments": int(len(best_df)),
        "mean_armse_2d_m": float(best_df["armse_2d_m"].mean()),
        "median_armse_2d_m": float(best_df["armse_2d_m"].median()),
        "mean_endpoint_xy_error_m": float(best_df["endpoint_xy_error_m"].mean()),
        "mean_abs_path_xy_error_m": float(best_df["path_xy_error_m"].abs().mean()),
        "mean_abs_disp_xy_error_m": float(best_df["disp_xy_error_m"].abs().mean()),
    }
    for detector, count in detector_counts.items():
        summary[f"count_{detector}"] = int(count)
    return pd.DataFrame([summary])


def export_best_run_details(
    best_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    output_dir: Path | str,
    window_size: int = 5,
    sigma_a: float = 0.01,
    sigma_w_dps: float = 0.1,
    align_dist_m: float = 0.8,
    time_column: str = "time_synced_to_imu",
    attitude_source: str = "gyro_integrated",
    quat_mode: str = "direct",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_lookup = segment_table.set_index("segment_id")
    written_paths = []

    for row in best_df.itertuples(index=False):
        segment = segment_lookup.loc[row.segment_id]
        _, detail_df = analyze_segment(
            imu_path=segment["imu_path"],
            opti_path=segment["opti_path"],
            segment_id=row.segment_id,
            direction_label=row.direction_label,
            detector=row.detector,
            threshold=row.threshold,
            window_size=window_size,
            sigma_a=sigma_a,
            sigma_w_dps=sigma_w_dps,
            align_dist_m=align_dist_m,
            time_column=time_column,
            attitude_source=attitude_source,
            quat_mode=quat_mode,
        )
        suffix = "quat" if attitude_source == "measured_quat" else "gyro"
        output_path = output_dir / f"{row.segment_name}_{row.detector}_{suffix}_detail.csv"
        detail_df.to_csv(output_path, index=False)
        written_paths.append(output_path)

    return written_paths


def plot_metric_summary(best_df: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    best_df = best_df.sort_values("segment_id")
    labels = best_df["segment_name"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    axes[0].bar(labels, best_df["armse_2d_m"], color="#3267c8")
    axes[0].set_title("Best 2D ARMSE by segment")
    axes[0].set_ylabel("ARMSE [m]")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(labels, best_df["gt_path_xy_m"], marker="o", label="OptiTrack path")
    axes[1].plot(labels, best_df["est_path_xy_m"], marker="o", label="Pyshoe path")
    axes[1].set_title("Horizontal distance by segment")
    axes[1].set_ylabel("Distance [m]")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()

    axes[2].plot(labels, best_df["gt_disp_xy_m"], marker="o", label="OptiTrack endpoint displacement")
    axes[2].plot(labels, best_df["est_disp_xy_m"], marker="o", label="Pyshoe endpoint displacement")
    axes[2].set_title("Endpoint displacement by segment")
    axes[2].set_ylabel("Displacement [m]")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend()

    return fig, axes


def plot_all_segment_overlays(
    best_df: pd.DataFrame,
    segment_table: pd.DataFrame,
    window_size: int = 5,
    sigma_a: float = 0.01,
    sigma_w_dps: float = 0.1,
    align_dist_m: float = 0.8,
    time_column: str = "time_synced_to_imu",
    cols: int = 3,
    attitude_source: str = "gyro_integrated",
    quat_mode: str = "direct",
) -> tuple[plt.Figure, np.ndarray]:
    best_df = best_df.sort_values("segment_id")
    rows = ceil(len(best_df) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    segment_lookup = segment_table.set_index("segment_id")

    for ax, row in zip(axes.flat, best_df.itertuples(index=False)):
        segment = segment_lookup.loc[row.segment_id]
        _, detail_df = analyze_segment(
            imu_path=segment["imu_path"],
            opti_path=segment["opti_path"],
            segment_id=row.segment_id,
            direction_label=row.direction_label,
            detector=row.detector,
            threshold=row.threshold,
            window_size=window_size,
            sigma_a=sigma_a,
            sigma_w_dps=sigma_w_dps,
            align_dist_m=align_dist_m,
            time_column=time_column,
            attitude_source=attitude_source,
            quat_mode=quat_mode,
        )
        ax.plot(detail_df["gt_x_m"], detail_df["gt_y_m"], label="OptiTrack", linewidth=2.0)
        ax.plot(
            detail_df["est_x_aligned_m"],
            detail_df["est_y_aligned_m"],
            label=_trajectory_label(attitude_source),
            linewidth=1.5,
        )
        source_tag = "quat" if attitude_source == "measured_quat" else "gyro"
        ax.set_title(
            f"S{row.segment_id:02d} {row.direction_label}\n"
            f"{row.detector}, {source_tag}, G={row.threshold:.3g}, ARMSE={row.armse_2d_m:.2f} m"
        )
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[len(best_df) :]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    return fig, axes


def plot_segment_detail(detail_df: pd.DataFrame, metrics: dict | pd.Series) -> tuple[plt.Figure, np.ndarray]:
    if isinstance(metrics, pd.Series):
        metrics = metrics.to_dict()

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)

    axes[0].plot(detail_df["gt_x_m"], detail_df["gt_y_m"], label="OptiTrack", linewidth=2.0)
    axes[0].plot(
        detail_df["est_x_aligned_m"],
        detail_df["est_y_aligned_m"],
        label=_trajectory_label(metrics.get("attitude_source", "gyro_integrated")),
        linewidth=1.5,
    )
    axes[0].axis("equal")
    source_tag = "quat" if metrics.get("attitude_source") == "measured_quat" else "gyro"
    axes[0].set_title(
        f"{metrics['segment_name']} | {metrics['detector']} {source_tag} G={metrics['threshold']:.3g} | "
        f"ARMSE2D={metrics['armse_2d_m']:.3f} m"
    )
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].legend()

    axes[1].plot(detail_df["segment_time_s"], detail_df["gt_cumdist_xy_m"], label="OptiTrack")
    axes[1].plot(
        detail_df["segment_time_s"],
        detail_df["est_cumdist_xy_m"],
        label=_trajectory_label(metrics.get("attitude_source", "gyro_integrated")),
    )
    axes[1].set_title("Cumulative horizontal distance")
    axes[1].set_xlabel("Segment time [s]")
    axes[1].set_ylabel("Distance [m]")
    axes[1].legend()

    axes[2].plot(detail_df["segment_time_s"], detail_df["lrt"], label="Detector statistic", color="#3267c8")
    axes[2].axhline(metrics["threshold"], linestyle="--", color="#d84b20", label="Threshold")
    axes[2].fill_between(
        detail_df["segment_time_s"],
        0.0,
        1.0,
        where=detail_df["zv_detected"].astype(bool).to_numpy(),
        transform=axes[2].get_xaxis_transform(),
        color="#2b8a3e",
        alpha=0.18,
        label="ZV detected",
    )
    ax2 = axes[2].twinx()
    ax2.plot(detail_df["segment_time_s"], detail_df["opti_speed_mps"], color="#6f42c1", alpha=0.7, label="Opti speed")
    axes[2].set_title("Zero-velocity detection vs OptiTrack ankle speed")
    axes[2].set_xlabel("Segment time [s]")
    axes[2].set_ylabel("Detector statistic")
    ax2.set_ylabel("Opti speed [m/s]")

    handles_1, labels_1 = axes[2].get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    axes[2].legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    return fig, axes
