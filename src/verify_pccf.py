import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# 设置随机种子以保证可复现性
np.random.seed(42)

class StandardPredictor:
    """
    标准预测器：模拟传统的固定上下文窗口模型。
    使用简单的加权移动平均（类似固定Attention）进行预测。
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def update(self, value):
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def predict(self):
        if not self.history:
            return 0
        # 简单假设：预测下一个值是最近几个值的线性趋势或平均
        # 这里使用简单的移动平均作为基准
        return np.mean(self.history)

class ExponentialMovingAveragePredictor:
    def __init__(self, alpha=0.2, initial_value=0.0):
        self.alpha = float(alpha)
        self.value = float(initial_value)
        self.has_value = False

    def update(self, value):
        value = float(value)
        if not self.has_value:
            self.value = value
            self.has_value = True
            return
        self.value = self.alpha * value + (1.0 - self.alpha) * self.value

    def predict(self):
        return self.value if self.has_value else 0.0

class KalmanFilter1DPredictor:
    def __init__(self, x0=0.0, p0=1.0, q=1e-2, r=1.0):
        self.x = float(x0)
        self.p = float(p0)
        self.q = float(q)
        self.r = float(r)
        self.has_value = False

    def predict(self):
        return self.x if self.has_value else 0.0

    def update(self, z):
        z = float(z)
        if not self.has_value:
            self.x = z
            self.p = 1.0
            self.has_value = True
            return
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p

class ErrorAdaptiveLRPredictor:
    def __init__(self, initial_belief=0.0, learning_rate_base=0.1, gain_scale=1.0, lr_min=0.01, lr_max=0.9):
        self.belief = float(initial_belief)
        self.lr_base = float(learning_rate_base)
        self.gain_scale = float(gain_scale)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)

    def predict(self):
        return self.belief

    def update(self, sensory_input):
        prediction = self.predict()
        prediction_error = float(sensory_input - prediction)
        adaptive_lr = self.lr_base * (1.0 + self.gain_scale * abs(prediction_error))
        adaptive_lr = float(np.clip(adaptive_lr, self.lr_min, self.lr_max))
        self.belief = self.belief + adaptive_lr * prediction_error
        return prediction_error, adaptive_lr

class PageHinkley:
    def __init__(self, delta=0.01, threshold=20.0):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self.t = 0
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, x: float) -> bool:
        x = float(x)
        self.t += 1
        self.mean += (x - self.mean) / float(self.t)
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.threshold

class DriftResetWrapper:
    def __init__(self, name: str, model_factory, detector: PageHinkley):
        self.name = name
        self.model_factory = model_factory
        self.detector = detector
        self.model = self.model_factory()

    def reset(self):
        self.detector.reset()
        self.model = self.model_factory()

    def predict(self):
        return self.model.predict()

    def update_with_pred(self, value: float, pred: float):
        err = abs(float(value) - float(pred))
        drift = self.detector.update(err)
        if drift:
            self.reset()
        self.model.update(value)

class PCCFPredictor:
    """
    PCCF 预测器：模拟基于预测编码和自由能原理的模型。
    核心机制：
    1. 动态精度权重 (Precision Weighting)：根据预测误差调整对历史（先验）的依赖。
    2. 贝叶斯更新：后验 = 先验 * 似然 (这里简化为加权更新)。
    """
    def __init__(
        self,
        initial_belief=0,
        learning_rate_base=0.1,
        gain_scale=4.0,
        lr_min=0.01,
        lr_max=0.9,
        precision_alpha=0.05,
        precision_init_var=1.0,
        precision_eps=1e-6,
    ):
        self.belief = float(initial_belief) # 内部模型/先验信念
        self.precision = 1.0
        self.lr_base = float(learning_rate_base)
        self.gain_scale = float(gain_scale)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.precision_alpha = float(precision_alpha)
        self.precision_var = float(precision_init_var)
        self.precision_eps = float(precision_eps)
        self.history_errors = []

    def predict(self):
        return self.belief

    def update(self, sensory_input):
        """
        根据感官输入 (sensory_input) 更新内部信念。
        PCCF 核心：Surprise (预测误差) 驱动的学习率调整。
        """
        prediction = self.predict()
        prediction_error = sensory_input - prediction
        
        squared_error = float(prediction_error ** 2)
        self.history_errors.append(squared_error)

        # 动态调整：
        # 如果误差很大 (Surprise high)，则 Kalman Gain (学习率) 变大，快速适应新数据
        # 这里的公式模拟卡尔曼滤波或贝叶斯更新中的精度加权
        # 假设感官噪声固定，模型不确定性随误差增加
        
        normalized_surprise = squared_error / (self.precision_var + self.precision_eps)
        adaptive_lr = self.lr_base * (1.0 + self.gain_scale * np.tanh(normalized_surprise))
        adaptive_lr = float(np.clip(adaptive_lr, self.lr_min, self.lr_max)) # 限制范围
        self.precision_var = (1.0 - self.precision_alpha) * self.precision_var + self.precision_alpha * squared_error

        # 更新信念：Belief_new = Belief_old + K * Error
        self.belief = self.belief + adaptive_lr * float(prediction_error)
        
        return prediction_error, adaptive_lr

def generate_data(n_steps=100, seed=42):
    """
    生成具有概念漂移 (Concept Drift) 的数据序列。
    0-50步：常数 10 (加噪声)
    50-100步：突变为 20 (加噪声)
    这模拟了环境规则的变化，测试模型打破“执着”（旧先验）的能力。
    """
    rng = np.random.default_rng(int(seed))
    data = []
    ground_truth = []
    for t in range(n_steps):
        noise = rng.normal(0, 1)
        if t < 50:
            val = 10
        else:
            val = 20
        
        ground_truth.append(val)
        data.append(val + noise)
    return data, ground_truth

@dataclass(frozen=True)
class ScalarRunResult:
    step: np.ndarray
    input_values: np.ndarray
    ground_truth: np.ndarray
    predictions: Dict[str, np.ndarray]
    abs_errors: Dict[str, np.ndarray]
    extra: Dict[str, np.ndarray]

def _recovery_steps(abs_error: np.ndarray, drift_idx: int, threshold: float, consecutive: int) -> int:
    n = abs_error.shape[0]
    if drift_idx >= n:
        return 0
    for i in range(drift_idx, n - consecutive + 1):
        if np.all(abs_error[i:i+consecutive] < threshold):
            return int(i - drift_idx)
    return int(n - drift_idx)

def run_single_seed(
    seed: int,
    steps: int = 100,
    drift_idx: int = 50,
    window_size: int = 10,
    ema_alpha: float = 0.2,
    kf_q: float = 1e-2,
    kf_r: float = 1.0,
    pccf_initial_belief: float = 10.0,
    pccf_lr_base: float = 0.1,
    pccf_gain_scale: float = 4.0,
    pccf_precision_alpha: float = 0.05,
    errlr_gain_scale: float = 1.0,
    ph_delta: float = 0.01,
    ph_threshold: float = 20.0,
) -> ScalarRunResult:
    data, ground_truth = generate_data(n_steps=steps, seed=seed)
    data = np.asarray(data, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)
    step = np.arange(steps, dtype=int)

    models = {
        "MovingAvg": StandardPredictor(window_size=window_size),
        "EMA": ExponentialMovingAveragePredictor(alpha=ema_alpha),
        "Kalman": KalmanFilter1DPredictor(q=kf_q, r=kf_r),
        "ErrorLR": ErrorAdaptiveLRPredictor(
            initial_belief=pccf_initial_belief,
            learning_rate_base=pccf_lr_base,
            gain_scale=errlr_gain_scale,
        ),
        "PCCF": PCCFPredictor(
            initial_belief=pccf_initial_belief,
            learning_rate_base=pccf_lr_base,
            gain_scale=pccf_gain_scale,
            precision_alpha=pccf_precision_alpha,
        ),
        "PH_Reset_MovingAvg": DriftResetWrapper(
            "PH_Reset_MovingAvg",
            lambda: StandardPredictor(window_size=window_size),
            PageHinkley(delta=ph_delta, threshold=ph_threshold),
        ),
        "PH_Reset_EMA": DriftResetWrapper(
            "PH_Reset_EMA",
            lambda: ExponentialMovingAveragePredictor(alpha=ema_alpha),
            PageHinkley(delta=ph_delta, threshold=ph_threshold),
        ),
    }

    preds: Dict[str, List[float]] = {k: [] for k in models.keys()}
    abs_errs: Dict[str, List[float]] = {k: [] for k in models.keys()}
    pccf_lr: List[float] = []
    errlr_lr: List[float] = []

    for t in range(steps):
        val = float(data[t])

        for name, model in models.items():
            pred = float(model.predict())
            preds[name].append(pred)
            abs_errs[name].append(abs(val - pred))

            if hasattr(model, "update_with_pred"):
                model.update_with_pred(val, pred)
            elif name == "PCCF":
                _, lr = model.update(val)
                pccf_lr.append(float(lr))
            elif name == "ErrorLR":
                _, lr = model.update(val)
                errlr_lr.append(float(lr))
            else:
                model.update(val)

    predictions_np = {k: np.asarray(v, dtype=float) for k, v in preds.items()}
    abs_errors_np = {k: np.asarray(v, dtype=float) for k, v in abs_errs.items()}
    extra = {
        "PCCF_lr": np.asarray(pccf_lr, dtype=float),
        "ErrorLR_lr": np.asarray(errlr_lr, dtype=float),
    }
    return ScalarRunResult(
        step=step,
        input_values=data,
        ground_truth=ground_truth,
        predictions=predictions_np,
        abs_errors=abs_errors_np,
        extra=extra,
    )

def _save_architecture_figure(path_base: str = "pccf_architecture"):
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)

    def box(x, y, w, h, text):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.6, 2.8, 2.6, 1.0, "Sensory Input\n(New Token)")
    box(3.7, 2.8, 2.6, 1.0, "Prediction\n(Generative Model)")
    box(6.8, 2.8, 2.6, 1.0, "Prediction Error\n(Surprise)")
    box(6.8, 0.6, 2.6, 1.0, "Precision\nController")
    box(3.7, 0.6, 2.6, 1.0, "Belief / Context\nState")

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        )

    arrow(3.2, 3.3, 3.7, 3.3)
    arrow(6.3, 3.3, 6.8, 3.3)
    arrow(7.9, 2.8, 7.9, 1.6)
    arrow(6.8, 1.1, 6.3, 1.1)
    arrow(3.7, 1.1, 3.2, 1.1)
    arrow(2.0, 2.8, 2.0, 1.6)
    arrow(2.0, 1.6, 3.7, 1.1)
    arrow(5.0, 1.6, 6.8, 3.0)

    ax.text(5.0, 4.2, "PCCF Closed-Loop Control", ha="center", va="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(f"{path_base}.png", dpi=200)
    fig.savefig(f"{path_base}.svg")
    plt.close(fig)

def _paired_permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    observed = float(np.mean(diff))
    rng = np.random.default_rng(int(seed))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.shape[0]), replace=True)
    perm_means = np.mean(signs * diff[None, :], axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1.0) / (float(n_perm) + 1.0)
    return float(p)

def _write_significance_csv(
    out_path: str,
    per_seed_metric: Dict[str, np.ndarray],
    reference: str,
    n_perm: int,
    seed: int,
):
    rows = []
    ref = per_seed_metric[reference]
    for name, values in per_seed_metric.items():
        if name == reference:
            continue
        p = _paired_permutation_pvalue(ref, values, n_perm=n_perm, seed=seed)
        rows.append(
            {
                "Reference": reference,
                "ComparedTo": name,
                "Mean(reference)": float(np.mean(ref)),
                "Mean(compared)": float(np.mean(values)),
                "MeanDiff(reference-compared)": float(np.mean(ref - values)),
                "PermutationPValue(two-sided)": float(p),
                "N": int(ref.shape[0]),
                "NPerm": int(n_perm),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)

def run_multiseed(
    n_seeds: int = 30,
    seed_offset: int = 0,
    steps: int = 100,
    drift_idx: int = 50,
    recovery_threshold: float = 2.0,
    recovery_consecutive: int = 3,
    window_size: int = 10,
    ema_alpha: float = 0.2,
    kf_q: float = 1e-2,
    kf_r: float = 1.0,
    pccf_initial_belief: float = 10.0,
    pccf_lr_base: float = 0.1,
    pccf_gain_scale: float = 4.0,
    pccf_precision_alpha: float = 0.05,
    errlr_gain_scale: float = 1.0,
    ph_delta: float = 0.01,
    ph_threshold: float = 20.0,
    do_significance: bool = True,
    do_sensitivity: bool = True,
    n_perm: int = 20000,
    out_prefix: str = "pccf_scalar_multiseed",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    seeds = [seed_offset + i for i in range(n_seeds)]
    first = run_single_seed(
        seed=seeds[0],
        steps=steps,
        drift_idx=drift_idx,
        window_size=window_size,
        ema_alpha=ema_alpha,
        kf_q=kf_q,
        kf_r=kf_r,
        pccf_initial_belief=pccf_initial_belief,
        pccf_lr_base=pccf_lr_base,
        pccf_gain_scale=pccf_gain_scale,
        pccf_precision_alpha=pccf_precision_alpha,
        errlr_gain_scale=errlr_gain_scale,
        ph_delta=ph_delta,
        ph_threshold=ph_threshold,
    )
    model_names = list(first.predictions.keys())

    abs_errors = {name: np.zeros((n_seeds, steps), dtype=float) for name in model_names}
    predictions = {name: np.zeros((n_seeds, steps), dtype=float) for name in model_names}
    inputs = np.zeros((n_seeds, steps), dtype=float)
    ground_truth = first.ground_truth.copy()

    for i, s in enumerate(seeds):
        res = run_single_seed(
            seed=s,
            steps=steps,
            drift_idx=drift_idx,
            window_size=window_size,
            ema_alpha=ema_alpha,
            kf_q=kf_q,
            kf_r=kf_r,
            pccf_initial_belief=pccf_initial_belief,
            pccf_lr_base=pccf_lr_base,
            pccf_gain_scale=pccf_gain_scale,
            pccf_precision_alpha=pccf_precision_alpha,
            errlr_gain_scale=errlr_gain_scale,
            ph_delta=ph_delta,
            ph_threshold=ph_threshold,
        )
        inputs[i, :] = res.input_values
        for name in model_names:
            abs_errors[name][i, :] = res.abs_errors[name]
            predictions[name][i, :] = res.predictions[name]

    rows = []
    for name in model_names:
        mse_all = np.mean(abs_errors[name] ** 2, axis=1)
        mse_post = np.mean(abs_errors[name][:, drift_idx:] ** 2, axis=1)
        rec = np.asarray(
            [
                _recovery_steps(abs_errors[name][i], drift_idx, recovery_threshold, recovery_consecutive)
                for i in range(n_seeds)
            ],
            dtype=float,
        )
        rows.append(
            {
                "Model": name,
                "MSE_mean": float(np.mean(mse_all)),
                "MSE_std": float(np.std(mse_all, ddof=1)) if n_seeds > 1 else 0.0,
                "PostDriftMSE_mean": float(np.mean(mse_post)),
                "PostDriftMSE_std": float(np.std(mse_post, ddof=1)) if n_seeds > 1 else 0.0,
                "RecoverySteps_mean": float(np.mean(rec)),
                "RecoverySteps_std": float(np.std(rec, ddof=1)) if n_seeds > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(rows).sort_values(by="MSE_mean", ascending=True).reset_index(drop=True)

    step = np.arange(steps, dtype=int)
    mean_abs_err = {name: np.mean(abs_errors[name], axis=0) for name in model_names}
    std_abs_err = {name: np.std(abs_errors[name], axis=0, ddof=1) if n_seeds > 1 else np.zeros(steps) for name in model_names}

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.axvline(x=drift_idx, color="gray", linestyle="--", linewidth=1.2)
    ax.set_title("Scalar Concept Drift: Mean Absolute Error (mean ± std over seeds)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Absolute Error")
    colors = {
        "MovingAvg": "#1f77b4",
        "EMA": "#2ca02c",
        "Kalman": "#9467bd",
        "ErrorLR": "#ff7f0e",
        "PCCF": "#d62728",
        "PH_Reset_MovingAvg": "#17becf",
        "PH_Reset_EMA": "#8c564b",
    }
    for name in model_names:
        c = colors.get(name, None)
        ax.plot(step, mean_abs_err[name], label=name, linewidth=2.0 if name == "PCCF" else 1.6, color=c)
        ax.fill_between(
            step,
            mean_abs_err[name] - std_abs_err[name],
            mean_abs_err[name] + std_abs_err[name],
            alpha=0.15,
            color=c,
            linewidth=0,
        )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_mae_band.png", dpi=200)
    fig.savefig(f"{out_prefix}_mae_band.svg")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    xs = np.arange(len(summary))
    ax.bar(
        xs,
        summary["MSE_mean"].to_numpy(),
        yerr=summary["MSE_std"].to_numpy(),
        capsize=4,
        color="#4c78a8",
        alpha=0.9,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(summary["Model"].tolist())
    ax.set_title("Scalar Concept Drift: MSE (mean ± std over seeds)")
    ax.set_ylabel("MSE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_mse_bar.png", dpi=200)
    fig.savefig(f"{out_prefix}_mse_bar.svg")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(step, np.mean(inputs, axis=0), "k.", alpha=0.25, label="Noisy Input (mean)")
    ax.plot(step, ground_truth, "k--", alpha=0.6, label="Ground Truth")
    ax.axvline(x=drift_idx, color="gray", linestyle="--", linewidth=1.2, label="Drift")
    for name in model_names:
        c = colors.get(name, None)
        ax.plot(step, np.mean(predictions[name], axis=0), label=f"{name} (mean pred)", color=c, linewidth=2.0 if name == "PCCF" else 1.4)
    ax.set_title("Scalar Concept Drift: Mean Predictions over Seeds")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_pred_mean.png", dpi=200)
    fig.savefig(f"{out_prefix}_pred_mean.svg")
    plt.close(fig)

    summary.to_csv(f"{out_prefix}_summary.csv", index=False)

    if do_significance:
        mse_per_seed = {name: np.mean(abs_errors[name] ** 2, axis=1) for name in model_names}
        rec_per_seed = {
            name: np.asarray(
                [
                    _recovery_steps(abs_errors[name][i], drift_idx, recovery_threshold, recovery_consecutive)
                    for i in range(n_seeds)
                ],
                dtype=float,
            )
            for name in model_names
        }
        _write_significance_csv(
            f"{out_prefix}_pvalues_mse.csv",
            per_seed_metric=mse_per_seed,
            reference="PCCF",
            n_perm=n_perm,
            seed=0,
        )
        _write_significance_csv(
            f"{out_prefix}_pvalues_recovery.csv",
            per_seed_metric=rec_per_seed,
            reference="PCCF",
            n_perm=n_perm,
            seed=1,
        )

    if do_sensitivity:
        thresholds = [1.5, 2.0, 2.5]
        consecutives = [1, 3, 5]
        rows = []
        mse_per_seed = {name: np.mean(abs_errors[name] ** 2, axis=1) for name in model_names}
        for thr in thresholds:
            for cons in consecutives:
                rec_per_seed = {
                    name: np.asarray(
                        [
                            _recovery_steps(abs_errors[name][i], drift_idx, thr, cons)
                            for i in range(n_seeds)
                        ],
                        dtype=float,
                    )
                    for name in model_names
                }
                for name in model_names:
                    rows.append(
                        {
                            "Threshold": float(thr),
                            "Consecutive": int(cons),
                            "Model": name,
                            "RecoverySteps_mean": float(np.mean(rec_per_seed[name])),
                            "RecoverySteps_std": float(np.std(rec_per_seed[name], ddof=1)) if n_seeds > 1 else 0.0,
                            "MSE_mean": float(np.mean(mse_per_seed[name])),
                            "NSeeds": int(n_seeds),
                        }
                    )
        pd.DataFrame(rows).to_csv(f"{out_prefix}_recovery_sensitivity.csv", index=False)

    _save_architecture_figure("pccf_architecture")

    payload = {
        "inputs": inputs,
        "ground_truth": ground_truth,
        **{f"abs_errors_{k}": v for k, v in abs_errors.items()},
    }
    return summary, payload

def run_experiment_single(seed: int = 42):
    steps = 100
    data, ground_truth = generate_data(steps, seed=seed)
    std_model = StandardPredictor(window_size=10)
    pccf_model = PCCFPredictor(initial_belief=10, learning_rate_base=0.1)

    results = {
        "step": [],
        "input": [],
        "std_pred": [],
        "pccf_pred": [],
        "std_error": [],
        "pccf_error": [],
        "pccf_lr": [],
    }

    print(f"{'Step':<5} | {'Input':<8} | {'Std Pred':<10} | {'PCCF Pred':<10} | {'Std Err':<10} | {'PCCF Err':<10}")
    print("-" * 70)

    for t in range(steps):
        val = float(data[t])
        pred_std = float(std_model.predict())
        std_model.update(val)
        err_std = abs(val - pred_std)

        pred_pccf = float(pccf_model.predict())
        pccf_err_raw, pccf_lr = pccf_model.update(val)
        err_pccf = abs(float(pccf_err_raw))

        results["step"].append(t)
        results["input"].append(val)
        results["std_pred"].append(pred_std)
        results["pccf_pred"].append(pred_pccf)
        results["std_error"].append(err_std)
        results["pccf_error"].append(err_pccf)
        results["pccf_lr"].append(float(pccf_lr))

        if t % 10 == 0 or (t >= 48 and t <= 55):
            print(f"{t:<5} | {val:<8.2f} | {pred_std:<10.2f} | {pred_pccf:<10.2f} | {err_std:<10.2f} | {err_pccf:<10.2f}")

    df = pd.DataFrame(results)
    mse_std = float(np.mean(df["std_error"] ** 2))
    mse_pccf = float(np.mean(df["pccf_error"] ** 2))

    drift_idx = 50
    recovery_std = _recovery_steps(df["std_error"].to_numpy(), drift_idx, threshold=2.0, consecutive=1)
    recovery_pccf = _recovery_steps(df["pccf_error"].to_numpy(), drift_idx, threshold=2.0, consecutive=1)

    print("\n" + "=" * 30)
    print("Experiment Results Summary")
    print("=" * 30)
    print(f"MSE (Standard): {mse_std:.4f}")
    print(f"MSE (PCCF):     {mse_pccf:.4f}")
    print(f"Improvement:    {((mse_std - mse_pccf) / mse_std) * 100:.2f}%")
    print("-" * 30)
    print("Recovery Steps after Drift (Step 50):")
    print(f"Standard Model: {recovery_std} steps")
    print(f"PCCF Model:     {recovery_pccf} steps")
    print("=" * 30)

    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["input"], "k.", alpha=0.3, label="Sensory Input (Noisy)")
    plt.plot(df["step"], ground_truth, "k--", alpha=0.5, label="True Reality")
    plt.plot(df["step"], df["std_pred"], "b-", label="Standard Model (Fixed Window)")
    plt.plot(df["step"], df["pccf_pred"], "r-", linewidth=2, label="PCCF Model (Dynamic Precision)")
    plt.axvline(x=50, color="g", linestyle=":", label="Concept Drift (Environment Change)")
    plt.title("PCCF Verification: Adaptation to Concept Drift (Single Seed)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pccf_verification_plot.png", dpi=200)
    plt.savefig("pccf_verification_plot.svg")
    plt.close()
    print("Plot saved to pccf_verification_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multiseed"], default="multiseed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out-prefix", type=str, default="pccf_scalar_multiseed")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--kf-q", type=float, default=1e-2)
    parser.add_argument("--kf-r", type=float, default=1.0)
    parser.add_argument("--pccf-lr-base", type=float, default=0.1)
    parser.add_argument("--pccf-gain-scale", type=float, default=4.0)
    parser.add_argument("--pccf-precision-alpha", type=float, default=0.05)
    parser.add_argument("--errlr-gain-scale", type=float, default=1.0)
    parser.add_argument("--ph-delta", type=float, default=0.01)
    parser.add_argument("--ph-threshold", type=float, default=20.0)
    parser.add_argument("--no-significance", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--n-perm", type=int, default=20000)
    args = parser.parse_args()

    if args.mode == "single":
        run_experiment_single(seed=args.seed)
    else:
        summary, _ = run_multiseed(
            n_seeds=args.n_seeds,
            seed_offset=args.seed_offset,
            window_size=args.window_size,
            ema_alpha=args.ema_alpha,
            kf_q=args.kf_q,
            kf_r=args.kf_r,
            pccf_lr_base=args.pccf_lr_base,
            pccf_gain_scale=args.pccf_gain_scale,
            pccf_precision_alpha=args.pccf_precision_alpha,
            errlr_gain_scale=args.errlr_gain_scale,
            ph_delta=args.ph_delta,
            ph_threshold=args.ph_threshold,
            do_significance=(not args.no_significance),
            do_sensitivity=(not args.no_sensitivity),
            n_perm=args.n_perm,
            out_prefix=args.out_prefix,
        )
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 140)
        print(summary.to_string(index=False))
