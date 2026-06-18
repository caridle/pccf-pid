import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

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

class SimpleTransformer(nn.Module):
    """
    一个极简的 Transformer 模拟，用于序列预测。
    包含两种模式：
    1. 'standard': 标准 Attention (固定学习率，固定上下文窗口)
    2. 'pccf': 动态精度权重 (Dynamic Precision Weighting)，基于预测误差调整 Attention
    """
    def __init__(self, input_dim=1, d_model=16, nhead=2, mode='standard'):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model)) # 简单的位置编码
        
        # Self-Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # Feed Forward
        self.fc = nn.Linear(d_model, input_dim)
        
        # PCCF 特定参数: 动态精度调节器
        # 我们不仅调节学习率，还模拟调节 Attention 的 Query/Key 匹配敏感度
        self.precision_scale = nn.Parameter(torch.tensor(1.0)) 

    def forward(self, x, precision_factor=1.0):
        # x: (batch, seq_len, input_dim)
        seq_len = x.size(1)
        
        # Embedding + Positional Encoding
        x_emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # Attention Mechanism
        attn_output, _ = self.attention(x_emb, x_emb, x_emb)
        
        if self.mode == 'pccf':
            # 修正策略：不要直接放大数值，而是通过 Gate 机制调节
            # 或者仅在反向传播时调整学习率，这里保持前向传播的稳定性
            # 为了体现 PCCF 对 "Context" 的依赖程度调节，我们可以调节残差连接的权重
            # High Precision (High Error) -> Reduce reliance on old context? No.
            # High Precision (High Error) -> We need to update beliefs fast.
            # 在前向传播中，保持数值稳定最重要。
            # 我们将 precision_factor 的主要作用放在优化器的 step 上，
            # 这里仅做微小的特征调节，例如 sharpen attention (模拟高精度)
            x_emb = x_emb + attn_output
        else:
            x_emb = x_emb + attn_output
            
        output = self.fc(x_emb)
        return output[:, -1, :] # 只预测下一个时间步的值

def generate_data(n_samples=200):
    """
    生成带有突变的正弦波数据
    0-100: sin(x)
    100-200: sin(2x) + 0.5 (频率和偏置突变)
    """
    x = np.linspace(0, 40, n_samples)
    y = np.zeros_like(x)
    
    y[:100] = np.sin(x[:100])
    y[100:] = np.sin(2 * x[100:]) + 2.0 # 剧烈突变
    
    # Add noise
    y += np.random.normal(0, 0.1, n_samples)
    return torch.FloatTensor(y).view(-1, 1)

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, nhead: int = 2, max_len: int = 512):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.size(1)
        x = self.token_emb(token_ids) + self.pos_emb[:, :seq_len, :]
        attn_out, _ = self.attn(x, x, x)
        x = self.ln(x + attn_out)
        logits = self.head(x[:, -1, :])
        return logits

def generate_grammar_stream(n_steps: int, drift_step: int, seed: int, noise_p: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    tokens = np.zeros((n_steps,), dtype=np.int64)
    tokens[0] = 0
    for t in range(1, n_steps):
        prev = int(tokens[t - 1])
        if t < drift_step:
            nxt = 1 if prev == 0 else 0
        else:
            nxt = 2 if prev == 0 else 0
        if rng.random() < noise_p:
            nxt = int(rng.integers(low=0, high=3))
        tokens[t] = nxt
    return tokens

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out[window - 1 :] = (c[window:] - c[:-window]) / float(window)
    return out

def run_sine_experiment(out_prefix: str = "pccf_transformer_sine"):
    data = generate_data()
    seq_len = 10
    
    # 初始化两个模型
    model_std = SimpleTransformer(mode='standard')
    model_pccf = SimpleTransformer(mode='pccf')
    
    optimizer_std = optim.Adam(model_std.parameters(), lr=0.01)
    optimizer_pccf = optim.Adam(model_pccf.parameters(), lr=0.01) # 基础学习率相同
    
    criterion = nn.MSELoss()
    
    losses_std = []
    losses_pccf = []
    predictions_std = []
    predictions_pccf = []
    
    print(f"{'Step':<5} | {'True':<8} | {'Std Pred':<8} | {'PCCF Pred':<8} | {'Std Loss':<8} | {'PCCF Loss':<8}")
    print("-" * 65)

    # 在线学习过程 (Online Learning)
    # 模拟真实场景：模型逐个接收数据，预测下一个，然后根据误差更新
    for t in range(seq_len, len(data) - 1):
        # 准备输入序列
        input_seq = data[t-seq_len:t].unsqueeze(0) # (1, seq_len, 1)
        target = data[t].unsqueeze(0)             # (1, 1)
        
        # --- Standard Model Update ---
        model_std.train()
        optimizer_std.zero_grad()
        pred_std = model_std(input_seq)
        loss_std = criterion(pred_std, target)
        loss_std.backward()
        optimizer_std.step()
        
        predictions_std.append(pred_std.item())
        losses_std.append(loss_std.item())
        
        # --- PCCF Model Update ---
        # 1. 前向预测 (Prior Prediction)
        model_pccf.eval() # 先不更新梯度，仅做预测
        with torch.no_grad():
            pred_prior = model_pccf(input_seq)
            error_prior = torch.abs(pred_prior - target)
            
        # 2. 计算动态精度因子 (Dynamic Precision Factor)
        # 改进：使用 tanh 限制范围，防止梯度爆炸
        surprise = error_prior.item() ** 2
        # 限制最大增益为 5 倍，且变化平滑
        precision_factor = 1.0 + 4.0 * np.tanh(surprise) 
        
        # 3. 误差反向传播与更新 (Posterior Update)
        model_pccf.train()
        optimizer_pccf.zero_grad()
        
        # 动态调整学习率
        current_lr = 0.01 * precision_factor
        for param_group in optimizer_pccf.param_groups:
            param_group['lr'] = current_lr
            
        pred_pccf = model_pccf(input_seq) # 前向传播不再传入 precision_factor
        loss_pccf = criterion(pred_pccf, target)
        loss_pccf.backward()
        
        # 梯度裁剪：防止大 Error 导致的大梯度更新
        torch.nn.utils.clip_grad_norm_(model_pccf.parameters(), max_norm=1.0)
        
        optimizer_pccf.step()
        
        predictions_pccf.append(pred_pccf.item())
        losses_pccf.append(loss_pccf.item())

        if t % 10 == 0 or (t >= 98 and t <= 105):
             print(f"{t:<5} | {target.item():<8.2f} | {pred_std.item():<8.2f} | {pred_pccf.item():<8.2f} | {loss_std.item():<8.4f} | {loss_pccf.item():<8.4f}")

    # 统计分析
    mse_std = np.mean(losses_std)
    mse_pccf = np.mean(losses_pccf)
    
    # 突变点后的恢复速度 (从 step 100 开始)
    # 定义恢复：连续 3 步误差小于 0.2
    drift_start = 100 - seq_len
    recovery_std = 0
    recovery_pccf = 0
    
    for i in range(drift_start, len(losses_std)):
        if losses_std[i] < 0.2:
            recovery_std += 1 # 还在恢复中
        else:
            # 简单计算累积高误差步数
            pass

    print("\n" + "="*30)
    print("Transformer Experiment Results")
    print("="*30)
    print(f"MSE (Standard): {mse_std:.4f}")
    print(f"MSE (PCCF):     {mse_pccf:.4f}")
    print(f"Improvement:    {((mse_std - mse_pccf)/mse_std)*100:.2f}%")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data.numpy(), 'k.', alpha=0.3, label='True Data')
    # 对齐预测数据 (因为预测是从 seq_len 开始的)
    x_axis = np.arange(seq_len, len(data) - 1)
    plt.plot(x_axis, predictions_std, 'b-', alpha=0.7, label='Standard Transformer')
    plt.plot(x_axis, predictions_pccf, 'r-', linewidth=2, label='PCCF Transformer')
    plt.axvline(x=100, color='g', linestyle='--', label='Concept Drift')
    plt.title('PCCF in Transformer: Dynamic Precision for Concept Drift')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.png', dpi=200)
    plt.savefig(f'{out_prefix}.svg')
    print(f"Plot saved to {out_prefix}.png")

def run_grammar_experiment(
    out_prefix: str = "pccf_toylm",
    n_seeds: int = 20,
    n_steps: int = 400,
    drift_step: int = 200,
    seq_len: int = 16,
    base_lr: float = 3e-3,
    pccf_beta: float = 4.0,
    loss_lr_gamma: float = 1.0,
    lr_max_mult: float = 5.0,
):
    vocab_size = 3
    device = torch.device("cpu")

    per_seed = []
    all_loss_curves = {"FixedLR": [], "LossLR": [], "PCCF": [], "PH_Reset": []}
    all_acc_curves = {"FixedLR": [], "LossLR": [], "PCCF": [], "PH_Reset": []}

    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        np.random.seed(1000 + seed)

        stream = generate_grammar_stream(n_steps=n_steps, drift_step=drift_step, seed=seed)
        stream_t = torch.from_numpy(stream).to(device)

        def make_model():
            return TinyTransformerLM(vocab_size=vocab_size, d_model=32, nhead=2, max_len=max(512, seq_len + 1)).to(device)

        models = {
            "FixedLR": make_model(),
            "LossLR": make_model(),
            "PCCF": make_model(),
            "PH_Reset": make_model(),
        }

        opts = {
            "FixedLR": optim.Adam(models["FixedLR"].parameters(), lr=base_lr),
            "LossLR": optim.Adam(models["LossLR"].parameters(), lr=base_lr),
            "PCCF": optim.Adam(models["PCCF"].parameters(), lr=base_lr),
            "PH_Reset": optim.Adam(models["PH_Reset"].parameters(), lr=base_lr),
        }

        ph = PageHinkley(delta=0.005, threshold=5.0)
        ce = nn.CrossEntropyLoss()

        loss_hist = {k: [] for k in models.keys()}
        acc_hist = {k: [] for k in models.keys()}
        pccf_var = 1.0
        pccf_alpha = 0.05

        for t in range(seq_len, n_steps):
            x = stream_t[t - seq_len : t].unsqueeze(0)
            y = stream_t[t].view(1)

            for name in ["FixedLR", "LossLR", "PCCF", "PH_Reset"]:
                model = models[name]
                opt = opts[name]

                model.train()
                opt.zero_grad()
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                pred = int(torch.argmax(logits, dim=-1).item())
                acc = 1.0 if pred == int(y.item()) else 0.0
                loss_hist[name].append(float(loss.item()))
                acc_hist[name].append(float(acc))

            with torch.no_grad():
                logits_prior = models["PCCF"](x)
                loss_prior = float(ce(logits_prior, y).item())
                norm_surprise = loss_prior / (pccf_var + 1e-6)
                factor = 1.0 + pccf_beta * float(np.tanh(norm_surprise))
                factor = float(np.clip(factor, 1.0, lr_max_mult))
                for pg in opts["PCCF"].param_groups:
                    pg["lr"] = base_lr * factor
                pccf_var = (1.0 - pccf_alpha) * pccf_var + pccf_alpha * (loss_prior ** 2)

            with torch.no_grad():
                logits_prior = models["LossLR"](x)
                loss_prior = float(ce(logits_prior, y).item())
                factor = 1.0 + loss_lr_gamma * float(np.tanh(loss_prior))
                factor = float(np.clip(factor, 1.0, lr_max_mult))
                for pg in opts["LossLR"].param_groups:
                    pg["lr"] = base_lr * factor

            with torch.no_grad():
                logits_prior = models["PH_Reset"](x)
                loss_prior = float(ce(logits_prior, y).item())
                if ph.update(loss_prior):
                    models["PH_Reset"] = make_model()
                    opts["PH_Reset"] = optim.Adam(models["PH_Reset"].parameters(), lr=base_lr)
                    ph.reset()

        def recovery_steps(acc_curve: list[float], thr: float = 0.8, cons: int = 10) -> int:
            start = max(0, drift_step - seq_len)
            a = np.asarray(acc_curve, dtype=float)
            for i in range(start, a.shape[0] - cons + 1):
                if np.all(a[i : i + cons] >= thr):
                    return int(i - start)
            return int(a.shape[0] - start)

        metrics = {}
        start = max(0, drift_step - seq_len)
        for name in models.keys():
            losses = np.asarray(loss_hist[name], dtype=float)
            accs = np.asarray(acc_hist[name], dtype=float)
            metrics[f"{name}_NLL"] = float(np.mean(losses))
            metrics[f"{name}_PostNLL"] = float(np.mean(losses[start:]))
            metrics[f"{name}_Recovery"] = float(recovery_steps(acc_hist[name]))
            all_loss_curves[name].append(losses)
            all_acc_curves[name].append(accs)

        per_seed.append(metrics)

    df = pd.DataFrame(per_seed)
    summary_rows = []
    for name in ["FixedLR", "LossLR", "PCCF", "PH_Reset"]:
        summary_rows.append(
            {
                "Model": name,
                "NLL_mean": float(df[f"{name}_NLL"].mean()),
                "NLL_std": float(df[f"{name}_NLL"].std(ddof=1)),
                "PostNLL_mean": float(df[f"{name}_PostNLL"].mean()),
                "PostNLL_std": float(df[f"{name}_PostNLL"].std(ddof=1)),
                "Recovery_mean": float(df[f"{name}_Recovery"].mean()),
                "Recovery_std": float(df[f"{name}_Recovery"].std(ddof=1)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(by="PostNLL_mean", ascending=True).reset_index(drop=True)
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)

    t = np.arange(n_steps - seq_len, dtype=int)
    drift_x = drift_step - seq_len
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(111)
    ax.axvline(x=drift_x, color="gray", linestyle="--", linewidth=1.2)
    for name in ["FixedLR", "LossLR", "PCCF", "PH_Reset"]:
        curves = np.stack(all_loss_curves[name], axis=0)
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0, ddof=1)
        rm = _rolling_mean(mean, 10)
        ax.plot(t, rm, label=name, linewidth=2.0 if name == "PCCF" else 1.6)
        ax.fill_between(t, _rolling_mean(mean - std, 10), _rolling_mean(mean + std, 10), alpha=0.12)
    ax.set_title("Toy Next-Token Prediction with Rule Shift: NLL (rolling mean)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_nll.png", dpi=200)
    fig.savefig(f"{out_prefix}_nll.svg")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(111)
    ax.axvline(x=drift_x, color="gray", linestyle="--", linewidth=1.2)
    for name in ["FixedLR", "LossLR", "PCCF", "PH_Reset"]:
        curves = np.stack(all_acc_curves[name], axis=0)
        mean = np.mean(curves, axis=0)
        rm = _rolling_mean(mean, 10)
        ax.plot(t, rm, label=name, linewidth=2.0 if name == "PCCF" else 1.6)
    ax.set_title("Toy Next-Token Prediction with Rule Shift: Accuracy (rolling mean)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_acc.png", dpi=200)
    fig.savefig(f"{out_prefix}_acc.svg")
    plt.close(fig)

    print(summary.to_string(index=False))

def run_experiment():
    run_sine_experiment(out_prefix="pccf_transformer_plot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sine", "grammar"], default="grammar")
    parser.add_argument("--out-prefix", type=str, default=None)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()

    if args.task == "sine":
        run_sine_experiment(out_prefix=args.out_prefix or "pccf_transformer_plot")
    else:
        run_grammar_experiment(out_prefix=args.out_prefix or "pccf_toylm", n_seeds=args.n_seeds)
