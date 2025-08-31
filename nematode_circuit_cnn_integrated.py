# -*- coding: utf-8 -*-
"""
NematodeCircuitCNN (完全統合版)
- 論文: "An Artificial Neural Network for Image Classification Inspired by Aversive
  Olfactory Learning Circuits in Caenorhabditis elegans" (arXiv:2409.07466)
- 仕様反映:
  * Fig.3 構成に沿って、回路モジュール(22ノード/21エッジ) -> Concat(M層) ->
    Conv(c) -> MaxPool(2) -> Conv(2c) -> MaxPool(2) -> Dropout2d(0.1) -> MLP(300→200→10)
  * CIFAR-10 用の学習レシピ強化: RandomCrop+Flip、Label Smoothing、SGD+Momentum、
    CosineAnnealingLR + Warmup、AMP対応
- 使い方:
    python nematode_circuit_cnn_integrated.py --epochs 200 --c 48 --bs 128 --lr 0.1 \
        --optimizer sgd --autoaugment 0
  ※ GPU 推奨 (torch.cuda.amp 対応)。

- EDGES について:
  * 既定はダミーの 21 本です。Fig.2(d) を忠実に再現する場合は、
    EDGES を差し替えてください (S→I および I→M の DAG)。
  * --edges_json で JSON ファイル (例: [["s0","i1"], ["i2","m3"], ...]) から読み込めます。
"""

import os
import json
import math
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# ========== 再現性 ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True は遅いので推奨しない
    torch.backends.cudnn.benchmark = True

# ========== 回路定義（22ノード・21結線） ==========
# レイヤ別にノードID: S(感覚)=s0..s9, I(介在)=i0..i4, M(運動)=m0..m6
S_NODES = [f"s{k}" for k in range(10)]
I_NODES = [f"i{k}" for k in range(5)]
M_NODES = [f"m{k}" for k in range(7)]
NODES = S_NODES + I_NODES + M_NODES

# デフォルトのダミー結線（21本）。必要に応じて差し替え。
EDGES_DEFAULT: List[Tuple[str, str]] = [
    ("s0","i0"),("s1","i0"),("s2","i1"),("s3","i1"),
    ("s4","i2"),("s5","i2"),("s6","i3"),("s7","i3"),
    ("s8","i4"),("s9","i4"),("s0","i2"),("s1","i3"),  # S→I 多様性
    ("i0","m0"),("i0","m1"),("i1","m2"),("i2","m3"),
    ("i3","m4"),("i4","m5"),("i2","m6"),("i1","m4"),("i3","m6")
]
assert len(EDGES_DEFAULT) == 21

# ========== モジュール定義 ==========
class ConvModule(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # Kaiming init
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class NematodeCircuitCNN(nn.Module):
    """
    Stem(ConvModule)->MaxPool(2) ->
    Functional circuit (22ノードDAG; 各ノードは ConvModule(c,c); エッジは和) ->
    Concat(M層7ノード) -> [Conv(7c->c)→Pool→Conv(c->2c)→Pool→Dropout2d] ->
    Flatten -> MLP(300→200→num_classes)
    """
    def __init__(self, in_ch=3, num_classes=10, c=48, k=3,
                 edges: List[Tuple[str, str]] = None):
        super().__init__()
        self.c = c
        self.edges = edges if edges is not None else EDGES_DEFAULT

        # Stem
        self.stem = ConvModule(in_ch, c, k=3)
        self.pool1 = nn.MaxPool2d(2)  # 32→16

        # ノード演算子: 全ノード (S, I, M) ともに ConvModule(c->c)
        self.node_ops = nn.ModuleDict({n: ConvModule(c, c, k=k) for n in NODES})

        # 親リスト（受信元）
        self.parents = {n: [] for n in NODES}
        for u, v in self.edges:
            self.parents[v].append(u)

        # 後段: Concat 後のチャネルは 7*c
        self.post = nn.Sequential(
            ConvModule(7*c, c, k=3),       # 16x16 -> 16x16
            nn.MaxPool2d(2),               # 16→8
            ConvModule(c, 2*c, k=3),
            nn.MaxPool2d(2),               # 8→4
            nn.Dropout2d(0.1),             # DP2d(p=0.1)
        )

        # MLP ヘッド: 入力次元は LazyLinear で自動
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(300), nn.ReLU(inplace=True),
            nn.Linear(300, 200), nn.ReLU(inplace=True),
            nn.Linear(200, num_classes)
        )

    @torch.no_grad()
    def _toposort(self) -> List[str]:
        indeg = {n: 0 for n in NODES}
        for u, v in self.edges:
            indeg[v] += 1
        order, q = [], [n for n in NODES if indeg[n] == 0]
        while q:
            n = q.pop(0)
            order.append(n)
            for (u, v) in self.edges:
                if u == n:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
        if len(order) != len(NODES):
            raise RuntimeError("EDGES がDAGになっていません (閉路がある可能性)")
        return order

    def forward(self, x):
        x = self.pool1(self.stem(x))  # [B, c, 16, 16]
        feats = {}
        order = self._toposort()  # S→I→M の順になるはず

        # S層: stem 出力を全Sノードの入力とする
        for n in order:
            if n in S_NODES:
                h_in = x
            else:
                ps = self.parents[n]
                assert len(ps) > 0, f"{n} への親エッジが必要です"
                # 親ノードの出力を加算
                h_in = feats[ps[0]]
                for p in ps[1:]:
                    h_in = h_in + feats[p]
            feats[n] = self.node_ops[n](h_in)

        # M層出力を concat
        m_feat = torch.cat([feats[m] for m in M_NODES], dim=1)  # [B, 7c, 16,16]
        y = self.post(m_feat)  # -> [B, 2c, 4,4]
        out = self.fc(y)       # -> [B, num_classes]
        return out

# ========== データセット & 変換 ==========

def build_transforms(use_autoaugment: bool = False):
    t_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_autoaugment:
        t_list.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
    t_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
    ])
    train_tf = transforms.Compose(t_list)
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
    ])
    return train_tf, test_tf

# ========== 学習ユーティリティ ==========

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

# ========== メイン学習ループ ==========

def train_cifar10(
    epochs=200,
    c=48,
    bs=128,
    lr=0.1,
    optimizer_name="sgd",
    weight_decay=5e-4,
    label_smoothing=0.1,
    autoaugment=False,
    edges_json: str = None,
    seed=42,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # EDGES の読み込み
    edges = None
    if edges_json and os.path.exists(edges_json):
        with open(edges_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
            edges = [(str(u), str(v)) for (u, v) in raw]
        assert len(edges) == 21, "edges_json は 21 本のエッジにしてください"

    # データ
    train_tf, test_tf = build_transforms(use_autoaugment=autoaugment)
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )

    # モデル
    model = NematodeCircuitCNN(in_ch=3, num_classes=10, c=c, k=3, edges=edges).to(device)

    # Optimizer
    if optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optimizer_name.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4 if lr is None else lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be 'sgd' or 'adamw'")

    # Scheduler: Warmup(5ep) + Cosine(残り)
    warmup_epochs = max(1, min(10, epochs // 40))  # 200epなら5ep程度
    sched_warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[sched_warm, sched_cos], milestones=[warmup_epochs])

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            # Unscale -> Clip -> Step の順序を厳守
            if device == "cuda":
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

        acc = evaluate(model, test_loader, device)
        best = max(best, acc)
        sched.step()
        print(f"[{ep:03d}/{epochs}] acc={acc:.4f} best={best:.4f} lr={opt.param_groups[0]['lr']:.5f}")

    print("done. best acc:", best)
    return model

# ========== CLI ==========

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--c', type=int, default=48, help='チャネル幅 (32/48/64 など)')
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1, help='SGD用の初期学習率 (AdamWなら 3e-4 推奨)')
    p.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--autoaugment', type=int, default=0, help='1 で AutoAugment(CIFAR10) を有効化')
    p.add_argument('--edges_json', type=str, default=None, help='[ ["s0","i1"], ["i2","m3"], ... ] のJSONファイル')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_cifar10(
        epochs=args.epochs,
        c=args.c,
        bs=args.bs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        autoaugment=bool(args.autoaugment),
        edges_json=args.edges_json,
        seed=args.seed,
    )
