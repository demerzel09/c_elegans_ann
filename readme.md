

# no official implementation of "An Artificial Neural Network for Image Classification Inspired by Aversive Olfactory Learning Circuits in Caenorhabditis Elegans"  

https://arxiv.org/abs/2409.07466  

Since there is no official implementation, I created an unofficial implementation.
It was created using Chat-GPT5.
The accuracy level is unknown because I have not investigated the details.

In validation, the accuracy was 91.92% with default settings. This is too high compared to the 85% accuracy in the original paper.

The CNN portion may be better than the ANN in the paper. It may be necessary to change the ANN to a CNN and compare the performance.
I have not thoroughly investigated whether the performance is superior to a similar CNN.
I would appreciate your feedback.  

(For Japanese)  
公式実装がないため、非公式実装を作成しました。  
Chat-GPT5 を用いて作成しました。詳細を調査していないため、精度は不明です。  

検証では、デフォルト設定で91.92%の精度が出ました。これは、元の論文の85%の精度と比較すると高すぎます。  

CNN部分は、論文のANNよりも優れている可能性があります。ANNをCNNに変更して性能比較する必要があるかもしれません。  
同様のCNNと比較して性能が優れているかどうかは、十分に調査していません。  
フィードバックをお待ちしております。  

## Installation

```bash
pip install -U pip
pip install -r requirements.txt
```

## How to Run
```bash
python main.py
```

## BP と PC-ALM の比較

`compare_bp_pcalm.py` で同じ `NematodeCircuitCNN` を BP と PC-ALM で学習できます。
PC-ALM は [Sakana AI の論文](https://arxiv.org/abs/2605.31022) の Algorithm 1 と
付録の feedforward graph extension を、このプロジェクトの分岐・合流する CNN に適用した実験実装です。
論文の公式実装の直接移植や、論文の実験結果の再現ではありません。

BP と PC-ALM は個別のコマンドで実行します。VS Code の実行設定も手法ごとに分けています。
`--methods` は必須で、省略した場合は学習を開始しません。

短い CIFAR-10 比較（各コマンドを別々に実行）:

```powershell
python compare_bp_pcalm.py --methods bp --epochs 3 --c 8 --bs 32 --train-samples 2048 --test-samples 1000 --output results/bp_pilot
python compare_bp_pcalm.py --methods pcalm --epochs 3 --c 8 --bs 32 --train-samples 2048 --test-samples 1000 --output results/pcalm_pilot
```

初期実装での動作確認結果（RTX 4060、seed 42、既定結線、幅8、FP32、
学習率0.1、16反復、状態学習率0.01、乗数更新率0.1）:

| 手法 | 3 epoch 後のテスト精度 | 合計学習時間 |
| --- | ---: | ---: |
| BP | 34.8% | 4.22 秒 |
| PC-ALM | 27.5% | 81.51 秒 |

少量データ・小さい幅・短期間での試行です。PC-ALM の設定は未調整のため、
この結果だけで十分に学習したモデルの優劣は判断できません。

元のチャネル幅で全データを使う比較（PC-ALM の反復により時間がかかります）:

```powershell
python compare_bp_pcalm.py --methods bp --epochs 200 --c 48 --bs 32 --output results/bp_full
python compare_bp_pcalm.py --methods pcalm --epochs 200 --c 48 --bs 32 --output results/pcalm_full
```

両方式は初期重み、学習データの順番・拡張、回路、交差エントロピー、label smoothing、
optimizer、学習率スケジュール、勾配クリップをそろえます。現在の共通の既定学習率は0.01です。
モデルと状態の演算は FP32、PC-ALM の目的関数比較の集計は FP64 を使います。
元のスクリプトの AMP を使う BP と実行時間が同じになるわけではありません。
`--edges-json edges.json` で比較する両方の回路を指定できます。
指定しない場合は元のモデルと同じ既定のダミー結線です。
`--methods bp` または `--methods pcalm` で片方だけ実行できます。
`--output` を省略した場合も、それぞれ `results/bp/` と `results/pcalm/` に分けて保存します。

PC-ALM は各回路ノード、stem、後段 CNN、隠れ全結合ブロックの出力を独立した状態として扱い、
予測誤差 `r = h - f(parents)` に対する拡張ラグランジアンを最適化します。
`--steps` は状態更新回数（既定64）、`--state-lr` は状態の最大学習率（0.03）、
`--dual-lr` は乗数の更新率（1.0）、`--rho` は二次ペナルティ係数（1.0）です。
これらはこの CNN で最適化済みの値ではありません。
状態更新は同時更新し、更新後の誤差で乗数を更新し、最後は状態更新だけを行います。
各反復で状態を detach し、重み更新は各ブロックの局所的な計算グラフから求めます。
PyTorch autograd はブロック内の微分に使用します。

初期設定では、幅48のモデルで入力側の勾配がほぼゼロとなり、全データの学習で
PC-ALM の精度が48.7%から10%へ低下しました。現在の既定設定は反復回数と
乗数更新率を増やし、共通の重み学習率を下げています。
固定した乗数に対する拡張ラグランジアンが減少するかを各状態更新で確認し、
増える場合は状態学習率を半分にする backtracking を使用します。
この状態更新は論文の固定刻み更新を安定化した変種です。
`--no-backtracking` で固定刻みに戻せます。
反復回数が増えるため、初期設定より学習時間が長くなります。

幅48の短い検証は、テストセットを調整に使わず、学習データから512枚を学習用、
別の256枚を検証用として分けます。次のコマンドで実行できます。

```powershell
python compare_bp_pcalm.py --methods pcalm --epochs 10 --c 48 --bs 32 --train-samples 512 --validation-samples 256 --lr 0.01 --steps 64 --state-lr 0.03 --dual-lr 1 --backtracking --output results/pcalm_stability_check
```

この設定の10 epoch確認では、PC-ALM の検証精度は最終19.5%、最高21.1%、
更新前の学習損失は2.307から2.109へ低下しました。少量データでの検証であり、
全データの長期学習で安定することや、BP と同程度の最終精度を保証する結果ではありません。
同じ幅・学習率・データ・epoch数の BP は最終27.0%、最高27.7%でした。

`--validation-samples` を使った実行の `test_accuracy` 列は、この学習内の検証データに対する精度です。
一度20%以上に学習した後、精度が12%以下かつ学習損失が2.25以上に3 epoch連続で落ちた場合は、
最新モデルを保存して停止します。`--collapse-patience` で連続回数を変更でき、0なら停止判定を無効にします。

この適用では元の交差エントロピーを維持します（論文の基本式は二乗誤差）。
BatchNorm は学習時のバッチ統計を使用し、running statistics は各バッチで一度だけ更新します。
Dropout は同じバッチの状態更新・乗数更新・重み更新すべてで同じマスクを再使用します。
評価には通常の forward を使い、正解ラベルや PC-ALM の状態更新を使いません。

出力ディレクトリに以下を保存します:

- `config.json`: 設定、データ数、回路、PyTorch バージョン。
- `metrics.csv`: 各 epoch の更新前の学習損失、テスト精度、学習時間、CUDA ピークメモリ、PC-ALM の予測誤差 RMS。
- PC-ALM の `state_lr_min` と `backtracks`: 実際に使った最小の状態学習率と、刻み幅を下げた回数。
- `summary.json`: 最終・最高テスト精度と合計学習時間。
- `bp.pt` / `pcalm.pt`: 各 epoch 終了時の最新チェックポイント。
- `bp_best.pt` / `pcalm_best.pt`: テスト精度が改善したときのチェックポイント（新規実行は最初の epoch も保存）。

### PC-ALM が崩れた原因の切り分け

前の安定化修正は複数の設定を同時に変更しているため、原因を特定する証拠にはなりません。
また、10 epochで減衰する短期検証の学習率スケジュールも、200 epoch実行とは違います。
`run_pcalm_ablation.py` は下の設定を一つずつ独立して実行するための入口です。

| case | 重みの学習率 | 反復回数 | 状態の学習率 | 乗数更新率 | backtracking |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.1 | 16 | 0.01 | 0.1 | 無効 |
| lr_only | 0.01 | 16 | 0.01 | 0.1 | 無効 |
| steps_only | 0.1 | 64 | 0.01 | 0.1 | 無効 |
| state_lr_only | 0.1 | 16 | 0.03 | 0.1 | 無効 |
| dual_lr_only | 0.1 | 16 | 0.01 | 1.0 | 無効 |
| backtracking_only | 0.1 | 16 | 0.01 | 0.1 | 有効 |
| inference_bundle | 0.1 | 64 | 0.03 | 1.0 | 無効 |
| lr_inference | 0.01 | 64 | 0.03 | 1.0 | 無効 |
| combined | 0.01 | 64 | 0.03 | 1.0 | 有効 |

共通条件は幅48、batch size32、seed42、SGD、同じ回路・初期重み・データ順・拡張・
label smoothing・weight decay・勾配クリップです。初期重みの一致は `config.json` の
`initial_model_sha256` でも確認できます。CIFAR-10の学習データを45,000枚の学習用と
5,000枚の検証用に分け、テストセットは調整に使いません。
従来の崩れた実行とデータ分割は変わるため、最初に baseline がこの条件で崩れるかを確認します。
baseline が崩れなければ、元の失敗の原因を説明したという結論にはできません。

200 epoch予定のwarmup + cosine scheduleをそのまま使い、最初の12 epochで一旦停止します。
同じ更新回数で比較するため、この実験では精度低下による自動停止を無効にしています。
NaNなどの数値異常は停止します。実データ実験の完了状況は下記と `suite_status.json` を確認してください。

#### 途中結果（2026-10-03、seed42）

初期重みのハッシュは一致し、保存設定の差は学習率・条件名・出力先のみでした。
同じ200 epoch予定のスケジュールで、12 epochまで実行した結果です。

| 条件 | 最高検証精度 | 12 epoch検証精度 | 崩れの検出 |
| --- | ---: | ---: | --- |
| baseline（学習率0.1） | 47.18% | 9.56% | 9 epoch |
| lr_only（学習率0.01） | 50.72% | 50.72% | 12 epochまでなし |

この条件では、学習率を下げるだけで12 epochまでの崩れを防げました。
ただし、両条件とも観測した入力層の学習勾配はほぼゼロです。
「入力側に勾配が届かない」だけでは、崩れる条件と崩れない条件の違いを説明できません。

入力層の重みノルムは12 epoch最初のバッチ時点で、baselineは初期の0.2855%、
lr_onlyは55.7754%でした。学習勾配をゼロと仮定し、実際の学習率・更新回数・
SGD momentum/Nesterov・weight decayだけで計算した縮小率と、両方とも相対誤差約0.00012%以内で一致します。
計算結果は `results/pcalm_ablation/weight_decay_analysis.json` に保存しています。
そのため「入力側の小さい学習勾配に対してweight decayが支配的になり、
高い学習率では特徴抽出層が急速に縮む」という機序が有力です。
重みの縮小との一致は、精度低下の原因を単独で証明するものではありません。

この仮説を直接検証する `weight_decay_only` を追加しました。
baselineの学習率0.1・16反復を維持し、weight decayだけを0にします。
下のコマンドで `weight_decay_only` を個別に実行できます。
この追加条件はまだ実行していません。実行中の9条件とGPUを重ねないでください。

```powershell
python run_pcalm_ablation.py --case weight_decay_only
```

現在はsteps_onlyを実行中で、元の9条件は順次進みます。
単一seed・12 epochの結果なので、低学習率で長期の崩れを防げるか、
この機序が別seedでも再現するかは未確認です。

```powershell
python run_pcalm_ablation.py --case baseline
python run_pcalm_ablation.py --case lr_only
python run_pcalm_ablation.py --case steps_only
```

切り分け実験は上のコマンドで `--case` に条件を指定して実行します。
`launch.json` は「BP」「PC-ALM」の2件です。起動時に「新規」「再開」を選びます。
PC-ALMの新規実行は学習率0.01、16反復、状態学習率0.01、乗数更新率0.1、backtrackingなしです。
学習率だけを変更した切り分け条件を使い、保存先は `results/pcalm_lr_only_full/` です。
「再開」もこの保存先のモデルを読みます。以前の64反復の実行は `results/pcalm_full/` に残します。
一回の起動で一条件だけ実行し、結果は `results/pcalm_ablation/<case>_seed42/` に保存します。
再開して次の12 epochを実行する場合は `--resume` を付けます。
数値異常で停止した条件も含め、完了 epoch数が揃っているか確認してください。

```powershell
python run_pcalm_ablation.py --case baseline --resume
python run_pcalm_ablation.py --list
```

`metrics.csv` では精度・学習損失・崩れたepoch・状態の最小刻み幅を比較します。
`gradients.csv` では9個の重みブロックについて、各epoch最初のバッチの勾配ノルム、
BPとのノルム比・コサイン類似度・重みノルムを記録します。
参照BPの計算は観測だけに使い、PC-ALMの重み更新には混ぜません。
乱数とBatchNorm状態を分離し、計測を入れても学習結果が変わらないことをテストで確認しています。
最初のバッチだけの観測にはばらつきがあるため、複数epoch・複数seedで確認してください。

判定は、まずbaselineと各単独変更で効果を比較し、次に
baseline / lr_only / inference_bundle / lr_inference の4条件で
学習率と反復設定の相互作用を確認します。
lr_inference / combined の比較では、変更後の設定に対するbacktrackingの効果を確認できます。
単独変更が効かず組み合わせだけが効く場合は、一つの要因に原因を決めつけません。
効果のある条件は seed43・44 でも再実行します（`--seed`で指定）。

設定と保存・計測の接続だけを確認する場合:

```powershell
python run_pcalm_ablation.py --case baseline --smoke --run-epochs 1 --device cpu --output-root results/ablation_smoke
```

このFakeData確認は、原因の切り分け結果としては扱いません。

全条件を順番に進め、途中の進捗と比較表も保存する場合:

```powershell
python run_ablation_suite.py --cases baseline lr_only steps_only state_lr_only dual_lr_only backtracking_only inference_bundle lr_inference combined --device cuda
```

この入口は指定した条件をGPUで一つずつ実行します。まずbaselineとlr_onlyを12 epochまで比較し、
baselineの崩れが再現した場合に残りへ進みます。再現しなければ`needs_reproduction`として停止します。
同じコマンドで再起動した場合、設定が一致する保存済みチェックポイントから不足epochだけを再開し、
12 epoch完了済みの条件はスキップします。実行中の同じ実験を重ねて起動しないでください。
元の一括比較のモデルと今回の実験モデルは別のフォルダにあります。

`results/pcalm_ablation/` に以下を保存します:

- `suite_status.json`: 実行中の条件、プロセスID、完了epoch、結果。
- `suite_summary.csv`: 全条件の比較表（実行中も更新）。
- `suite_findings.json`: 完了した条件から計算した暫定の単独効果・相互作用。単一seedのため原因の確定ではありません。
- `<case>_seed42.log`: 各条件の実行ログ。
- `<case>_seed42/`: 各条件のモデル、設定、精度・勾配の記録。

基準設定でも1条件は数時間、反復回数を増やした条件はさらに時間がかかる見込みです。
各epoch終了時に保存するため、中断した場合も最後に完了したepochから再開できます。

### 保存したモデルからの再開

モデル名は「出力フォルダ + 手法」で固定です。全データの個別実行では
`results/bp_full/bp.pt` と `results/pcalm_full/pcalm.pt` を更新します。
同じ出力フォルダで新規実行すると、以前のファイルを更新するため、
別の実験を残す場合は `--output` を変えてください。

新しいチェックポイントには、重み・完了 epoch・optimizer・scheduler・乱数状態・
DataLoader の乱数状態・履歴を保存します。各 epoch 終了時に一時ファイルから置換し、
次回は保存済み epoch の次から再開します。epoch の途中で停止した場合は、直前に完了した epoch からの再開です。
PC-ALM の状態と乗数はバッチごとに初期化するので、epoch 間の再開では復元する必要がありません。

中断した個別実行を、元の予定 epoch 数まで再開するコマンド:

```powershell
python compare_bp_pcalm.py --methods bp --resume results/bp_full/bp.pt
python compare_bp_pcalm.py --methods pcalm --resume results/pcalm_full/pcalm.pt
```

`launch.json` の「BP」で「再開」を選んだ場合も同じコマンドです。
「PC-ALM」の再開先は、新しい16反復の実行の `results/pcalm_lr_only_full/pcalm.pt` です。
指定ファイルがない場合は、別のモデルを自動選択せず、エラーにします。
再開時の回路・幅・バッチサイズ・損失・optimizer・元の学習期間などは保存された設定から読み戻します。
デバイスと出力先は `--device`、`--output` で変更できます。
実行環境・GPU・PyTorch バージョンを変えた場合は、数値の完全一致は保証されません。

既に完了した個別モデルから、さらに50 epoch学習する場合:

```powershell
python compare_bp_pcalm.py --methods bp --resume results/bp_full/bp.pt --extra-epochs 50 --restart-lr 0.01
python compare_bp_pcalm.py --methods pcalm --resume results/pcalm_full/pcalm.pt --extra-epochs 50 --restart-lr 0.01
```

`--extra-epochs 50` は保存済み epoch から50回追加する指定です。
optimizer の状態は保ち、学習率を `--restart-lr` から始める新しい cosine schedule に切り替えます。
`--restart-lr` を省略すると、保存設定の初期学習率を使います。
追加学習で反復設定を変える場合は `--steps`、`--state-lr`、`--dual-lr`、`--rho`、
`--backtracking` を明示してください。通常の中断再開では保存済み設定を維持します。
backtracking の設定が保存されていない旧モデルでは、固定刻みとして読み込みます。
例えば200 epoch完了後なら201〜250 epochを学習します。
完了済みモデルで `--extra-epochs` を省略した場合は、追加指定が必要というエラーになります。

今回の既存の一括比較実行で保存されるモデルから追加学習する場合:

```powershell
python compare_bp_pcalm.py --methods bp --resume results/bp_pcalm_full/bp.pt --extra-epochs 50 --restart-lr 0.01 --output results/bp_full
python compare_bp_pcalm.py --methods pcalm --resume results/bp_pcalm_full/pcalm.pt --extra-epochs 50 --restart-lr 0.01 --steps 64 --state-lr 0.03 --dual-lr 1 --backtracking --output results/pcalm_full
```

既存モデルからの追加学習は、上のコマンドを使います（モデルファイルが必要です）。
旧形式には optimizer・scheduler・乱数状態がないため、重みから追加学習し、これらの状態は新しく作ります。
旧形式には最高精度時の重みもないため、その時点のモデルを復元することはできません。
実行中のプロセスは旧コードを読み込み済みなので、今回の実行の保存タイミングは従来どおり各方式の終了時です。
毎 epoch の保存は、この変更後に起動する実行から適用されます。

学習時間にはデータ読み込みを含み、評価は含みません。GPU の初回初期化も含むため、
非常に短い試行の時間は参考値です。テスト精度の最高値は参考情報とし、
パラメータ調整は学習データから分けた検証データで行ってください。
優劣を判断するには十分な学習期間と複数 seed が必要です。
同じ epoch 数での比較に加えて、同じ時間内での精度も `metrics.csv` から比較できます。

ダウンロード不要の動作確認と数値検証:

```powershell
python compare_bp_pcalm.py --methods bp --synthetic --epochs 1 --c 2 --bs 4 --train-samples 8 --test-samples 8 --steps 4 --output results/bp_smoke
python compare_bp_pcalm.py --methods pcalm --synthetic --epochs 1 --c 2 --bs 4 --train-samples 8 --test-samples 8 --steps 4 --output results/pcalm_smoke
python -m unittest test_pcalm test_compare_resume test_pcalm_ablation test_ablation_suite -v
```

`--synthetic` の精度はランダム画像に対する動作確認で、実データの性能を示しません。

## 以前の BP 実行結果

```bash
[005/200] acc=0.7340 best=0.7340 lr=0.10000　　
[010/200] acc=0.7708 best=0.7795 lr=0.09984　　
[020/200] acc=0.8026 best=0.8068 lr=0.09855　　
[030/200] acc=0.8002 best=0.8197 lr=0.09600　　
[040/200] acc=0.8253 best=0.8298 lr=0.09226　　
[050/200] acc=0.8238 best=0.8396 lr=0.08743　　
[060/200] acc=0.8368 best=0.8585 lr=0.08162　　
[070/200] acc=0.8344 best=0.8585 lr=0.07500　　
[080/200] acc=0.8463 best=0.8636 lr=0.06773　　
[090/200] acc=0.8552 best=0.8636 lr=0.06000　　
[100/200] acc=0.8552 best=0.8816 lr=0.05201　　
[110/200] acc=0.8615 best=0.8816 lr=0.04397　　
[120/200] acc=0.8762 best=0.8816 lr=0.03609　　
[130/200] acc=0.8859 best=0.8866 lr=0.02857　　
[140/200] acc=0.8780 best=0.8977 lr=0.02160　　
[150/200] acc=0.9020 best=0.9020 lr=0.01536　　
[160/200] acc=0.8964 best=0.9059 lr=0.01003　　
[170/200] acc=0.9092 best=0.9104 lr=0.00573　　
[180/200] acc=0.9167 best=0.9167 lr=0.00257　　
[190/200] acc=0.9192 best=0.9192 lr=0.00065　　
[199/200] acc=0.9167 best=0.9192 lr=0.00001　　
```
