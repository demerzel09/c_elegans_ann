

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

## Results

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