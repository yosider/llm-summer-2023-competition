松尾研究室主催の公開講座「サマースクール 2023 大規模言語モデル」最終課題のコンペティション参加時のコードです。  
内容の詳細については`report.md`も参照ください。  
実行するには講座で配布されたjsonデータが必要です。

## Installation
Python 3.10.13, CUDA 11.8で動作確認済み
- Pythonパッケージインストール
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```
- [Serper](https://serper.dev/)のAPI keyを取得し、`.env`ファイルに記述
```
SERPER_API_KEY=xxx
```
- (任意)環境変数`HF_HOME`を設定
```bash
export HF_HOME=/mnt/ms-nas-2/yoshida/huggingface/
```
- (fine-tuningする場合)wandbにログイン  
  (現状、openorca-stxのJCommonsenseQAによるfine-tuningのみ対応しています)
```bash
wandb login
```

## Run
`data/`以下にデータを配置し、`main.py`内でファイル名を指定した後、実行してください。
```python
python main.py
```
