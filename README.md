# garch_origin

標準的な `GARCH(1,1)` を使った生成 baseline です。

この baseline では、`sp500` と `DGS10` の各系列に単変量 GARCH(1,1) を最尤推定し、標準化残差の経験相関を使って2資産の同時系列を生成します。TimeGAN のようなニューラルネット学習ではなく、統計モデルのパラメータ推定なので CPU で十分です。GPU は使いません。

## ディレクトリ構成

```text
garch_origin/
  config/
    garch11_baseline.json
    garch11_correlogram_design.json
  mahalanobis_eval/
    scripts/run_mahalanobis_eval.py
  scripts/
    garch_utils.py
    generate_garch.py
    evaluate_with_mahalanobis.py
    plot_train_correlogram.py
  runs/
    garch11_baseline/
    garch11_correlogram_design/
      config/
      data/
      generated/
      evaluation/
      diagnostics/
      logs/
```

## 環境構築

```bash
cd /path/to/DSS_code
python3 -m venv .venv
source .venv/bin/activate
pip install -r garch_origin/requirements.txt
```

## 生成

```bash
python3 garch_origin/scripts/generate_garch.py \
  --config garch_origin/config/garch11_baseline.json
```

生成データは以下に保存されます。

```text
garch_origin/runs/garch11_baseline/generated/
```

## Mahalanobis評価

```bash
python3 garch_origin/scripts/evaluate_with_mahalanobis.py \
  --config garch_origin/config/garch11_baseline.json
```

評価結果は以下に保存されます。

```text
garch_origin/runs/garch11_baseline/evaluation/mahalanobis_results/
```

## trainデータのコレログラム

```bash
python3 garch_origin/scripts/plot_train_correlogram.py \
  --config garch_origin/config/garch11_baseline.json
```

出力は以下に保存されます。

```text
garch_origin/runs/garch11_baseline/diagnostics/correlogram/
```

## 最小手順

```bash
cd /path/to/DSS_code
source .venv/bin/activate

python3 garch_origin/scripts/generate_garch.py \
  --config garch_origin/config/garch11_baseline.json

python3 garch_origin/scripts/evaluate_with_mahalanobis.py \
  --config garch_origin/config/garch11_baseline.json
```

## モデル上の注意

- ここでの「学習」は GARCH パラメータの最尤推定です。
- GPU は不要です。
- baseline は単純さを優先し、各資産のボラティリティ過程は別々に推定します。
- 2資産間の同時性は標準化残差の固定相関で表現します。
- 時変相関まで扱う DCC-GARCH は、この baseline より一段複雑な改善候補です。

## コレログラムを反映したGARCH設計

trainデータのコレログラムでは、`sp500` は短中期のボラティリティクラスタリング、`DGS10` はより長期のボラティリティ持続性が見えます。

baseline は残したまま、この結果を反映した別バージョンを `config/garch11_correlogram_design.json` として用意しています。

```bash
python3 garch_origin/scripts/generate_garch.py \
  --config garch_origin/config/garch11_correlogram_design.json

python3 garch_origin/scripts/evaluate_with_mahalanobis.py \
  --config garch_origin/config/garch11_correlogram_design.json
```

この config では、`garch_fit.features` で系列ごとの persistence 制約を分けています。

- `sp500`: `short_mid_volatility_clustering`
- `DGS10`: `long_volatility_clustering`

推定後の `alpha + beta` と half-life は以下に保存されます。

```text
garch_origin/runs/garch11_baseline/data/fitted_params.json
garch_origin/runs/garch11_correlogram_design/data/fitted_params.json
```

また、生成時の標準化残差相関は、trainデータから推定した標準化残差相関に一致するようにしています。各生成ファイルでの確認値は以下に保存されます。

```text
garch_origin/runs/garch11_baseline/data/generated_standardized_residual_correlations.csv
garch_origin/runs/garch11_correlogram_design/data/generated_standardized_residual_correlations.csv
```
