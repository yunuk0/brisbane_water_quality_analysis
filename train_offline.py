import pandas as pd
import numpy as np
from pathlib import Path
import random

from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

import optuna
from optuna.logging import set_verbosity, ERROR as OPTUNA_ERROR

# Optuna 로그 최소화
set_verbosity(OPTUNA_ERROR)

# =====================================================================
# 1. 설정값
# =====================================================================
DATA_PATH = Path(__file__).parent / "data" / "df_final.csv"
OUT_PATH  = Path(__file__).parent / "data" / "future_week_forecast.csv"

TARGET_COL  = "Chlorophyll_Kalman"   # 모델 타깃
RAW_COL     = "Chlorophyll"          # 원본 클로로필 컬럼
TEST_DAYS   = 30                     # 최근 30일을 테스트로 사용
N_TRIALS    = 30                     # Optuna 탐색 횟수 (너무 길면 20~30 정도)
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)

EXOG_COLS = [
    "Dissolved Oxygen_Kalman", "Salinity_Kalman", "Temperature_Kalman",
    "Turbidity_Kalman", "pH_Kalman", "W_Relative Humidity",
    "W_Shortwave Radiation", "W_Temperature"
]


def mean_abs_percentage_error(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def make_features_with_diff(
    df: pd.DataFrame,
    target_col: str,
    exog_cols=None,
    lag_list=[2],
    roll_windows=[6, 72, 144],
    dropna=True
):
    if exog_cols is None:
        exog_cols = []

    data = df.copy()
    diff_col = f"{target_col}_diff"
    data[diff_col] = data[target_col].diff()

    feats = pd.DataFrame(index=data.index)

    # 타깃 Lag
    for lag in lag_list:
        feats[f"{target_col}_lag{lag}"] = data[target_col].shift(lag)

    # 타깃 Rolling
    for win in roll_windows:
        feats[f"{target_col}_roll_mean_{win}"] = (
            data[target_col].shift(1).rolling(win).mean()
        )
        feats[f"{target_col}_roll_std_{win}"] = (
            data[target_col].shift(1).rolling(win).std()
        )

    # Diff lag
    for lag in [1, 2]:
        feats[f"{diff_col}_lag{lag}"] = data[diff_col].shift(lag)

    # Diff rolling
    for win in [6, 72]:
        feats[f"{diff_col}_roll_mean_{win}"] = (
            data[diff_col].shift(1).rolling(win).mean()
        )
        feats[f"{diff_col}_roll_std_{win}"] = (
            data[diff_col].shift(1).rolling(win).std()
        )

    # 외생변수 Lag + Rolling
    exog_lags = [6, 72, 144]          # 1시간, 12시간, 1일
    exog_roll_windows = [72, 144]     # 12시간, 1일

    for col in exog_cols:
        if col not in data.columns:
            continue

        for lag in exog_lags:
            feats[f"{col}_lag{lag}"] = data[col].shift(lag)

        for win in exog_roll_windows:
            feats[f"{col}_roll_mean_{win}"] = (
                data[col].shift(1).rolling(win).mean()
            )

    # 시간 피처
    feats["hour"]      = data.index.hour
    feats["dayofweek"] = data.index.dayofweek

    if dropna:
        valid_idx = feats.dropna().index
        X = feats.loc[valid_idx]
        y = data.loc[valid_idx, target_col]
        return X, y
    else:
        return feats, data[target_col]


def recursive_forecast(df, model, target_col, n_steps, freq_td, feature_means, exog_cols):
    data = df.copy()
    preds = []
    idxs = []

    for _ in range(n_steps):
        last_idx = data.index[-1]
        next_idx = last_idx + freq_td

        base_row = data.iloc[-1].copy()
        base_row[target_col] = np.nan
        data.loc[next_idx] = base_row

        X_tmp, _ = make_features_with_diff(
            data,
            target_col,
            exog_cols=exog_cols,
            lag_list=[2],
            dropna=False,
        )

        x_next = X_tmp.loc[[next_idx]].fillna(feature_means)
        y_next = model.predict(x_next)[0]

        data.loc[next_idx, target_col] = y_next
        preds.append(y_next)
        idxs.append(next_idx)

    return pd.Series(preds, index=idxs)


def main():
    print("데이터 로드:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").set_index("Timestamp")

    freq_td = df.index.to_series().diff().dropna().mode()[0]
    steps_week = int(pd.Timedelta("7D") / freq_td)
    print("추정 간격:", freq_td, " / 1주일 스텝 수:", steps_week)

    X_all, y_all = make_features_with_diff(
        df,
        TARGET_COL,
        exog_cols=EXOG_COLS
    )
    print("전체 피처 크기:", X_all.shape)

    cutoff_time = X_all.index.max() - pd.Timedelta(days=TEST_DAYS)
    X_train = X_all[X_all.index <= cutoff_time]
    y_train = y_all.loc[X_train.index]

    X_test  = X_all[X_all.index > cutoff_time]
    y_test  = y_all.loc[X_test.index]

    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Optuna 목적함수
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "random_state": SEED,
            "verbose": -1,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 200),
            "max_depth":        trial.suggest_int("max_depth", -1, 20),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 200),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1e1),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
            "n_estimators":     1000,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        maes = []

        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            maes.append(mae)

        return np.mean(maes)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest Params:", study.best_params)
    print("Best CV MAE:", study.best_value)

    best_params = study.best_params
    best_params.update({
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "random_state": SEED,
        "verbose": -1,
        "n_estimators": 1000,
    })

    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    mae_test  = mean_absolute_error(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_test = mean_abs_percentage_error(y_test.values, y_pred)

    bt_pair = df.loc[df.index > cutoff_time, [RAW_COL, TARGET_COL]].dropna()
    mape_raw_vs_kalman = mean_abs_percentage_error(
        bt_pair[RAW_COL].values,
        bt_pair[TARGET_COL].values
    )

    print("\n=== Test(백테스트) 성능 ===")
    print(f"[모델 vs Kalman 타깃] MAE  : {mae_test:.4f}")
    print(f"[모델 vs Kalman 타깃] RMSE : {rmse_test:.4f}")
    print(f"[모델 vs Kalman 타깃] MAPE : {mape_test:.2f}%")
    print(f"[원본 vs Kalman     ] MAPE : {mape_raw_vs_kalman:.2f}%")

    feature_means = X_train.mean()
    future_week = recursive_forecast(
        df=df,
        model=final_model,
        target_col=TARGET_COL,
        n_steps=steps_week,
        freq_td=freq_td,
        feature_means=feature_means,
        exog_cols=EXOG_COLS,
    )

    future_week.index.name = "Timestamp"
    future_week.to_frame(name="Forecast_Chlorophyll_Kalman").to_csv(
        OUT_PATH,
        index=True,
        encoding="utf-8-sig"
    )

    print(f'\n일주일 미래 예측값을 "{OUT_PATH}" 파일로 저장했습니다.')


if __name__ == "__main__":
    main()
