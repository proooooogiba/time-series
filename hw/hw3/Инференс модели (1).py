from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


@dataclass
class RetailSalesGBForecaster:
    """
    Прогнозировщик продаж для одного магазина (агрегированный ряд по store_id)
    на основе заранее обученных моделей градиентного бустинга.

    Важно
    -----
    * Код обучения удалён: класс работает только в режиме инференса.
    * Веса моделей загружаются из файлов:
        - models/catboost_model.cbm
        - models/lightgbm_model.txt

    Параметры
    ---------
    sales_df:
        DataFrame с историей продаж (item_id, store_id, date_id, cnt).
    calendar_df:
        DataFrame с календарём (shop_sales_dates.csv).
    prices_df:
        DataFrame с ценами (shop_sales_prices.csv).
    date_col:
        Колонка даты в sales_df (по умолчанию "date_id").
    target_col:
        Колонка целевой переменной.
    store_col:
        Колонка магазина (по умолчанию "store_id").
    store_id:
        Идентификатор магазина, для которого строим прогноз.
    freq:
        Частота ряда ("D" для дневных данных).
    test_size:
        Размер тестовой выборки (последние N дней). Должен быть >= max(horizons) для evaluate.
    lags:
        Список лагов (в днях).
    rolling_windows:
        Окна для скользящих статистик (в днях).
    catboost_path:
        Путь к файлу CatBoost модели (.cbm).
    lightgbm_path:
        Путь к файлу LightGBM модели (.txt).
    """

    sales_df: pd.DataFrame
    calendar_df: pd.DataFrame
    prices_df: pd.DataFrame

    date_col: str = "date_id"
    target_col: str = "cnt"
    store_col: Optional[str] = "store_id"
    store_id: Optional[str] = None

    freq: str = "D"
    test_size: int = 90

    lags: Sequence[int] = (1, 7, 14, 28)
    rolling_windows: Sequence[int] = (7, 28)

    # paths to pre-trained models
    catboost_path: str = "models/catboost_model.cbm"
    lightgbm_path: str = "models/lightgbm_model.txt"

    # prepared artifacts
    df_full_: pd.DataFrame = field(init=False, repr=False)
    series_original_: pd.Series = field(init=False, repr=False)
    series_: pd.Series = field(init=False, repr=False)
    train_: pd.Series = field(init=False, repr=False)
    test_: pd.Series = field(init=False, repr=False)

    fitted_models_: Dict[str, object] = field(default_factory=dict, init=False, repr=False)
    feature_cols_: List[str] = field(default_factory=list, init=False, repr=False)
    cat_feature_cols_: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._prepare_full_dataframe()
        self._split_train_test()

        # load pre-trained models
        self._init_fitted_models()

        # infer feature list (prefer LightGBM model feature names if available)
        self._init_feature_columns()

    # ------------------------
    # Data preparation
    # ------------------------
    def _prepare_full_dataframe(self) -> None:
        sales = self.sales_df.copy()

        if self.store_id is not None and self.store_col is not None and self.store_col in sales.columns:
            sales = sales[sales[self.store_col] == self.store_id].copy()

        sales_agg = (
            sales.groupby(self.date_col, as_index=False)[self.target_col]
            .sum()
            .sort_values(self.date_col)
        )

        # calendar -> real datetime + external features
        calendar = self.calendar_df.copy()
        if "date" in calendar.columns:
            calendar["date"] = pd.to_datetime(calendar["date"])
        else:
            calendar["date"] = pd.to_datetime("2011-01-01") + pd.to_timedelta(
                calendar[self.date_col] - 1, unit="D"
            )

        df = sales_agg.merge(calendar, on=self.date_col, how="left")

        # make continuous daily index and fill gaps with 0 sales
        df = df.sort_values("date").reset_index(drop=True)
        all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq=self.freq)
        df = (
            df.set_index("date")
            .reindex(all_dates)
            .rename_axis("date")
            .reset_index()
        )
        if self.date_col not in df.columns:
            df[self.date_col] = np.nan

        # remove potentially duplicated calendar cols then merge by actual date
        df = df.drop(columns=[c for c in calendar.columns if c != "date" and c in df.columns], errors="ignore")
        df = df.merge(calendar, on="date", how="left")

        df[self.target_col] = df[self.target_col].fillna(0.0)

        # prices: average by week
        prices = self.prices_df.copy()
        if self.store_id is not None and "store_id" in prices.columns:
            prices = prices[prices["store_id"] == self.store_id].copy()

        if {"wm_yr_wk", "sell_price"}.issubset(prices.columns):
            p = (
                prices.groupby("wm_yr_wk", as_index=False)["sell_price"]
                .mean()
                .rename(columns={"sell_price": "avg_sell_price"})
            )
            df = df.merge(p, on="wm_yr_wk", how="left")
            df["avg_sell_price"] = df["avg_sell_price"].ffill().bfill()
        else:
            df["avg_sell_price"] = np.nan

        # cashback column (optional)
        cashback_col = None
        if self.store_id is not None:
            cand = f"CASHBACK_{self.store_id}"
            if cand in df.columns:
                cashback_col = cand
        if cashback_col is None:
            cb_cols = [c for c in df.columns if c.startswith("CASHBACK_")]
            if len(cb_cols) == 1:
                cashback_col = cb_cols[0]
        self._cashback_col = cashback_col

        # basic date features
        df["dayofweek"] = df["date"].dt.dayofweek.astype(int)
        df["day"] = df["date"].dt.day.astype(int)
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month.astype(int)
        df["year"] = df["date"].dt.year.astype(int)

        # categorical features (consistent with training-time preprocessing)
        cat_cols: List[str] = []
        for c in ["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            if c in df.columns:
                df[c] = df[c].fillna("None").astype(str)
                cat_cols.append(c)

        if self.store_id is not None:
            df["store_id"] = self.store_id
            cat_cols.append("store_id")

        self.cat_feature_cols_ = cat_cols

        self.df_full_ = df
        self.series_original_ = df.set_index("date")[self.target_col].astype(float)
        self.series_ = self.series_original_.copy()

    def _split_train_test(self) -> None:
        s = self.series_.copy()
        if len(s) <= self.test_size + max(self.lags):
            raise ValueError(
                f"Слишком короткий ряд: длина={len(s)}. Нужно хотя бы test_size+max(lags) наблюдений."
            )
        self.train_ = s.iloc[:-self.test_size]
        self.test_ = s.iloc[-self.test_size:]

    # ------------------------
    # Feature engineering
    # ------------------------
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for lag in self.lags:
            out[f"lag_{lag}"] = out[self.target_col].shift(lag)
        for w in self.rolling_windows:
            out[f"roll_mean_{w}"] = out[self.target_col].shift(1).rolling(w).mean()
            out[f"roll_std_{w}"] = out[self.target_col].shift(1).rolling(w).std()
        out["expanding_mean"] = out[self.target_col].shift(1).expanding().mean()
        return out

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        drop_cols = {self.target_col}
        if "date" in df.columns:
            drop_cols.add("date")
        return [c for c in df.columns if c not in drop_cols]

    # ------------------------
    # Models (inference only)
    # ------------------------
    def _init_fitted_models(self) -> None:
        """
        Загрузка заранее обученных моделей из файлов.

        Ожидаемые пути (по умолчанию):
        - models/catboost_model.cbm
        - models/lightgbm_model.txt
        """
        cb = CatBoostRegressor()
        cb.load_model(self.catboost_path)

        lgbm = lgb.Booster(model_file=self.lightgbm_path)

        self.fitted_models_ = {
            "catboost": cb,
            "lightgbm": lgbm,
        }

    def _init_feature_columns(self) -> None:
        """
        Определяем список признаков, который ожидают модели.

        * Если доступен LightGBM Booster, предпочитаем его feature_name().
        * Иначе — восстанавливаем список из feature engineering на train_ части.
        """
        # 1) try to read feature names from LightGBM model
        model_feats: Optional[List[str]] = None
        lgbm = self.fitted_models_.get("lightgbm")
        if lgbm is not None:
            try:
                model_feats = list(lgbm.feature_name())  # type: ignore[attr-defined]
            except Exception:
                model_feats = None

        # 2) fallback: reconstruct from engineered train features
        if not model_feats:
            df = self.df_full_.set_index("date")
            df_train = df.loc[self.train_.index].reset_index()
            df_feat = self._add_lag_features(df_train).dropna().reset_index(drop=True)
            model_feats = self._get_feature_columns(df_feat)

        self.feature_cols_ = model_feats

    def _prepare_row_for_model(self, model_name: str, row: pd.DataFrame) -> pd.DataFrame:
        """
        Выравниваем колонки и типы под конкретную модель.
        """
        if row.shape[0] != 1:
            raise ValueError("row должен содержать ровно 1 строку")

        # Ensure all expected columns exist, order is correct
        X = row.reindex(columns=self.feature_cols_, fill_value=0)

        model_name = model_name.lower()

        # Cast categoricals
        if self.cat_feature_cols_:
            for c in self.cat_feature_cols_:
                if c in X.columns:
                    if model_name == "lightgbm":
                        # LightGBM works well with pandas 'category'
                        X[c] = X[c].astype(str).astype("category")
                    else:
                        # CatBoost: keep as string/object
                        X[c] = X[c].astype(str)

        # Non-categorical -> numeric
        for c in X.columns:
            if c not in self.cat_feature_cols_:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

        return X

    def _predict_one(self, model_name: str, row: pd.DataFrame) -> float:
        model_name = model_name.lower()
        if model_name not in self.fitted_models_:
            raise ValueError(f"Модель '{model_name}' не загружена. Доступно: {list(self.fitted_models_.keys())}")

        X = self._prepare_row_for_model(model_name, row)

        if model_name == "catboost":
            cb: CatBoostRegressor = self.fitted_models_["catboost"]  # type: ignore[assignment]
            return float(cb.predict(X)[0])

        if model_name == "lightgbm":
            booster: lgb.Booster = self.fitted_models_["lightgbm"]  # type: ignore[assignment]
            return float(booster.predict(X)[0])

        raise ValueError(model_name)

    # ------------------------
    # Forecasting
    # ------------------------
    def forecast(self, model_name: str = "catboost", horizon: int = 7, output_csv: Optional[str | Path] = None) -> pd.Series:
        """
        Рекурсивный прогноз на horizon дней вперёд, стартуя от конца train_.

        Возвращает Series с датами (datetime) и прогнозом.
        """
        if horizon <= 0:
            raise ValueError("horizon должен быть > 0")

        df = self.df_full_.set_index("date")

        # Обычно прогнозируем первые дни теста
        future_idx = self.test_.index[:horizon]
        if len(future_idx) < horizon:
            raise ValueError("Недостаточно данных в test_ для указанного horizon.")

        history = self.train_.copy().astype(float).tolist()

        preds: List[float] = []
        pred_index: List[pd.Timestamp] = []

        for t in future_idx:
            row = df.loc[[t]].reset_index()

            feat = row.copy()
            feat[self.target_col] = np.nan  # placeholder for feature generation naming

            # lag features from history
            for lag in self.lags:
                feat[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan

            # rolling stats from history
            for w in self.rolling_windows:
                vals = history[-w:] if len(history) >= w else history[:]
                if len(vals) == 0:
                    feat[f"roll_mean_{w}"] = np.nan
                    feat[f"roll_std_{w}"] = np.nan
                else:
                    feat[f"roll_mean_{w}"] = float(np.mean(vals))
                    feat[f"roll_std_{w}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            feat["expanding_mean"] = float(np.mean(history)) if len(history) else np.nan

            # prepare final row: select only expected features + fill NaNs
            X_row = feat.reindex(columns=self.feature_cols_, fill_value=0).fillna(0)

            yhat = self._predict_one(model_name, X_row)
            preds.append(yhat)
            pred_index.append(t)

            history.append(yhat)

        series = pd.Series(preds, index=pd.DatetimeIndex(pred_index), name=f"forecast_{model_name}")

        if output_csv is not None:
            self.save_forecast_csv(series, output_csv)

        return series

    def save_forecast_csv(
        self,
        forecast: pd.Series,
        path: str | Path,
        date_col: str = "date",
        value_col: str = "yhat",
    ) -> Path:
        """Сохраняет прогноз (Series с DatetimeIndex) в CSV.

        Формат: две колонки — date_col и value_col.
        Возвращает Path до сохранённого файла.
        """
        if not isinstance(forecast, pd.Series):
            raise TypeError("forecast должен быть pd.Series")
        if not isinstance(forecast.index, pd.DatetimeIndex):
            # попробуем привести
            forecast = forecast.copy()
            forecast.index = pd.to_datetime(forecast.index)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df_out = forecast.rename(value_col).to_frame()
        df_out.insert(0, date_col, df_out.index)
        df_out = df_out.reset_index(drop=True)
        df_out.to_csv(out_path, index=False)
        return out_path


    def evaluate(
        self,
        horizons: Sequence[int] = (7, 30, 90),
        models: Sequence[str] = ("catboost", "lightgbm"),
    ) -> pd.DataFrame:
        """
        Оценка MAE / RMSE / MAPE на первых h днях теста для нескольких горизонтов.
        (Инференс по заранее обученным моделям, без дообучения.)
        """
        rows = []
        for model_name in models:
            for h in horizons:
                h = int(h)
                fc = self.forecast(model_name=model_name, horizon=h)
                y_true = self.test_.iloc[:h].values.astype(float)
                y_pred = fc.values.astype(float)

                rows.append(
                    {
                        "model": model_name,
                        "horizon": h,
                        "MAE": mean_absolute_error(y_true, y_pred),
                        "RMSE": mean_squared_error(y_true, y_pred),
                        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
                    }
                )

        return (
            pd.DataFrame(rows)
            .sort_values(["model", "horizon"])
            .reset_index(drop=True)
        )

    def get_model(self, model_name: str = "catboost") -> object:
        model_name = model_name.lower()
        if model_name not in self.fitted_models_:
            raise ValueError(f"Модель '{model_name}' не загружена.")
        return self.fitted_models_[model_name]
