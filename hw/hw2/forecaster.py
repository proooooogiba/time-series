from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


import numpy as np
import pandas as pd

@dataclass
class RetailSalesGBForecaster:
    """
    Прогнозировщик продаж для одного магазина (агрегированный ряд по store_id)
    на основе моделей градиентного бустинга.

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
    model_params:
        Словарь параметров по моделям (catboost / lightgbm / xgboost).
    random_seed:
        Seed для воспроизводимости.
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

    model_params: Optional[Dict[str, Dict]] = None
    random_seed: int = 42

    # fitted artifacts
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

        if self.model_params is None:
            self.model_params = {
                "catboost": {
                    "iterations": 1500,
                    "learning_rate": 0.03,
                    "depth": 8,
                    "loss_function": "MAE",
                    "random_seed": self.random_seed,
                    "verbose": False,
                },
                "lightgbm": {
                    "n_estimators": 2500,
                    "learning_rate": 0.03,
                    "num_leaves": 63,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": self.random_seed,
                },
                "xgboost": {
                    "n_estimators": 2500,
                    "learning_rate": 0.03,
                    "max_depth": 8,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "random_state": self.random_seed,
                },
            }

    # Data preparation
    def _prepare_full_dataframe(self) -> None:
        sales = self.sales_df.copy()

        if self.store_id is not None and self.store_col is not None and self.store_col in sales.columns:
            sales = sales[sales[self.store_col] == self.store_id].copy()

        sales_agg = (
            sales.groupby(self.date_col, as_index=False)[self.target_col]
            .sum()
            .sort_values(self.date_col)
        )

        # присоединяем календарь, чтобы получить реальные даты + внешние признаки
        calendar = self.calendar_df.copy()
        if "date" in calendar.columns:
            calendar["date"] = pd.to_datetime(calendar["date"])
        else:
            calendar["date"] = pd.to_datetime("2011-01-01") + pd.to_timedelta(calendar[self.date_col] - 1, unit="D")

        df = sales_agg.merge(calendar, on=self.date_col, how="left")

        # восстановим непрерывный daily-индекс, заполним пропуски 0 продажами
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
        df = df.drop(columns=[c for c in calendar.columns if c != "date" and c in df.columns], errors="ignore")
        df = df.merge(calendar, on="date", how="left")

        df[self.target_col] = df[self.target_col].fillna(0.0)

        # агрегируем среднюю цену магазина по неделе и мержим на календарную неделю
        prices = self.prices_df.copy()
        if self.store_id is not None and "store_id" in prices.columns:
            prices = prices[prices["store_id"] == self.store_id].copy()

        if {"wm_yr_wk", "sell_price"}.issubset(prices.columns):
            p = prices.groupby("wm_yr_wk", as_index=False)["sell_price"].mean().rename(columns={"sell_price": "avg_sell_price"})
            df = df.merge(p, on="wm_yr_wk", how="left")
            df["avg_sell_price"] = df["avg_sell_price"].ffill().bfill()
        else:
            df["avg_sell_price"] = np.nan

        # если в календаре есть CASHBACK_STORE_X — возьмём подходящую колонку
        cashback_col = None
        if self.store_id is not None:
            cashback_col = f"CASHBACK_{self.store_id}"
            if cashback_col not in df.columns:
                cashback_col = None
        if cashback_col is None:
            # пробуем store_col формата STORE_2 -> CASHBACK_STORE_2
            if self.store_id is not None:
                cashback_col = f"CASHBACK_{self.store_id}"
        if self.store_id is not None:
            cand = f"CASHBACK_{self.store_id}"
            if cand in df.columns:
                cashback_col = cand
        if cashback_col is None and self.store_id is not None:
            cand2 = f"CASHBACK_{self.store_id.replace('STORE_', 'STORE_')}"
            if cand2 in df.columns:
                cashback_col = cand2
        if cashback_col is None:
            # fallback: если есть ровно одна cashback-колонка, используем её
            cb_cols = [c for c in df.columns if c.startswith("CASHBACK_")]
            if len(cb_cols) == 1:
                cashback_col = cb_cols[0]
        self._cashback_col = cashback_col

        # базовые календарные признаки (часть уже в календаре)
        df["dayofweek"] = df["date"].dt.dayofweek.astype(int)
        df["day"] = df["date"].dt.day.astype(int)
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month.astype(int)
        df["year"] = df["date"].dt.year.astype(int)

        # категориальные признаки (для CatBoost)
        cat_cols = []
        for c in ["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            if c in df.columns:
                df[c] = df[c].fillna("None").astype(str)
                cat_cols.append(c)
        self.cat_feature_cols_ = cat_cols

        # store_id как категория (константа) — полезно, если делать общий модельный класс
        if self.store_id is not None:
            df["store_id"] = self.store_id
            self.cat_feature_cols_.append("store_id")

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

    # Feature engineering
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

    def _make_train_matrix(self) -> Tuple[pd.DataFrame, np.ndarray]:
        df = self.df_full_.copy()
        # обозначим train часть: по индексу даты
        df = df.set_index("date")
        df_train = df.loc[self.train_.index].reset_index()

        df_feat = self._add_lag_features(df_train)

        # признаки cashback
        if getattr(self, "_cashback_col", None) and self._cashback_col in df_feat.columns:
            pass  # уже есть
        df_feat = df_feat.dropna().reset_index(drop=True)

        self.feature_cols_ = self._get_feature_columns(df_feat)

        X = df_feat[self.feature_cols_]
        y = df_feat[self.target_col].values.astype(float)
        return X, y

    # Models
    def _fit_model(self, model_name: str = "catboost") -> object:
        model_name = model_name.lower()
        X, y = self._make_train_matrix()

        if model_name == "catboost":
            model = CatBoostRegressor(**self.model_params.get("catboost", {}))
            cat_features = [c for c in self.cat_feature_cols_ if c in X.columns]
            model.fit(X, y, cat_features=cat_features)
            self.fitted_models_[model_name] = model
            return model

        if model_name == "lightgbm":
            X_lgb = X.copy()
            for c in self.cat_feature_cols_:
                if c in X_lgb.columns:
                    X_lgb[c] = X_lgb[c].astype("category")

            model = lgb.LGBMRegressor(**self.model_params.get("lightgbm", {}))
            model.fit(X_lgb, y)
            self.fitted_models_[model_name] = model
            return model

        if model_name == "xgboost":
            X_xgb = pd.get_dummies(X, columns=[c for c in self.cat_feature_cols_ if c in X.columns], dummy_na=False)
            model = XGBRegressor(**self.model_params.get("xgboost", {}))
            model.fit(X_xgb, y)
            self.fitted_models_[model_name] = (model, X_xgb.columns.tolist())
            return self.fitted_models_[model_name]

        raise ValueError(f"Неизвестная модель: {model_name}. Доступно: catboost, lightgbm, xgboost.")

    def _predict_one(self, model_name: str, model_obj: object, row: pd.DataFrame) -> float:
        model_name = model_name.lower()
        if model_name == "catboost":
            return float(model_obj.predict(row)[0])
        if model_name == "lightgbm":
            tmp = row.copy()
            for c in self.cat_feature_cols_:
                if c in tmp.columns:
                    tmp[c] = tmp[c].astype("category")
            return float(model_obj.predict(tmp)[0])
        if model_name == "xgboost":
            model, train_cols = model_obj  # type: ignore[misc]
            tmp = pd.get_dummies(row, columns=[c for c in self.cat_feature_cols_ if c in row.columns], dummy_na=False)
            # выровняем колонки
            for c in train_cols:
                if c not in tmp.columns:
                    tmp[c] = 0
            tmp = tmp[train_cols]
            return float(model.predict(tmp)[0])
        raise ValueError(model_name)

    # Forecasting
    def forecast(
        self,
        model_name: str = "catboost",
        horizon: int = 7,
        refit: bool = True,
    ) -> pd.Series:
        """
        Рекурсивный прогноз на horizon дней вперёд, стартуя от конца train_.

        Возвращает Series с датами (datetime) и прогнозом.
        """
        if horizon <= 0:
            raise ValueError("horizon должен быть > 0")

        if refit or model_name.lower() not in self.fitted_models_:
            model_obj = self._fit_model(model_name)
        else:
            model_obj = self.fitted_models_[model_name.lower()]

        df = self.df_full_.set_index("date")

        # Берём будущие строки (обычно это начало теста)
        future_idx = self.test_.index[:horizon]
        if len(future_idx) < horizon:
            raise ValueError("Недостаточно данных в test_ для указанного horizon.")

        history = self.train_.copy().astype(float).tolist()  # список значений y до точки прогноза
        history_index = list(self.train_.index)

        preds = []
        pred_index = []

        for t in future_idx:
            row = df.loc[[t]].reset_index()
            # Соберём признаки лагов из history
            feat = row.copy()
            # временно создаём целевую колонку, чтобы использовать общий генератор
            # (значение не используется напрямую, но нужно для имён)
            feat[self.target_col] = np.nan

            # добавляем лаги/скользящие на основе history
            for lag in self.lags:
                feat[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
            for w in self.rolling_windows:
                vals = history[-w:] if len(history) >= w else history[:]
                if len(vals) == 0:
                    feat[f"roll_mean_{w}"] = np.nan
                    feat[f"roll_std_{w}"] = np.nan
                else:
                    feat[f"roll_mean_{w}"] = float(np.mean(vals))
                    feat[f"roll_std_{w}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            feat["expanding_mean"] = float(np.mean(history)) if len(history) else np.nan

            # оставляем только нужные колонки
            X_row = feat[self.feature_cols_].copy()
            X_row = X_row.fillna(0)

            yhat = self._predict_one(model_name, model_obj, X_row)
            preds.append(yhat)
            pred_index.append(t)

            history.append(yhat)
            history_index.append(t)

        return pd.Series(preds, index=pd.DatetimeIndex(pred_index), name=f"forecast_{model_name}")

    def evaluate(
        self,
        horizons: Sequence[int] = (7, 30, 90),
        models: Sequence[str] = ("catboost", "lightgbm"),
    ) -> pd.DataFrame:
        """
        Оценка MAE / RMSE / MAPE на первых h днях теста для нескольких горизонтов.
        """
        rows = []
        for model_name in models:
            # обучаем один раз на train и переиспользуем внутри разных горизонтов
            self._fit_model(model_name)
            for h in horizons:
                fc = self.forecast(model_name=model_name, horizon=int(h), refit=False)
                y_true = self.test_.iloc[: int(h)].values.astype(float)
                y_pred = fc.values.astype(float)

                rows.append(
                    {
                        "model": model_name,
                        "horizon": int(h),
                        "MAE": mean_absolute_error(y_true, y_pred),
                        "RMSE": mean_squared_error(y_true, y_pred),
                        "MAPE":  mean_absolute_percentage_error(y_true, y_pred) * 100,
                    }
                )
        return pd.DataFrame(rows).sort_values(["model", "horizon"]).reset_index(drop=True)

    def get_model(self, model_name: str = "catboost") -> object:
        model_obj = self.fitted_models_[model_name]
        return model_obj

    def feature_importance(self, model_name: str = "catboost", top_n: int = 25) -> pd.DataFrame:
        """
        Возвращает DataFrame с важностью признаков (если модель поддерживает).
        """
        model_name = model_name.lower()
        if model_name not in self.fitted_models_:
            self._fit_model(model_name)
        model_obj = self.fitted_models_[model_name]

        if model_name == "catboost":
            importances = model_obj.get_feature_importance()
            return (
                pd.DataFrame({"feature": self.feature_cols_, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

        if model_name == "lightgbm":
            importances = model_obj.feature_importances_
            return (
                pd.DataFrame({"feature": self.feature_cols_, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

        if model_name == "xgboost":
            model, train_cols = model_obj  # type: ignore[misc]
            # для xgboost важности по one-hot колонкам
            imp = model.feature_importances_
            return (
                pd.DataFrame({"feature": train_cols, "importance": imp})
                .sort_values("importance", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

        raise ValueError(model_name)
