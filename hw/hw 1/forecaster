"""
Модуль с классом для классического прогнозирования временных рядов
продаж розничного ритейлера.

Класс:
    - предобрабатывает исходные данные в удобный формат,
    - обучает несколько классических моделей (ARIMA, ETS, SARIMA),
    - строит прогноз на неделю, месяц и квартал,
    - оценивает качество по MAE / RMSE / MAPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsforecast.models import AutoETS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class RetailSalesForecaster:
    """
    Классический прогнозировщик для одного ряда store_id.

    Параметры
    ---------
    sales_df:
        DataFrame с историей продаж.
    date_col:
        Название колонки с датой (например, "date_id").
    target_col:
        Название колонки с целевой переменной (например, "cnt").
    store_col:
        Название колонки идентификатора магазина.
    store_id:
        Значение идентификатора магазина (STORE_1 / STORE_2 / ...).
    freq:
        Частота временного ряда (для ежедневных данных — "D").
    test_size:
        Размер тестовой выборки (последние N наблюдений).
        Должен быть ≥ максимального горизонта прогноза.
    seasonal_period:
        Основной сезонный период (для дневных розничных данных часто 7).
    """

    sales_df: pd.DataFrame
    date_col: str = "date_id"
    target_col: str = "cnt"
    store_col: Optional[str] = "store_id"
    store_id: Optional[str] = None
    freq: str = "D"
    test_size: int = 90
    seasonal_period: int = 7

    series_: pd.Series = field(init=False, repr=False)
    series_original_: pd.Series = field(init=False, repr=False)
    train_: pd.Series = field(init=False, repr=False)
    test_: pd.Series = field(init=False, repr=False)
    models_: Dict[str, object] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._prepare_series()
        self._train_test_split()

    # Предобработка данных
    def _prepare_series(self) -> None:
        """
        Преобразование сырого датафрейма продаж в один временной ряд.

        Шаги:
        * фильтрация по магазину,
        * группировка по дате и агрегация,
        * сортировка и приведение к равномерной частоте
        """
        df = self.sales_df.copy()

        if self.store_col is not None and self.store_id is not None:
            df = df[df[self.store_col] == self.store_id]

        if df.empty:
            raise ValueError("После фильтрации по store_id данные оказались пустыми.")

        df[self.date_col] = pd.to_datetime("2011-01-01") + pd.to_timedelta(df[self.date_col] - 1, unit="D")

        df = (
            df[[self.date_col, self.target_col]]
            .groupby(self.date_col)[self.target_col]
            .sum()
            .sort_index()
        )

        series_original = df.asfreq(self.freq).fillna(0)
        series = series_original.copy()
        self.series_original_ = series_original
        self.series_ = series

    def _train_test_split(self) -> None:
        """
        Hold-out разбиение: последние `test_size` наблюдений — тест.
        """
        if self.test_size <= 0:
            raise ValueError("test_size должен быть положительным числом.")

        if self.test_size >= len(self.series_):
            raise ValueError(
                "test_size больше или равен длине ряда. "
                "Сделайте test_size меньше."
            )

        # series = [train, test_size]
        self.train_ = self.series_.iloc[:-self.test_size]
        self.test_ = self.series_.iloc[-self.test_size:]

    # Обучение моделей
    def fit_all_models(self) -> None:
        """
        Обучение всех реализованных классических моделей
        на обучающей части ряда.
        """
        self.models_["arima"] = self._fit_arima()
        self.models_["ets"] = self._fit_ets()
        self.models_["sarima"] = self._fit_sarima()

    def _fit_arima(self) -> ARIMA:
        """
        Обучение ARIMA с подбором набора (p, d, q) по AIC.
        AIC - мера качества модели, которая выбирает оптимальный порядок параметров для модели.

        Используется statsmodels.tsa.arima.model.ARIMA.
        """
        best_aic = np.inf
        best_order: Optional[Tuple[int, int, int]] = None
        best_model = None

        arima_param_grid = [
            (p, d, q)
            for p in range(0, 3)
            for d in (0, 1)
            for q in range(0, 3)
            if not (p == 0 and d == 0 and q == 0)
        ]

        for (p, d, q) in arima_param_grid:
            try:
                model = ARIMA(self.train_, order=(p, d, q))
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
                    best_model = res
            except Exception:
                # На некоторых параметрах модель может не обучиться
                continue

        if best_model is None or best_order is None:
            raise RuntimeError("Не удалось подобрать ни одной ARIMA-модели.")

        self.models_["arima_order"] = best_order
        return best_model

    def _fit_ets(self) -> ExponentialSmoothing:
        """
        Обучение ETS с аддитивным трендом и сезонностью.
        """
        AutoETS
        model = ExponentialSmoothing(
            self.train_,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_period,
        )
        res = model.fit(optimized=True)
        return res

    def _fit_sarima(self) -> SARIMAX:
        """
        Обучение сезонной ARIMA (SARIMA).

        Сезонный период задаётся параметром `seasonal_period`.
        Для упрощения перебирается небольшая сетка параметров.
        """
        best_aic = np.inf
        best_params: Optional[Tuple[int, int, int, int, int, int, int]] = None
        best_model = None

        sarima_param_grid = [
            (p, d, q)
            for p in range(0, 2)
            for d in (0, 1)
            for q in range(0, 2)
            for P in range(0, 2)
            for D in (0, 1)
            for Q in range(0, 2)
        ]

        for (p, d, q, P, D, Q) in sarima_param_grid:
            order = (p, d, q)
            seasonal_order = (P,D,Q,self.seasonal_period)
            try:
                model = SARIMAX(
                    self.train_,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_params = (
                        p,
                        d,
                        q,
                        P,
                        D,
                        Q,
                        self.seasonal_period,
                    )
                    best_model = res
            except Exception:
                continue

        if best_model is None or best_params is None:
            raise RuntimeError("Не удалось подобрать ни одной SARIMA-модели.")

        self.models_["sarima_params"] = best_params
        return best_model

    # Прогноз и метрики
    def _make_future_index(self, steps: int) -> pd.DatetimeIndex:
        """
        Создаёт индекс для будущего горизонта прогноза.
        """
        last_date = self.train_.index[-1]
        freq = pd.infer_freq(self.series_.index) or self.freq
        return pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=steps,
            freq=freq,
        )

    def forecast(self, model_name: str, steps: int) -> pd.Series:
        """
        Точечный прогноз выбранной моделью на заданный горизонт.

        Параметры
        ---------
        model_name:
            Название модели ("arima", "ets", "sarima").
        steps:
            Горизонт в шагах (днях).

        Возвращает
        ----------
        pd.Series с прогнозом в исходной шкале (без логарифма).
        """
        if model_name not in self.models_:
            raise KeyError(
                f"Модель '{model_name}' не обучена. "
                "Сначала вызовите `fit_all_models`."
            )

        model = self.models_[model_name]

        try:
            pred = model.forecast(steps=steps)
        except Exception:
            pred = model.get_forecast(steps=steps).predicted_mean

        pred = pd.Series(pred, index=self._make_future_index(steps))

        return pred

    # Метрики
    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_safe = np.where(y_true == 0, 1e-8, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)

    def evaluate(
        self,
        horizons: Sequence[int] = (7, 30, 90),
    ) -> pd.DataFrame:
        """
        Оценка обученных моделей на тестовом периоде.

        Строится один прогноз до максимального горизонта,
        затем метрики считаются отдельно на:
            - 1 неделю
            - 1 месяц
            - 1 квартал

        Параметры
        ---------
        horizons:
            Набор горизонтов в шагах.

        Возвращает
        ----------
        pd.DataFrame с колонками:
        ["model", "horizon", "mae", "rmse", "mape"].
        """
        if not self.models_:
            raise RuntimeError(
                "Сначала обучите модели методом `fit_all_models`."
            )

        horizons = sorted(set(int(h) for h in horizons))
        max_h = horizons[-1]

        if max_h > len(self.test_):
            raise ValueError(
                "Максимальный горизонт больше длины тестового набора. "
                "Увеличьте test_size или уменьшите горизонты."
            )

        results: List[Dict[str, object]] = []

        full_test_original = self.series_original_.iloc[-self.test_size :]

        for model_name in ("arima", "ets", "sarima"):
            if model_name not in self.models_:
                continue

            forecast_series = self.forecast(model_name, steps=max_h)

            for h in horizons:
                y_true = full_test_original.iloc[:h].to_numpy()
                y_pred = forecast_series.iloc[:h].to_numpy()

                mae = self._mae(y_true, y_pred)
                rmse = self._rmse(y_true, y_pred)
                mape = self._mape(y_true, y_pred)

                results.append(
                    {
                        "model": model_name,
                        "horizon": h,
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                    }
                )

        return pd.DataFrame(results)