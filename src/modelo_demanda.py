"""Modulo de preparacion y entrenamiento del modelo de demanda."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preparar_conjuntos(df: pd.DataFrame):
    """Prepara variables, conjuntos de entrenamiento y transforma columnas."""
    objetivo = "Units Sold"
    caracteristicas = [columna for columna in df.columns if columna not in {objetivo, "Date"}]

    columnas_categoricas = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Holiday/Promotion",
        "Seasonality",
    ]

    columnas_numericas = [col for col in caracteristicas if col not in columnas_categoricas]

    X = df[caracteristicas]
    y = df[objetivo]

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("numericas", StandardScaler(), columnas_numericas),
        ]
    )

    return (
        X_entrenamiento,
        X_prueba,
        y_entrenamiento,
        y_prueba,
        preprocesador,
        columnas_categoricas,
        columnas_numericas,
        caracteristicas,
    )


def entrenar_modelo(X_train, X_test, y_train, y_test, preprocesador) -> Pipeline:
    """Ajusta XGBoost con preprocesamiento y devuelve el flujo completo."""
    print("\n[Modelo] Ajustando transformaciones...")
    X_train_proc = preprocesador.fit_transform(X_train)
    X_test_proc = preprocesador.transform(X_test)

    modelo_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
        tree_method="hist",
    )

    print("[Modelo] Entrenando XGBoost con detencion temprana...")
    modelo_xgb.fit(
        X_train_proc,
        y_train,
        eval_set=[(X_train_proc, y_train), (X_test_proc, y_test)],
        verbose=False,
    )

    flujo_modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("regresor", modelo_xgb),
        ]
    )

    return flujo_modelo, X_test_proc, modelo_xgb


def evaluar_modelo(modelo, X_test_proc, y_test):
    """Calcula MAE y RMSE del conjunto de prueba."""
    print("\n[Evaluacion] Generando predicciones...")
    y_pred = modelo.predict(X_test_proc)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Error Absoluto Medio (MAE): {mae:.2f} unidades")
    print(f"Raiz del Error Cuadratico Medio (RMSE): {rmse:.2f} unidades")

    return mae, rmse, y_pred
