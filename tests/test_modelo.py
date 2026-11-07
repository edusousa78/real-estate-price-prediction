
# ============================================================
# PROJETO 04 — REAL ESTATE PRICE PREDICTION
# AULA 05 — ETAPA 03
# TESTES AUTOMATIZADOS DO MODELO
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parents[1]

CAMINHO_MODELO = (
    BASE_DIR / "models" / "modelo_hgb.joblib"
)

CAMINHO_METADATA = (
    BASE_DIR / "models" / "metadata_modelo.json"
)

COLUNAS_ESPERADAS = [
    "Área Construída (m2)",
    "Fração Ideal",
    "Padrão (IPTU)",
    "cep_3",
]


@pytest.fixture(scope="module")
def artefatos():
    assert CAMINHO_MODELO.is_file()
    assert CAMINHO_METADATA.is_file()

    # Carregar somente o artefato local confiável.
    modelo = joblib.load(CAMINHO_MODELO)

    with CAMINHO_METADATA.open(
        encoding="utf-8"
    ) as arquivo:
        metadata = json.load(arquivo)

    return modelo, metadata


@pytest.fixture(scope="module")
def entrada_exemplo(artefatos):
    modelo, _ = artefatos

    preprocessador = modelo.named_steps[
        "preprocessamento"
    ]

    pipeline_categorico = (
        preprocessador.named_transformers_[
            "categoricas"
        ]
    )

    codificador = pipeline_categorico.named_steps[
        "codificador"
    ]

    padrao = codificador.categories_[0][0]
    cep = codificador.categories_[1][0]

    return pd.DataFrame([{
        "Área Construída (m2)": 80.0,
        "Fração Ideal": 0.01,
        "Padrão (IPTU)": padrao,
        "cep_3": cep,
    }])


def test_arquivos_exportados():
    assert CAMINHO_MODELO.exists()
    assert CAMINHO_METADATA.exists()

    assert CAMINHO_MODELO.stat().st_size > 0
    assert CAMINHO_METADATA.stat().st_size > 0


def test_metadata(artefatos):
    _, metadata = artefatos

    assert metadata["modelo"] == (
        "HistGradientBoostingRegressor"
    )

    assert metadata["variaveis_entrada"] == (
        COLUNAS_ESPERADAS
    )

    assert metadata["metricas_teste"]["registros"] == 10105

    assert metadata["metricas_teste"]["mae"] > 0


def test_previsao_individual(
    artefatos,
    entrada_exemplo,
):
    modelo, _ = artefatos

    previsao = modelo.predict(entrada_exemplo)

    assert len(previsao) == 1
    assert np.isfinite(previsao).all()

    # Mesma regra de pós-processamento do app.
    valor_exibido = max(float(previsao[0]), 0.0)

    assert valor_exibido >= 0


def test_previsoes_em_lote(
    artefatos,
    entrada_exemplo,
):
    modelo, _ = artefatos

    entradas = pd.concat(
        [entrada_exemplo] * 3,
        ignore_index=True,
    )

    entradas.loc[
        1, "Área Construída (m2)"
    ] = 120.0

    entradas.loc[
        2, "Área Construída (m2)"
    ] = 160.0

    previsoes = modelo.predict(entradas)

    assert len(previsoes) == 3
    assert np.isfinite(previsoes).all()


def test_categoria_geografica_desconhecida(
    artefatos,
    entrada_exemplo,
):
    modelo, _ = artefatos

    entrada = entrada_exemplo.copy()

    # Testa o tratamento técnico de categoria
    # desconhecida; não valida uso fora da região.
    entrada["cep_3"] = "ZZZ"

    previsao = modelo.predict(entrada)

    assert len(previsao) == 1
    assert np.isfinite(previsao).all()
