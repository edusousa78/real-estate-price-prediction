"""Estimativa de valor de apartamentos — interface para consumidores.

O campo Fração Ideal é armazenado como proporção entre 0 e 1 na
base de IPTU (aba EXPLICAÇÕES da fonte original).
"""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Quanto vale meu apartamento?",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "models" / "modelo_hgb.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata_modelo.json"

# Descrições oficiais da aba "Tabela de PADRÕES" da planilha ITBI.
# Não representam uma escala comercial de qualidade.
CODIGOS_APARTAMENTO = {"20", "21", "22", "23", "24", "25"}

DESCRICOES_PADRAO = {
    0: "Terreno",
    **{n: "Residencial horizontal" for n in range(10, 16)},
    **{n: "Residencial vertical" for n in range(20, 26)},
    **{n: "Comercial horizontal" for n in range(31, 36)},
    **{n: "Comercial vertical" for n in range(41, 46)},
    81: "Barracão / telheiro",
    82: "Uso industrial ou de serviços",
    83: "Uso industrial ou de serviços",
    84: "Uso industrial ou de serviços",
    85: "Indústria",
    91: "Edifício de garagens",
}


def formatar_moeda(valor):
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_decimal(valor, casas=4):
    return f"{valor:.{casas}f}".replace(".", ",")


def ler_numero(texto):
    texto = str(texto).strip()
    if not re.fullmatch(r"\d+(?:[.,]\d+)?", texto):
        raise ValueError("Número inválido")
    valor = float(texto.replace(",", "."))
    if not np.isfinite(valor):
        raise ValueError("Número inválido")
    return valor


def validar_cep(texto):
    # Permite 8 dígitos ou formato XXXXX-XXX, sem caracteres adicionais.
    texto = texto.strip()
    if not re.fullmatch(r"(?:\d{8}|\d{5}-\d{3})", texto):
        return None
    return texto.replace("-", "")


def descricao_padrao(valor):
    try:
        codigo = int(str(valor))
        descricao = DESCRICOES_PADRAO.get(codigo)
    except (TypeError, ValueError):
        descricao = None
    if descricao:
        return f"{descricao} — código {valor}"
    return f"Código cadastral {valor}"


@st.cache_resource
def carregar_modelo():
    # Carregue somente artefatos produzidos pelo próprio projeto.
    return joblib.load(MODELO_PATH)


@st.cache_data
def carregar_metadata():
    with METADATA_PATH.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def obter_categorias(modelo):
    preprocessador = modelo.named_steps["preprocessamento"]
    pipeline = preprocessador.named_transformers_["categoricas"]
    encoder = pipeline.named_steps["codificador"]
    return {
        "padroes": list(encoder.categories_[0]),
        "prefixos": [str(v) for v in encoder.categories_[1]],
    }


st.title("🏡 Quanto vale meu apartamento?")
st.write("Faça uma simulação do valor do seu apartamento em poucos passos.")
st.info("Disponível para apartamentos localizados no **município de São Paulo**.")
st.caption("Estimativa estatística baseada em transações imobiliárias de 2025.")

if not MODELO_PATH.is_file() or not METADATA_PATH.is_file():
    st.error("O serviço está temporariamente indisponível.")
    st.stop()

try:
    modelo = carregar_modelo()
    metadata = carregar_metadata()
    categorias = obter_categorias(modelo)
    padroes_apartamento = [
        codigo for codigo in categorias["padroes"]
        if str(codigo).removesuffix(".0") in CODIGOS_APARTAMENTO
    ]
    if not padroes_apartamento:
        raise ValueError("Nenhum padrão residencial vertical encontrado")
except Exception:
    st.error("Não foi possível iniciar a simulação.")
    st.stop()

st.divider()
st.subheader("🏠 Informações do apartamento")
st.write("Preencha os dados do imóvel. Você pode consultar as informações cadastrais no IPTU ou na matrícula.")

with st.form("simulacao_apartamento"):
    cep = st.text_input(
        "CEP do apartamento",
        placeholder="01001-000",
        help="Informe os oito dígitos do CEP do imóvel, com ou sem hífen.",
    )

    area = st.number_input(
        "Área construída cadastrada (m²)",
        min_value=20.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
        help="Use a área construída do cadastro do IPTU. Ela pode diferir da área privativa do anúncio.",
    )

    st.markdown("#### Dados do cadastro")
    padrao = st.selectbox(
        "Padrão de construção no IPTU",
        options=padroes_apartamento,
        index=None,
        placeholder="Selecione o padrão informado no cadastro",
        format_func=descricao_padrao,
        help="Escolha o código do IPTU correspondente ao apartamento. Os códigos apresentados são residenciais verticais; não representam uma nota de luxo.",
    )

    st.caption("Mostramos somente os padrões residenciais verticais compatíveis com a finalidade desta simulação.")

    st.markdown("#### Participação do apartamento no condomínio")
    st.write(
        "A **fração ideal** é a participação da unidade no terreno e nas áreas comuns. "
        "Você pode encontrá-la na matrícula ou na documentação do condomínio."
    )

    formato_fracao = st.radio(
        "Como deseja informar a fração ideal?",
        ["Percentual (%)", "Número decimal do cadastro"],
        horizontal=True,
    )

    if formato_fracao == "Percentual (%)":
        fracao_texto = st.text_input(
            "Fração ideal (%)",
            placeholder="Ex.: 0,1",
            help="Digite 0,1 para 0,1%; 1 para 1%; 10 para 10%. Não é necessário digitar o símbolo %.",
        )
        st.caption("Exemplo: **0,1** significa **0,1%**, equivalente a **0,001** no cálculo.")
    else:
        fracao_texto = st.text_input(
            "Fração ideal (número decimal)",
            placeholder="Ex.: 0,001",
            help="Use a proporção decimal do cadastro: 0,001 equivale a 0,1%; 0,01 equivale a 1%.",
        )
        st.caption("Exemplo: **0,001** corresponde a **0,1%** de participação.")

    with st.expander("Onde encontro a fração ideal?"):
        st.write(
            "Consulte a matrícula do apartamento, a instituição do condomínio ou o cadastro imobiliário. "
            "Não confunda fração ideal com o percentual de entrada do financiamento."
        )

    calcular = st.form_submit_button(
        "🏡 Calcular estimativa",
        type="primary",
        use_container_width=True,
    )

if calcular:
    cep_limpo = validar_cep(cep)
    if cep_limpo is None:
        st.error("Informe um CEP com oito números, por exemplo 01001-000.")
        st.stop()

    prefixo = cep_limpo[:3]
    if prefixo not in categorias["prefixos"]:
        st.warning("Ainda não é possível calcular uma estimativa para esse CEP com o modelo atual.")
        st.stop()

    if padrao is None:
        st.error("Selecione o padrão de construção informado no IPTU.")
        st.stop()

    try:
        fracao_informada = ler_numero(fracao_texto)
        if formato_fracao == "Percentual (%)":
            if not 0 < fracao_informada <= 100:
                raise ValueError("Percentual fora do intervalo")
            fracao_modelo = fracao_informada / 100.0
            percentual = fracao_informada
        else:
            if not 0 < fracao_informada <= 1:
                raise ValueError("Proporção fora do intervalo")
            fracao_modelo = fracao_informada
            percentual = fracao_modelo * 100.0
    except (ValueError, OverflowError):
        st.error(
            "Confira a fração ideal. Informe um percentual maior que 0 e até 100, "
            "ou um número decimal maior que 0 e até 1, conforme a opção escolhida."
        )
        st.stop()

    entrada = pd.DataFrame([{
        "Área Construída (m2)": float(area),
        "Fração Ideal": fracao_modelo,
        "Padrão (IPTU)": str(padrao),
        "cep_3": prefixo,
    }])

    try:
        entrada = entrada[metadata["variaveis_entrada"]]
        previsao = float(modelo.predict(entrada)[0])
        if not np.isfinite(previsao):
            raise ValueError("Previsão não finita")
        previsao = max(0.0, previsao)
    except Exception:
        st.error("Não foi possível calcular a estimativa com os dados informados.")
        st.stop()

    st.divider()
    st.success("Sua estimativa está pronta!")
    st.subheader("💰 Valor estimado do apartamento")
    st.metric("Estimativa de valor", formatar_moeda(previsao))
    st.info(
        "Esta é uma estimativa estatística baseada em transações de 2025, "
        "não uma avaliação profissional ou garantia de preço de venda."
    )

    with st.expander("Conferir os dados utilizados"):
        st.write(f"**CEP:** {cep_limpo[:5]}-{cep_limpo[5:]}")
        st.write(f"**Área construída:** {formatar_decimal(area, 2)} m²")
        st.write(f"**Padrão IPTU:** {descricao_padrao(padrao)}")
        st.write(f"**Fração ideal:** {formatar_decimal(percentual, 4)}%")
        st.write(f"**Proporção utilizada no cálculo:** {formatar_decimal(fracao_modelo, 6)}")

st.divider()
with st.expander("Como funciona a estimativa?"):
    st.write(
        "O cálculo utiliza dados históricos de transações de apartamentos no município de São Paulo. "
        "Considera a área construída cadastral, a localização aproximada pelo CEP, "
        "o padrão cadastral do IPTU e a fração ideal."
    )
    st.write(
        "Não considera diretamente reformas, vagas, andar, vista ou estado de conservação. "
        "O valor real de mercado pode ser diferente."
    )

st.caption("Ferramenta demonstrativa. Não substitui avaliação imobiliária profissional.")
