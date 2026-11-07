# Real Estate Price Prediction — São Paulo

## Previsão de preços de apartamentos com Machine Learning

Projeto de Ciência de Dados que estima valores de transações de apartamentos residenciais no município de São Paulo com base em registros públicos do ITBI de 2025. O trabalho reúne auditoria de dados, engenharia de atributos, validação temporal, modelagem preditiva, testes automatizados e uma aplicação web em Streamlit.

## Problema de negócio

Estimar o valor de transação de apartamentos a partir de características cadastrais disponíveis nos registros do ITBI, oferecendo uma referência estatística para exploração de dados imobiliários.

**Importante:** a estimativa não é um laudo de avaliação, não substitui análise profissional e não representa uma oferta de compra ou venda.

## Dados e preparação

- Fonte: planilha pública de guias de ITBI pagas do município de São Paulo, com 12 abas mensais referentes a 2025.
- Registros brutos: **230.525**.
- Registros elegíveis após filtros iniciais: **64.951**.
- Registros com data de transação em 2025: **64.227**.
- Base candidata após controles de qualidade e duplicidades: **62.121**.
- Segmento estudado: apartamentos residenciais, identificados por `Uso (IPTU) = 20`.

Os arquivos brutos não são incluídos no repositório. O notebook documenta as etapas de processamento.

## Variáveis do modelo

| Variável | Descrição |
|---|---|
| `Área Construída (m2)` | Área construída cadastral do imóvel |
| `Fração Ideal` | Fração ideal informada no cadastro |
| `Padrão (IPTU)` | Código cadastral do padrão construtivo |
| `cep_3` | Prefixo de três dígitos do CEP |

A variável-alvo é o valor de transação declarado nos registros analisados.

## Metodologia

1. Consolidação das planilhas mensais e verificação de consistência.
2. Filtragem de transações elegíveis e controle de registros atípicos e duplicados.
3. Engenharia de atributos e preparação de variáveis numéricas e categóricas.
4. Separação temporal para treinamento, validação e teste, evitando embaralhamento aleatório entre períodos.
5. Comparação de modelos com uma referência inicial (*baseline*).
6. Treinamento e avaliação final do `HistGradientBoostingRegressor`.
7. Exportação do pipeline e desenvolvimento da aplicação Streamlit.

### Divisão temporal

| Conjunto | Período | Registros |
|---|---|---:|
| Treinamento | Janeiro a setembro de 2025 | 46.225 |
| Validação | Outubro de 2025 | 5.791 |
| Teste | Novembro e dezembro de 2025 | 10.105 |

### Resultados no conjunto de teste

| Métrica | Resultado |
|---|---:|
| MAE | R$ 194.276,29 |
| RMSE | R$ 407.130,39 |
| WAPE | 26,87% |

Esses resultados descrevem o desempenho agregado no conjunto de teste. Erros individuais podem ser substancialmente maiores.

## Aplicação Streamlit

A interface permite informar CEP, área construída, padrão IPTU e fração ideal para gerar uma estimativa. O formulário é direcionado a apartamentos e apresenta os códigos de padrão residencial vertical `20` a `25`.

Para executar localmente, em um ambiente virtual Python compatível:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O arquivo do modelo e seus metadados devem estar disponíveis em `models/`.

## Estrutura do projeto

```text
real-estate-price-prediction/
├── app.py
├── real_estate_price_prediction.ipynb
├── models/
│   ├── modelo_hgb.joblib
│   └── metadata_modelo.json
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Testes

```bash
python -m pytest tests -v
```

## Limitações

- A base corresponde a transações registradas no município de **São Paulo**, não à Grande São Paulo como um todo.
- O modelo não recebe diretamente atributos como dormitórios, vagas de garagem, andar, estado de conservação e reformas.
- A qualidade dos registros cadastrais e a distribuição dos imóveis influenciam a precisão das estimativas.
- O uso fora do escopo geográfico e temporal estudado exige nova avaliação.
- A aplicação é demonstrativa e não substitui avaliação imobiliária profissional.

## Tecnologias

Python, pandas, NumPy, scikit-learn, Jupyter Notebook, Streamlit, joblib e pytest.

## Autor

**Luis Eduardo Sousa**  
[GitHub](https://github.com/edusousa78) · [LinkedIn](https://www.linkedin.com/in/luissousa78)

Projeto integrante de um portfólio de Ciência de Dados.
