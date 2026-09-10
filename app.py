import streamlit as st
import tensorflow as tf
import pickle
import numpy as np


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Predição de Reclamações",
    page_icon="⚡",
    layout="centered"
)


# ============================================================
# CARREGAMENTO DOS ARTEFATOS
# ============================================================

@st.cache_resource
def carregar_modelo():

    modelo = tf.keras.models.load_model(
        "modelo_reclamacao_15d.keras"
    )

    return modelo



@st.cache_resource
def carregar_artefatos():

    with open(
        "scaler_num.pkl",
        "rb"
    ) as f:
        scaler = pickle.load(f)


    with open(
        "encoder_classe.pkl",
        "rb"
    ) as f:
        encoder_classe = pickle.load(f)


    with open(
        "encoder_municipio.pkl",
        "rb"
    ) as f:
        encoder_municipio = pickle.load(f)


    with open(
        "config_modelo.pkl",
        "rb"
    ) as f:
        config = pickle.load(f)


    return (
        scaler,
        encoder_classe,
        encoder_municipio,
        config
    )



modelo = carregar_modelo()


scaler, encoder_classe, encoder_municipio, config = carregar_artefatos()


threshold = config["threshold"]



# ============================================================
# PREPARAÇÃO DAS ENTRADAS
# ============================================================

def preparar_entrada(
    classe,
    municipio,
    dados_numericos
):

    X_num = scaler.transform(
        [
            dados_numericos
        ]
    )


    X_classe = encoder_classe.transform(
        [
            [classe]
        ]
    )


    municipio_encoded = encoder_municipio.transform(
        [
            municipio
        ]
    )


    X_municipio = np.array(
        municipio_encoded
    ).reshape(
        -1,
        1
    )


    return (
        X_num,
        X_classe,
        X_municipio
    )



# ============================================================
# VARIÁVEIS DERIVADAS DE CONSUMO
# ============================================================

def calcular_consumo(
    m0,
    m1,
    m2
):

    consumos = np.array(
        [
            m0,
            m1,
            m2
        ]
    )


    media_3m = np.mean(
        consumos
    )


    # aproximação para demonstração
    # ideal seria possuir M3-M5
    media_6m = media_3m


    variacao_1m = (
        (m0 - m1) / m1
        if m1 != 0
        else 0
    )


    desvio_6m = np.std(
        consumos
    )


    return (
        media_3m,
        media_6m,
        variacao_1m,
        desvio_6m
    )



# ============================================================
# INTERFACE
# ============================================================

st.title(
    "⚡ Predição de Reclamações - MLP"
)


st.markdown(
    """
    Modelo de rede neural para estimar a probabilidade
    de uma unidade consumidora registrar reclamação
    nos próximos 15 dias.
    
    Threshold operacional utilizado:
    **0,40**
    """
)



st.sidebar.header(
    "Dados da Unidade Consumidora"
)



# ============================================================
# DADOS CATEGÓRICOS
# ============================================================

classe = st.sidebar.selectbox(
    "Classe de consumo",
    encoder_classe.categories_[0]
)


municipio = st.sidebar.selectbox(
    "Município",
    encoder_municipio.classes_
)



# ============================================================
# CONSUMO
# ============================================================

st.sidebar.subheader(
    "Histórico de Consumo"
)


consumo_m0 = st.sidebar.number_input(
    "Consumo atual M0",
    value=100.0
)


consumo_m1 = st.sidebar.number_input(
    "Consumo M1",
    value=100.0
)


consumo_m2 = st.sidebar.number_input(
    "Consumo M2",
    value=100.0
)



# ============================================================
# HISTÓRICO
# ============================================================

st.sidebar.subheader(
    "Histórico de relacionamento"
)


reclamacoes_30 = st.sidebar.number_input(
    "Quantidade reclamações 30 dias",
    min_value=0,
    value=0
)


reclamacoes_90 = st.sidebar.number_input(
    "Quantidade reclamações 90 dias",
    min_value=0,
    value=0
)


reclamacoes_180 = st.sidebar.number_input(
    "Quantidade reclamações 180 dias",
    min_value=0,
    value=0
)


servicos_30 = st.sidebar.number_input(
    "Quantidade serviços 30 dias",
    min_value=0,
    value=0
)


servicos_90 = st.sidebar.number_input(
    "Quantidade serviços 90 dias",
    min_value=0,
    value=0
)



# ============================================================
# PREDIÇÃO
# ============================================================

if st.button(
    "Calcular risco"
):


    # --------------------------------------------
    # Flags calculadas
    # --------------------------------------------

    flag_nunca_reclamou = int(
        (
            reclamacoes_30 == 0
        )
        and
        (
            reclamacoes_90 == 0
        )
        and
        (
            reclamacoes_180 == 0
        )
    )


    flag_nunca_servico = int(
        (
            servicos_30 == 0
        )
        and
        (
            servicos_90 == 0
        )
    )



    if flag_nunca_reclamou == 1:

        dias_ultima_reclamacao = -1

    else:

        dias_ultima_reclamacao = 30



    if flag_nunca_servico == 1:

        dias_ultimo_servico = -1

    else:

        dias_ultimo_servico = 30



    # --------------------------------------------
    # Variáveis consumo
    # --------------------------------------------

    media_3m, media_6m, variacao_consumo, desvio_consumo = calcular_consumo(
        consumo_m0,
        consumo_m1,
        consumo_m2
    )



    # --------------------------------------------
    # Vetor numérico final
    # Ordem igual ao treinamento
    # --------------------------------------------

    dados_numericos = [

        # MES_REFERENCIA
        7,

        # ANO_REFERENCIA
        2026,


        # Consumos

        consumo_m0,
        consumo_m1,
        consumo_m2,


        # Derivadas

        media_3m,
        media_6m,

        variacao_consumo,

        desvio_consumo,


        # Reclamações

        reclamacoes_30,
        reclamacoes_90,
        reclamacoes_180,


        # Última reclamação

        dias_ultima_reclamacao,


        # Flag nunca reclamou

        flag_nunca_reclamou,


        # Serviços

        servicos_30,
        servicos_90,


        # Suspensões

        0,


        # Religações

        0,


        # Último serviço

        dias_ultimo_servico,


        # Nunca teve serviço

        flag_nunca_servico,


        # Flags ausência consumo

        0,
        0,
        0,
        0,


        # Flags individuais M0/M1/M2

        0,
        0,
        0

    ]



    # --------------------------------------------
    # Transformação
    # --------------------------------------------

    X_num, X_classe, X_municipio = preparar_entrada(
        classe,
        municipio,
        dados_numericos
    )



    # --------------------------------------------
    # Predição
    # --------------------------------------------

    probabilidade = modelo.predict(
        [
            X_num,
            X_classe,
            X_municipio
        ],
        verbose=0
    )[0][0]



    # --------------------------------------------
    # Resultado
    # --------------------------------------------

    st.subheader(
        "Resultado"
    )


    st.metric(
        "Probabilidade de reclamação",
        f"{probabilidade:.2%}"
    )


    if probabilidade >= threshold:

        st.error(
            "⚠️ CLIENTE DE ALTO RISCO"
        )


        st.write(
            "Recomenda-se atuação preventiva."
        )


    else:

        st.success(
            "✅ CLIENTE DE BAIXO RISCO"
        )


        st.write(
            "Cliente abaixo do limite operacional."
        )


    st.caption(
        f"Threshold utilizado: {threshold}"
    )