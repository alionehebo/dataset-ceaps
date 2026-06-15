import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard Senado", layout="wide")

# --- FUNÇÃO DE CARREGAMENTO (Com Cache para evitar erros de memória) ---
@st.cache_data
def load_data():
    url = "https://www.senado.gov.br/transparencia/LAI/verba/despesa_ceaps_2022.csv"
    # Usando os parâmetros que descobrimos na Etapa 1
    data = pd.read_csv(url, sep=';', encoding='latin1', skiprows=1)
    
    # LIMPEZA IMEDIATA (Para o df já nascer pronto)
    data['VALOR_REEMBOLSADO'] = data['VALOR_REEMBOLSADO'].str.replace(',', '.').astype(float)
    data['DATA'] = pd.to_datetime(data['DATA'], dayfirst=True, errors='coerce')
    return data

# --- EXECUÇÃO DO CARREGAMENTO ---
# Aqui garantimos que o 'df' existe antes de qualquer outra linha rodar
try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop() # Interrompe a aplicação se não carregar o dado

# --- INTERFACE DO USUÁRIO (UI) ---
st.title("🎯 Análise de Gastos - Senado 2022")

# Sidebar para filtros
st.sidebar.header("Filtros")
senadores = sorted(df['SENADOR'].unique())
senador_selecionado = st.sidebar.selectbox("Selecione um Senador", ["Todos"] + senadores)

# Lógica de Filtro
if senador_selecionado != "Todos":
    df_plot = df[df['SENADOR'] == senador_selecionado]
else:
    df_plot = df

# --- VISUALIZAÇÃO ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Maiores Gastos por Senador")
    
    # 1. Criamos a figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2. Lógica de Agrupamento Correta:
    # Agrupamos por SENADOR, somamos o valor e pegamos os 10 maiores
    top_gastadores = df_plot.groupby('SENADOR')['VALOR_REEMBOLSADO'].sum().sort_values(ascending=False).head(10).reset_index()
    
    # 3. Geramos o gráfico usando o novo DataFrame 'top_gastadores'
    sns.barplot(
        data=top_gastadores, 
        x='VALOR_REEMBOLSADO', 
        y='SENADOR', 
        ax=ax, 
        palette='magma'
    )
    
    ax.set_xlabel("Soma de Gastos (R$)")
    ax.set_ylabel("Senador")
    
    st.pyplot(fig)

    #Dados brutos

with col2:
    st.subheader("Visualização dos Dados")
    
    # Se "Todos" estiver selecionado, mostramos os maiores gastos individuais de 2022
    if senador_selecionado == "Todos":
        st.write("Exibindo os 100 maiores reembolsos individuais de 2022:")
        # Ordenamos pelo valor para não ver apenas o primeiro senador da lista
        dados_exibicao = df_plot.sort_values(by='VALOR_REEMBOLSADO', ascending=False).head(100)
    else:
        st.write(f"Notas fiscais de: {senador_selecionado}")
        # Se um senador for selecionado, mostramos os gastos dele do mais caro para o mais barato
        dados_exibicao = df_plot.sort_values(by='VALOR_REEMBOLSADO', ascending=False)

    # Exibindo a tabela com colunas selecionadas para não poluir a tela
    colunas_visiveis = ['SENADOR', 'DATA', 'TIPO_DESPESA', 'VALOR_REEMBOLSADO', 'FORNECEDOR']
    st.dataframe(dados_exibicao[colunas_visiveis], use_container_width=True)

    # --- CRIANDO ABAS NA INTERFACE ---
aba_historico, aba_previsao = st.tabs([" Análise Histórica", " Previsão do Orçamento (Avançado)"])

with aba_historico:
    # Mova os códigos dos seus gráficos antigos (Barras e Dados Brutos) para dentro daqui
    st.subheader("Visualizações de Gastos Passados")
    # ... (seu código antigo de colunas, col1, col2, etc, fica aqui dentro) ...


with aba_previsao:
    st.header("🔮 Previsão de Gastos para os Próximos 90 Dias")
    st.markdown("Esta aba utiliza o algoritmo de Machine Learning **Facebook Prophet** para antecipar o comportamento dos gastos.")

    try:
        # 1. Carregar o modelo que a fábrica salvou
        with open('modelo_prophet.pkl', 'rb') as f:
            model_carregado = pickle.load(f)

        # 2. Gerar o futuro e fazer a previsão na hora (isso é super rápido porque o modelo já está treinado!)
        future = model_carregado.make_future_dataframe(periods=90)
        forecast = model_carregado.predict(future)
        
        # Ajustar valores negativos para zero (Regra de negócio que validamos!)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

        # 3. Plotar o Gráfico do próprio Prophet na Web
        st.subheader("Linha de Tendência Futura")
        fig_prophet = model_carregado.plot(forecast)
        st.pyplot(fig_prophet)

        # 4. Mostrar os dados em formato de insight de negócio
        st.subheader("📈 Resumo das Próximas Previsões Semanais")
        futuro_filtrado = forecast[forecast['ds'] > '2023-01-31'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(14)
        
        # Renomear colunas para o pessoal de negócio entender melhor
        futuro_filtrado.columns = ['Data Prevista', 'Previsão Alvo (R$)', 'Cenário Mínimo (R$)', 'Cenário Máximo (R$)']
        st.dataframe(futuro_filtrado, use_container_width=True)

    except FileNotFoundError:
        st.error("⚠️ O arquivo 'modelo_prophet.pkl' não foi encontrado. Rode o script 'treinar_modelo.py' primeiro no terminal.")