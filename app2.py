import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Ignorar avisos que podem poluir o dashboard
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Análise Imobiliária - Ames", layout="wide")


# Injeta HTML e CSS para criar um cabeçalho fixo no topo da página
header_html = """
<style>
    /* Estilo do container do cabeçalho */
    #app-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #0f1116; /* Cor de fundo escura, similar ao tema do Streamlit */
        color: #fafafa;            /* Cor do texto clara */
        padding: 10px 25px;
        z-index: 999;              /* Garante que o cabeçalho fique sobre os outros elementos */
        border-bottom: 1px solid #31333f;
        text-align: center;
    }
    /* Estilo do título dentro do cabeçalho */
    #app-header h2 {
        margin: 0;
        font-size: 26px;
        font-weight: 600;
    }
    /* Estilo do parágrafo com os nomes */
    #app-header p {
        margin: 5px 0 0 0;
        font-size: 16px;
        color: #a3a3a3; /* Cor mais suave para o subtítulo */
    }
    /* Adiciona um espaço no topo da página para o conteúdo principal não ficar escondido atrás do cabeçalho */
    .main .block-container {
        padding-top: 7rem; 
    }
</style>

<div id="app-header">
    <h2>🏡 Análise de Dados Imobiliários: Ames, Iowa</h2>
    <p>Integrantes: Pedro Russo e Daniel Vianna</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# st.title("...")
# st.markdown("...")

# --- CARREGAMENTO E CACHE DOS DADOS ---
@st.cache_data
def load_data():
    """Carrega e faz um pré-processamento leve nos dados."""
    df = pd.read_csv("AmesHousing.csv")
    # Remover espaços nos nomes das colunas para facilitar o uso
    df.columns = df.columns.str.replace(' ', '')
    # Corrigir valores ausentes em colunas-chave que serão usadas
    # (Estratégia simples: para variáveis numéricas, usar mediana; para categóricas, usar moda)
    for col in ['MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageCars', 'OverallQual']:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    for col in ['BsmtQual', 'FireplaceQu', 'GarageType', 'MSZoning', 'HouseStyle']:
         if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

df_original = load_data()

# --- BARRA LATERAL DE NAVEGAÇÃO E FILTROS ---
st.sidebar.title("Navegação")
aba = st.sidebar.radio("Selecione a Análise:", ["📊 Visão Geral", "📈 Etapa I – ANOVA", "📉 Etapa II – Regressão", "📘 Sobre o Projeto"])

st.sidebar.markdown("---")
st.sidebar.header("Filtros Gerais")
st.sidebar.markdown("Filtre os dados para refinar as análises em todas as abas.")

# Filtro por Qualidade Geral (OverallQual)
qualidade_geral = st.sidebar.multiselect(
    'Filtre por Qualidade Geral do Imóvel:',
    options=sorted(df_original['OverallQual'].unique()),
    default=sorted(df_original['OverallQual'].unique())
)

# Filtro por Ano de Construção (YearBuilt)
ano_min, ano_max = int(df_original['YearBuilt'].min()), int(df_original['YearBuilt'].max())
ano_range = st.sidebar.slider(
    'Filtre por Ano de Construção:',
    min_value=ano_min,
    max_value=ano_max,
    value=(ano_min, ano_max)
)

# Aplicar filtros ao DataFrame
df = df_original[
    (df_original['OverallQual'].isin(qualidade_geral)) &
    (df_original['YearBuilt'] >= ano_range[0]) &
    (df_original['YearBuilt'] <= ano_range[1])
].copy()


# --- ABA 1: VISÃO GERAL ---
if aba == "📊 Visão Geral":
    st.header("📊 Visão Geral dos Dados de Imóveis")
    st.markdown(f"Exibindo *{df.shape[0]}* de *{df_original.shape[0]}* imóveis após a aplicação dos filtros.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição do Preço de Venda (SalePrice)")
        fig, ax = plt.subplots()
        sns.histplot(df['SalePrice'], kde=True, ax=ax)
        ax.set_title("Distribuição do Preço de Venda")
        ax.set_xlabel("Preço de Venda ($)")
        ax.set_ylabel("Frequência")
        st.pyplot(fig)

    with col2:
        st.subheader("Preço de Venda vs. Área Construída")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', alpha=0.5, ax=ax)
        ax.set_title("Preço de Venda vs. Área de Estar (GrLivArea)")
        ax.set_xlabel("Área de Estar (Pés Quadrados)")
        ax.set_ylabel("Preço de Venda ($)")
        st.pyplot(fig)
        
    st.subheader("Amostra dos Dados")
    st.dataframe(df.head())


# --- ABA 2: ANOVA ---
elif aba == "📈 Etapa I – ANOVA":
    st.header("📈 Análise de Variância (ANOVA)")
    st.markdown("""
    *Objetivo:* Verificar se existe uma diferença estatisticamente significativa no preço médio de venda (SalePrice) 
    entre diferentes categorias de uma variável escolhida.
    """)

    # Seleção de variáveis
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() < 10]
    var_cat = st.selectbox(
        "*1. Escolha a variável categórica para análise:*",
        options=cat_cols,
        index=cat_cols.index('BldgType') if 'BldgType' in cat_cols else 0,
        help="A ANOVA de fator único compara o preço médio entre os grupos de UMA variável por vez."
    )

    if var_cat:
        st.markdown("---")
        st.subheader(f"Análise de 'SalePrice' por '{var_cat}'")

        # Visualização: Boxplot
        st.write("*Visualização da Distribuição:*")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x=var_cat, y='SalePrice', ax=ax)
        ax.set_title(f"Distribuição do Preço de Venda por {var_cat}")
        ax.set_xlabel(var_cat)
        ax.set_ylabel("Preço de Venda ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("2. Verificação dos Pressupostos da ANOVA")

        formula = f"SalePrice ~ C({var_cat})"
        try:
            model_anova = ols(formula, data=df.dropna(subset=[var_cat, 'SalePrice'])).fit()
            residuals = model_anova.resid
        except Exception as e:
            st.error(f"Não foi possível ajustar o modelo para os resíduos. Erro: {e}")
            st.stop()
            
        st.markdown("*a) Normalidade dos Resíduos*")
        shapiro_test = stats.shapiro(residuals)
        p_valor_shapiro = shapiro_test.pvalue
        if p_valor_shapiro < 0.05:
            st.warning(f"*Pressuposto violado:* Os resíduos *não* seguem uma distribuição normal (p-valor do teste de Shapiro-Wilk = {p_valor_shapiro:.4f}).")
        else:
            st.success(f"*Pressuposto atendido:* Os resíduos parecem seguir uma distribuição normal (p-valor do teste de Shapiro-Wilk = {p_valor_shapiro:.4f}).")
        
        fig = sm.qqplot(residuals, line='s', fit=True)
        plt.title("Gráfico Q-Q dos Resíduos")
        st.pyplot(fig)

        st.markdown("*b) Homocedasticidade*")
        df_anova_clean = df.dropna(subset=[var_cat, 'SalePrice'])
        groups = [df_anova_clean['SalePrice'][df_anova_clean[var_cat] == g] for g in df_anova_clean[var_cat].unique()]
        groups_for_levene = [g for g in groups if len(g) > 1]
        if len(groups_for_levene) > 1:
            levene_test = stats.levene(*groups_for_levene)
            p_valor_levene = levene_test.pvalue
            if p_valor_levene < 0.05:
                st.warning(f"*Pressuposto violado:* As variâncias *não* são homogêneas entre os grupos (p-valor do teste de Levene = {p_valor_levene:.4f}).")
            else:
                st.success(f"*Pressuposto atendido:* As variâncias são homogêneas entre os grupos (p-valor do teste de Levene = {p_valor_levene:.4f}).")
        else:
            st.error("Não há grupos suficientes para realizar o teste de Levene.")

        st.markdown("---")
        st.subheader("3. Resultados do Teste Estatístico")
        
        if p_valor_shapiro >= 0.05 and p_valor_levene >= 0.05:
            st.info("*Teste Aplicado: ANOVA* (pois os pressupostos foram atendidos).")
            anova_table = sm.stats.anova_lm(model_anova, typ=2)
            st.write(anova_table)
            p_valor_final = anova_table.iloc[0]['PR(>F)']
            if p_valor_final < 0.05:
                st.success(f"*Conclusão:* Existe uma diferença estatisticamente significativa nos preços médios de venda entre as diferentes categorias de '{var_cat}' (p-valor = {p_valor_final:.4f}).")
            else:
                st.warning(f"*Conclusão:* Não há evidência de uma diferença significativa nos preços médios de venda para '{var_cat}' (p-valor = {p_valor_final:.4f}).")

        else:
            st.info("*Teste Aplicado: Kruskal-Wallis* (alternativa não paramétrica, pois um ou mais pressupostos da ANOVA foram violados).")
            kruskal_test = stats.kruskal(*groups)
            p_valor_kruskal = kruskal_test.pvalue
            st.write(f"*Estatística H:* {kruskal_test.statistic:.4f}")
            st.write(f"*P-valor:* {p_valor_kruskal:.4f}")
            if p_valor_kruskal < 0.05:
                st.success(f"*Conclusão:* Existe uma diferença estatisticamente significativa nas distribuições de preço de venda entre as diferentes categorias de '{var_cat}' (p-valor = {p_valor_kruskal:.4f}).")
            else:
                st.warning(f"*Conclusão:* Não há evidência de uma diferença significativa nas distribuições de preço para '{var_cat}' (p-valor = {p_valor_kruskal:.4f}).")
        
        st.markdown("---")
        st.subheader("💡 Insights e Recomendações (ANOVA)")
        st.markdown(f"A análise da variável *'{var_cat}'* indica que ela *{'tem' if ('p_valor_final' in locals() and p_valor_final < 0.05) or ('p_valor_kruskal' in locals() and p_valor_kruskal < 0.05) else 'não tem'}* um impacto estatisticamente significativo no preço do imóvel.")

# --- ABA 3: REGRESSÃO LINEAR ---
elif aba == "📉 Etapa II – Regressão":
    st.header("📉 Modelagem Preditiva com Regressão Linear")
    st.markdown("""
    *Objetivo:* Construir um modelo para prever o SalePrice com base em múltiplas características do imóvel.
    """)

    st.subheader("1. Seleção de Variáveis e Transformação")
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_reg = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() < 20]
    
    desired_cat_defaults = ['MSZoning', 'HouseStyle', 'BldgType']
    actual_cat_defaults = [col for col in desired_cat_defaults if col in cat_cols_reg]
    
    desired_num_defaults = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'OverallQual']
    actual_num_defaults = [col for col in desired_num_defaults if col in num_cols]

    col1, col2 = st.columns(2)
    with col1:
        vars_cont = st.multiselect(
            "*Escolha as variáveis contínuas (numéricas):*",
            options=num_cols,
            default=actual_num_defaults
        )
    with col2:
        vars_cat = st.multiselect(
            "*Escolha as variáveis categóricas:*",
            options=cat_cols_reg,
            default=actual_cat_defaults
        )
        
    log_transform = st.checkbox("Aplicar transformação logarítmica em SalePrice e nas variáveis contínuas? (Modelo Log-Log)", value=True)

    if len(vars_cont) > 0 and len(vars_cat) > 0:
        st.markdown("---")
        st.subheader("2. Ajuste do Modelo e Diagnósticos")
        
        df_model = df[['SalePrice'] + vars_cont + vars_cat].dropna()
        
        if log_transform:
            df_model['SalePrice'] = np.log1p(df_model['SalePrice'])
            for col in vars_cont:
                if np.issubdtype(df_model[col].dtype, np.number):
                    df_model[col] = np.log1p(df_model[col])

        df_model = pd.get_dummies(df_model, columns=vars_cat, drop_first=True, dtype=float)
        
        X = df_model.drop('SalePrice', axis=1)
        y = df_model['SalePrice']
        X = sm.add_constant(X)
        
        try:
            model_reg = sm.OLS(y, X).fit()
            
            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("*Resumo do Modelo:*")
                st.text(model_reg.summary())
            
            with col2:
                st.write("*Métricas de Desempenho:*")
                y_pred = model_reg.predict(X)
                
                if log_transform:
                    y_true_orig = np.expm1(y)
                    y_pred_orig = np.expm1(y_pred)
                else:
                    y_true_orig = y
                    y_pred_orig = y_pred

                r2 = model_reg.rsquared_adj
                rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
                mae = mean_absolute_error(y_true_orig, y_pred_orig)
                
                st.metric(label="R² Ajustado", value=f"{r2:.4f}")
                st.metric(label="RMSE (Erro Médio Quadrático)", value=f"${rmse:,.2f}")
                st.metric(label="MAE (Erro Médio Absoluto)", value=f"${mae:,.2f}")
                st.markdown(f"*Interpretação:* O modelo explica aproximadamente *{r2:.1%}* da variância no preço de venda. Em média, as previsões do modelo erram em *${mae:,.2f}* (MAE).")

            st.markdown("---")
            st.subheader("3. Diagnóstico dos Pressupostos")
            
            residuals_reg = model_reg.resid

            diag1, diag2 = st.columns(2)
            with diag1:
                st.markdown("*a) Normalidade dos Resíduos*")
                shapiro_reg = stats.shapiro(residuals_reg)
                if shapiro_reg.pvalue < 0.05:
                    st.warning(f"P-valor (Shapiro-Wilk): {shapiro_reg.pvalue:.4f}. Os resíduos podem não ser normais.")
                else:
                    st.success(f"P-valor (Shapiro-Wilk): {shapiro_reg.pvalue:.4f}. Resíduos parecem normais.")
                fig = sm.qqplot(residuals_reg, line='s', fit=True)
                plt.title("Q-Q Plot dos Resíduos")
                st.pyplot(fig)

                st.markdown("*b) Homocedasticidade (Breusch-Pagan Test)*")
                bp_test = het_breuschpagan(residuals_reg, model_reg.model.exog)
                if bp_test[1] < 0.05:
                    st.warning(f"P-valor: {bp_test[1]:.4f}. Há evidência de heterocedasticidade.")
                else:
                    st.success(f"P-valor: {bp_test[1]:.4f}. Não há evidência de heterocedasticidade.")
                fig, ax = plt.subplots()
                sns.scatterplot(x=model_reg.fittedvalues, y=residuals_reg, ax=ax, alpha=0.5)
                ax.axhline(0, color='red', linestyle='--')
                ax.set_xlabel("Valores Ajustados")
                ax.set_ylabel("Resíduos")
                ax.set_title("Resíduos vs. Valores Ajustados")
                st.pyplot(fig)

            with diag2:
                st.markdown("*c) Multicolinearidade (VIF)*")
                X_no_const = X.drop('const', axis=1)
                vif_data = pd.DataFrame()
                vif_data["feature"] = X_no_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])]
                st.dataframe(vif_data.style.apply(
                    lambda x: ['background-color: #FF7F7F' if v > 5 else '' for v in x], subset=['VIF']))
                st.markdown("Valores de VIF acima de 5 podem indicar multicolinearidade.")

            st.markdown("---")
            st.subheader("💡 4. Interpretação dos Coeficientes e Recomendações Práticas")
            coef_df = pd.DataFrame({
                'Coeficiente': model_reg.params,
                'p-valor': model_reg.pvalues
            }).reset_index().rename(columns={'index': 'Variável'})
            
            coef_significativos = coef_df[(coef_df['p-valor'] < 0.05) & (coef_df['Variável'] != 'const')]
            
            if not coef_significativos.empty:
                st.markdown("As seguintes variáveis têm um impacto estatisticamente significativo no preço de venda:")
                for _, row in coef_significativos.iterrows():
                    var, coef = row['Variável'], row['Coeficiente']
                    if log_transform:
                        impacto = "aumenta" if coef > 0 else "reduz"
                        st.markdown(f"• *{var}: Um aumento de 1% nesta variável, mantendo as outras constantes, *{impacto}* o preço do imóvel em aproximadamente *{abs(coef):.2%}**.")
                    else:
                        impacto = "aumenta" if coef > 0 else "reduz"
                        st.markdown(f"• *{var}: Um aumento de uma unidade nesta variável, mantendo as outras constantes, *{impacto}* o preço do imóvel em *${abs(coef):,.2f}**.")
            else:
                st.warning("Nenhuma variável selecionada apresentou impacto estatisticamente significativo no preço.")
        
        except Exception as e:
            st.error(f"Erro ao ajustar o modelo de regressão: {e}. Verifique as variáveis selecionadas ou se há dados suficientes após a filtragem.")
            
    else:
        st.warning("Por favor, selecione pelo menos uma variável contínua e uma categórica para a análise de regressão.")

# --- ABA 4: SOBRE O PROJETO ---
elif aba == "📘 Sobre o Projeto":
    st.header("📘 Sobre o Projeto e Autoria")
    st.markdown("""
    Este dashboard interativo foi desenvolvido como um projeto de análise de dados, com o objetivo de analisar os fatores que influenciam o preço de imóveis na cidade de Ames, Iowa, utilizando técnicas de ANOVA e Regressão Linear Múltipla.
    """)
    st.markdown("---")
    st.subheader("📌 Funcionalidades")
    st.markdown("""
    - ✔️ Análise de Variância (ANOVA) com verificação completa de pressupostos.
    - ✔️ Alternativa Robusta (Kruskal-Wallis) acionada automaticamente.
    - ✔️ Regressão Linear Múltipla com opção de transformação logarítmica (modelo log-log).
    - ✔️ Diagnósticos de Regressão: Normalidade (Shapiro-Wilk), Homocedasticidade (Breusch-Pagan) e Multicolinearidade (VIF).
    - ✔️ Métricas de Desempenho do Modelo: R², RMSE e MAE.
    - ✔️ Dashboard Interativo em Streamlit com filtros dinâmicos e geração de análises em tempo real.
    """)