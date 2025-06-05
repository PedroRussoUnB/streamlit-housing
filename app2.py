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

# --- CABEÇALHO FIXO COM CSS ---
header_html = """
<style>
    #app-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #0f1116; 
        color: #fafafa;          
        padding: 10px 25px;
        z-index: 999;             
        border-bottom: 1px solid #31333f;
        text-align: center;
    }
    #app-header h2 {
        margin: 0;
        font-size: 26px;
        font-weight: 600;
    }
    #app-header p {
        margin: 5px 0 0 0;
        font-size: 16px;
        color: #a3a3a3; 
    }
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

# --- CARREGAMENTO E CACHE DOS DADOS ---
@st.cache_data
def load_data():
    """Carrega e faz um pré-processamento leve nos dados."""
    df_loaded = pd.read_csv("AmesHousing.csv")
    df_loaded.columns = df_loaded.columns.str.replace(' ', '')
    
    numeric_cols_to_impute = ['MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageCars', 'OverallQual']
    for col in numeric_cols_to_impute:
        if col in df_loaded.columns:
            df_loaded[col].fillna(df_loaded[col].median(), inplace=True)
            
    categorical_cols_to_impute = ['BsmtQual', 'FireplaceQu', 'GarageType', 'MSZoning', 'HouseStyle', 'BldgType']
    for col in categorical_cols_to_impute:
         if col in df_loaded.columns:
            df_loaded[col].fillna(df_loaded[col].mode()[0], inplace=True)
    return df_loaded

df_original = load_data()

# --- BARRA LATERAL DE NAVEGAÇÃO E FILTROS ---
st.sidebar.title("Navegação")
aba = st.sidebar.radio("Selecione a Análise:", ["📊 Visão Geral", "📈 Etapa I – ANOVA", "📉 Etapa II – Regressão", "📘 Sobre o Projeto"])

st.sidebar.markdown("---")
st.sidebar.header("Filtros Gerais")
st.sidebar.markdown("Filtre os dados para refinar as análises em todas as abas.")

if 'OverallQual' in df_original.columns and not df_original['OverallQual'].empty:
    overall_qual_options = sorted(df_original['OverallQual'].unique())
    default_overall_qual = overall_qual_options
else:
    overall_qual_options = [0]
    default_overall_qual = [0]

qualidade_geral = st.sidebar.multiselect(
    'Filtre por Qualidade Geral do Imóvel:',
    options=overall_qual_options,
    default=default_overall_qual
)

if 'YearBuilt' in df_original.columns and not df_original['YearBuilt'].empty:
    ano_min_orig, ano_max_orig = int(df_original['YearBuilt'].min()), int(df_original['YearBuilt'].max())
else:
    ano_min_orig, ano_max_orig = 1900, 2020

ano_range = st.sidebar.slider(
    'Filtre por Ano de Construção:',
    min_value=ano_min_orig,
    max_value=ano_max_orig,
    value=(ano_min_orig, ano_max_orig)
)

df_filtered = df_original.copy()
if 'OverallQual' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['OverallQual'].isin(qualidade_geral)]
if 'YearBuilt' in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered['YearBuilt'] >= ano_range[0]) &
        (df_filtered['YearBuilt'] <= ano_range[1])
    ]


# --- ABA 1: VISÃO GERAL ---
if aba == "📊 Visão Geral":
    st.header("📊 Visão Geral dos Dados de Imóveis")
    st.markdown(f"Exibindo *{df_filtered.shape[0]}* de *{df_original.shape[0]}* imóveis após a aplicação dos filtros.")
    
    if not df_filtered.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribuição do Preço de Venda (SalePrice)")
            fig, ax = plt.subplots()
            sns.histplot(df_filtered['SalePrice'], kde=True, ax=ax)
            ax.set_title("Distribuição do Preço de Venda")
            ax.set_xlabel("Preço de Venda ($)")
            ax.set_ylabel("Frequência")
            st.pyplot(fig)

        with col2:
            st.subheader("Preço de Venda vs. Outra Variável")
            numeric_cols_scatter = [col for col in df_filtered.select_dtypes(include=np.number).columns if col not in ['PID', 'Id', 'SalePrice']]
            if numeric_cols_scatter:
                default_scatter_var = 'GrLivArea' if 'GrLivArea' in numeric_cols_scatter else numeric_cols_scatter[0]
                
                x_axis_var = st.selectbox("Escolha a variável para o eixo X:", options=numeric_cols_scatter, index=numeric_cols_scatter.index(default_scatter_var))
                
                fig, ax = plt.subplots()
                sns.scatterplot(data=df_filtered, x=x_axis_var, y='SalePrice', alpha=0.5, ax=ax)
                ax.set_title(f"Preço de Venda vs. {x_axis_var}")
                ax.set_xlabel(x_axis_var)
                ax.set_ylabel("Preço de Venda ($)")
                st.pyplot(fig)
            else:
                st.info("Nenhuma variável numérica disponível para o gráfico de dispersão.")
            
        st.subheader("Amostra dos Dados")
        st.dataframe(df_filtered.head())
    else:
        st.warning("Nenhum dado disponível após a aplicação dos filtros.")


# --- ABA 2: ANOVA ---
elif aba == "📈 Etapa I – ANOVA":
    st.header("📈 Análise de Variância (ANOVA)")
    st.markdown("""
    *Objetivo:* Verificar se existe uma diferença estatisticamente significativa no preço médio de venda (SalePrice) 
    entre diferentes categorias de uma variável escolhida. O usuário deve selecionar de 2 a 3 variáveis sequencialmente para análise.
    """)

    cat_cols_anova = [col for col in df_filtered.select_dtypes(include=['object', 'category']).columns if df_filtered[col].nunique() < 10 and df_filtered[col].nunique() > 1]
    
    if not cat_cols_anova:
        st.warning("Não há variáveis categóricas adequadas para ANOVA após os filtros aplicados (precisa > 1 e < 10 categorias únicas).")
    else:
        default_anova_var = 'BldgType' if 'BldgType' in cat_cols_anova else cat_cols_anova[0]
        var_cat = st.selectbox(
            "*1. Escolha a variável categórica para análise:*",
            options=cat_cols_anova,
            index=cat_cols_anova.index(default_anova_var),
            help="A ANOVA de fator único compara o preço médio entre os grupos de UMA variável por vez."
        )

        if var_cat and not df_filtered.empty:
            st.markdown("---")
            st.subheader(f"Análise de 'SalePrice' por '{var_cat}'")

            st.write("*Visualização da Distribuição:*")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df_filtered, x=var_cat, y='SalePrice', ax=ax)
            ax.set_title(f"Distribuição do Preço de Venda por {var_cat}")
            ax.set_xlabel(var_cat)
            ax.set_ylabel("Preço de Venda ($)")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("2. Verificação dos Pressupostos da ANOVA")

            df_anova_current = df_filtered[[var_cat, 'SalePrice']].dropna()

            if df_anova_current.empty or df_anova_current[var_cat].nunique() < 2:
                st.error(f"Não há dados suficientes ou categorias suficientes para '{var_cat}' após remover NaNs para realizar a ANOVA.")
            else:
                formula = f"SalePrice ~ C({var_cat})"
                try:
                    model_anova = ols(formula, data=df_anova_current).fit()
                    residuals = model_anova.resid
                except Exception as e:
                    st.error(f"Não foi possível ajustar o modelo para os resíduos. Erro: {e}")
                    st.stop()
                
                st.markdown("*a) Normalidade dos Resíduos*")
                if len(residuals) > 2:
                    shapiro_test = stats.shapiro(residuals)
                    p_valor_shapiro = shapiro_test.pvalue
                    if p_valor_shapiro < 0.05:
                        st.warning(f"*Pressuposto violado:* Os resíduos *não* seguem uma distribuição normal (p-valor do teste de Shapiro-Wilk = {p_valor_shapiro:.4f}).")
                    else:
                        st.success(f"*Pressuposto atendido:* Os resíduos parecem seguir uma distribuição normal (p-valor do teste de Shapiro-Wilk = {p_valor_shapiro:.4f}).")
                    
                    fig_qq = sm.qqplot(residuals, line='s', fit=True)
                    plt.title("Gráfico Q-Q dos Resíduos")
                    st.pyplot(fig_qq)
                else:
                    st.warning("Não há resíduos suficientes para o teste de Shapiro-Wilk.")
                    p_valor_shapiro = 0

                st.markdown("*b) Homocedasticidade*")
                groups = [df_anova_current['SalePrice'][df_anova_current[var_cat] == g] for g in df_anova_current[var_cat].unique()]
                groups_for_levene = [g for g in groups if len(g) > 1]
                
                if len(groups_for_levene) > 1:
                    levene_test = stats.levene(*groups_for_levene)
                    p_valor_levene = levene_test.pvalue
                    if p_valor_levene < 0.05:
                        st.warning(f"*Pressuposto violado:* As variâncias *não* são homogêneas entre os grupos (p-valor do teste de Levene = {p_valor_levene:.4f}).")
                    else:
                        st.success(f"*Pressuposto atendido:* As variâncias são homogêneas entre os grupos (p-valor do teste de Levene = {p_valor_levene:.4f}).")
                else:
                    st.warning("Não há grupos suficientes para realizar o teste de Levene.")
                    p_valor_levene = 0

                st.markdown("---")
                st.subheader("3. Resultados do Teste Estatístico e Interpretação")
                
                if p_valor_shapiro >= 0.05 and p_valor_levene >= 0.05:
                    st.info("*Teste Aplicado: ANOVA* (pois os pressupostos foram atendidos).")
                    anova_table = sm.stats.anova_lm(model_anova, typ=2)
                    st.write("Tabela ANOVA:")
                    st.dataframe(anova_table)
                    p_valor_final = anova_table.iloc[0]['PR(>F)']
                    if p_valor_final < 0.05:
                        st.success(f"*Conclusão (ANOVA):* Existe uma diferença estatisticamente significativa nos preços médios de venda entre as diferentes categorias de '{var_cat}' (p-valor = {p_valor_final:.4f}).")
                    else:
                        st.warning(f"*Conclusão (ANOVA):* Não há evidência de uma diferença estatisticamente significativa para '{var_cat}' (p-valor = {p_valor_final:.4f}).")

                else:
                    st.info("*Teste Aplicado: Kruskal-Wallis* (alternativa não paramétrica, pois um ou mais pressupostos da ANOVA foram violados).")
                    if len(groups) > 1 :
                        kruskal_test = stats.kruskal(*groups)
                        p_valor_kruskal = kruskal_test.pvalue
                        st.write(f"*Estatística H (Kruskal-Wallis):* {kruskal_test.statistic:.4f}")
                        st.write(f"*P-valor:* {p_valor_kruskal:.4f}")
                        if p_valor_kruskal < 0.05:
                            st.success(f"*Conclusão (Kruskal-Wallis):* Existe uma diferença estatisticamente significativa nas distribuições de preço para '{var_cat}' (p-valor = {p_valor_kruskal:.4f}).")
                        else:
                            st.warning(f"*Conclusão (Kruskal-Wallis):* Não há evidência de uma diferença significativa para '{var_cat}' (p-valor = {p_valor_kruskal:.4f}).")
                    else:
                        st.error("Não foi possível realizar o teste de Kruskal-Wallis.")
                
                st.markdown(f"*Orientação para Corretores/Investidores:* Se '{var_cat}' mostrou impacto significativo, foque nas categorias de maior valor.")
        elif df_filtered.empty:
            st.warning("Nenhum dado disponível após a aplicação dos filtros para realizar a ANOVA.")


# --- ABA 3: REGRESSÃO LINEAR ---
elif aba == "📉 Etapa II – Regressão":
    st.header("📉 Modelagem Preditiva com Regressão Linear")
    st.markdown("""
    *Objetivo:* Construir um modelo para prever o SalePrice com base em múltiplas características do imóvel.
    """)

    if df_filtered.empty:
        st.warning("Nenhum dado disponível após a aplicação dos filtros para a Regressão Linear.")
    else:
        st.subheader("1. Seleção de Variáveis e Transformação")
        
        num_cols_reg = df_filtered.select_dtypes(include=np.number).columns.tolist()
        num_cols_reg = [col for col in num_cols_reg if col not in ['PID', 'SalePrice', 'Id']]

        cat_cols_reg = [col for col in df_filtered.select_dtypes(include=['object', 'category']).columns if df_filtered[col].nunique() < 20 and df_filtered[col].nunique() > 1]
        
        desired_cat_defaults = ['MSZoning', 'HouseStyle', 'BldgType']
        actual_cat_defaults = [col for col in desired_cat_defaults if col in cat_cols_reg]
        
        desired_num_defaults = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'OverallQual']
        actual_num_defaults = [col for col in desired_num_defaults if col in num_cols_reg]

        col1_reg, col2_reg = st.columns(2)
        with col1_reg:
            vars_cont = st.multiselect(
                "*Escolha variáveis contínuas (numéricas):*",
                options=num_cols_reg,
                default=actual_num_defaults
            )
        with col2_reg:
            vars_cat_reg = st.multiselect(
                "*Escolha variáveis categóricas:*",
                options=cat_cols_reg,
                default=actual_cat_defaults
            )
            
        log_transform = st.checkbox("Aplicar transformação logarítmica em SalePrice e nas variáveis contínuas? (Modelo Log-Log)", value=True)

        if len(vars_cont) >= 1 and len(vars_cat_reg) >= 1:
            st.markdown("---")

            st.subheader("2. Análise Visual da Linearidade")
            st.markdown("Abaixo estão os gráficos de dispersão para cada variável contínua selecionada versus o SalePrice. Isso ajuda a verificar visualmente a premissa de linearidade.")
            
            num_plots = len(vars_cont)
            cols_per_row = 3
            plot_cols = st.columns(cols_per_row)
            
            for i, var in enumerate(vars_cont):
                with plot_cols[i % cols_per_row]:
                    fig_reg, ax_reg = plt.subplots()
                    sns.regplot(data=df_filtered, x=var, y='SalePrice', ax=ax_reg, line_kws={"color": "red"}, scatter_kws={'alpha': 0.3})
                    ax_reg.set_title(f"SalePrice vs. {var}", fontsize=10)
                    ax_reg.set_xlabel(var, fontsize=8)
                    ax_reg.set_ylabel("SalePrice", fontsize=8)
                    st.pyplot(fig_reg)
            
            st.markdown("---")
            st.subheader("3. Ajuste do Modelo e Resultados")
            
            # Preparação dos dados
            cols_for_model = ['SalePrice'] + vars_cont + vars_cat_reg
            df_model = df_filtered[cols_for_model].copy().dropna()
            
            y = df_model['SalePrice']
            X_vars = df_model[vars_cont + vars_cat_reg]

            if log_transform:
                y = np.log1p(y)
                for col in vars_cont:
                    if pd.api.types.is_numeric_dtype(X_vars[col]):
                        if (X_vars[col] >= 0).all():
                            X_vars[col] = np.log1p(X_vars[col])
                        else:
                            st.warning(f"Variável {col} não transformada com log1p pois contém valores negativos.")

            X_vars = pd.get_dummies(X_vars, columns=vars_cat_reg, drop_first=True, dtype=float)
            X_vars = sm.add_constant(X_vars)

            # ===================== INÍCIO DA CORREÇÃO (VERIFICAÇÃO DE DADOS) =====================
            num_observations = X_vars.shape[0]
            num_predictors = X_vars.shape[1]

            if num_observations <= num_predictors:
                st.error(f"Dados insuficientes para o modelo. Após o preparo, há {num_observations} linhas e {num_predictors} preditores. O número de linhas deve ser maior que o número de preditores. Tente usar menos variáveis ou filtros menos restritivos.")
            else:
                try:
                    model_reg = sm.OLS(y, X_vars).fit()
                    
                    st.write("*Resumo do Modelo (Statsmodels OLS):*")
                    st.text(model_reg.summary())
                
                    st.markdown("---")
                    st.subheader("4. Métricas de Desempenho do Modelo")
                    y_pred = model_reg.predict(X_vars)
                    
                    if log_transform:
                        y_true_orig = np.expm1(y)
                        y_pred_orig = np.expm1(y_pred)
                        y_pred_orig = np.nan_to_num(y_pred_orig, nan=np.nanmedian(y_pred_orig), posinf=np.nanmax(y_true_orig[np.isfinite(y_true_orig)]))
                    else:
                        y_true_orig = y
                        y_pred_orig = y_pred

                    r2_adj = model_reg.rsquared_adj
                    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
                    mae = mean_absolute_error(y_true_orig, y_pred_orig)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric(label="R² Ajustado", value=f"{r2_adj:.4f}")
                    m2.metric(label="RMSE (Erro Médio Quadrático)", value=f"${rmse:,.2f}")
                    m3.metric(label="MAE (Erro Médio Absoluto)", value=f"${mae:,.2f}")
                    st.markdown(f"*Discussão do Ajuste:* O modelo explica aproximadamente *{r2_adj:.1%}* da variância no preço de venda. O MAE indica que, em média, as previsões do modelo (na escala original) erram em *${mae:,.2f}*.")

                    st.markdown("---")
                    st.subheader("5. Diagnóstico dos Pressupostos do Modelo")
                    residuals_reg = model_reg.resid

                    diag1, diag2 = st.columns(2)
                    with diag1:
                        st.markdown("*a) Linearidade e Homocedasticidade (Visual)*")
                        fig_res_fit, ax_res_fit = plt.subplots()
                        sns.scatterplot(x=model_reg.fittedvalues, y=residuals_reg, ax=ax_res_fit, alpha=0.5)
                        ax_res_fit.axhline(0, color='red', linestyle='--')
                        ax_res_fit.set_xlabel("Valores Ajustados")
                        ax_res_fit.set_ylabel("Resíduos")
                        ax_res_fit.set_title("Resíduos vs. Valores Ajustados")
                        st.pyplot(fig_res_fit)
                        st.caption("Ideal: Pontos aleatoriamente dispersos em torno da linha 0.")

                        st.markdown("*b) Normalidade dos Resíduos*")
                        if len(residuals_reg) > 2:
                            shapiro_reg_test = stats.shapiro(residuals_reg)
                            if shapiro_reg_test.pvalue < 0.05:
                                st.warning(f"P-valor (Shapiro-Wilk): {shapiro_reg_test.pvalue:.4f}. Resíduos podem não ser normais.")
                            else:
                                st.success(f"P-valor (Shapiro-Wilk): {shapiro_reg_test.pvalue:.4f}. Resíduos parecem normais.")
                            fig_qq_reg = sm.qqplot(residuals_reg, line='s', fit=True)
                            plt.title("Q-Q Plot dos Resíduos (Regressão)")
                            st.pyplot(fig_qq_reg)
                        else:
                             st.warning("Não há resíduos suficientes para o teste de Shapiro-Wilk.")

                    with diag2:
                        st.markdown("*c) Homocedasticidade (Teste Quantitativo)*")
                        try:
                            bp_test = het_breuschpagan(residuals_reg, model_reg.model.exog)
                            if bp_test[1] < 0.05:
                                st.warning(f"P-valor (Breusch-Pagan): {bp_test[1]:.4f}. Há evidência de heterocedasticidade.")
                            else:
                                st.success(f"P-valor (Breusch-Pagan): {bp_test[1]:.4f}. Não há evidência de heterocedasticidade.")
                        except Exception as e_bp:
                            st.warning(f"Não foi possível rodar o teste de Breusch-Pagan: {e_bp}")
                        
                        st.markdown("*d) Multicolinearidade (VIF)*")
                        X_vif = X_vars.drop('const', axis=1, errors='ignore')
                        if not X_vif.empty:
                            vif_data = pd.DataFrame()
                            vif_data["Variável"] = X_vif.columns
                            vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
                            st.dataframe(vif_data[vif_data['VIF'] > 0].style.apply(
                                lambda x: ['background-color: #FF7F7F' if v > 5 else '' for v in x], subset=['VIF']))
                            st.caption("VIF > 5 pode indicar multicolinearidade.")
                        else:
                            st.info("Nenhuma variável para calcular VIF.")

                    st.markdown("---")
                    st.subheader("6. Interpretação dos Coeficientes e Recomendações Práticas")
                    coef_df = pd.DataFrame({'Coeficiente': model_reg.params, 'p-valor': model_reg.pvalues}).reset_index().rename(columns={'index': 'Variável'})
                    
                    coef_significativos = coef_df[(coef_df['p-valor'] < 0.05) & (coef_df['Variável'] != 'const')]
                    
                    if not coef_significativos.empty:
                        st.markdown("*Recomendações e Insights (baseado em variáveis com p-valor < 0.05):*")
                        for _, row in coef_significativos.iterrows():
                            var, coef_val = row['Variável'], row['Coeficiente']
                            original_cat_var_name = next((cat_var for cat_var in vars_cat_reg if cat_var in var), None)

                            if log_transform:
                                impacto_desc = "aumenta" if coef_val > 0 else "reduz"
                                if original_cat_var_name:
                                    st.markdown(f"• Ser da categoria *'{var.replace(original_cat_var_name + '_', '')}'* *{impacto_desc}* o preço em *{abs(coef_val):.2%}*.")
                                elif var in vars_cont:
                                    st.markdown(f"• Um aumento de 1% em *'{var}'* *{impacto_desc}* o preço em *{abs(coef_val):.2%}*.")
                            else:
                                impacto_desc = "aumenta" if coef_val > 0 else "reduz"
                                if original_cat_var_name:
                                    st.markdown(f"• Ser da categoria *'{var.replace(original_cat_var_name + '_', '')}'* *{impacto_desc}* o preço em *${abs(coef_val):,.0f}*.")
                                elif var in vars_cont:
                                    st.markdown(f"• Um aumento de uma unidade em *'{var}'* *{impacto_desc}* o preço em *${abs(coef_val):,.0f}*.")
                        st.caption("Interpretações são ceteris paribus (mantendo outras variáveis constantes).")
                    else:
                        st.warning("Nenhuma variável selecionada apresentou impacto estatisticamente significativo.")
                
                except Exception as e_reg:
                    st.error(f"Erro ao ajustar o modelo de regressão: {e_reg}")
            # ===================== FIM DA CORREÇÃO (VERIFICAÇÃO DE DADOS) =====================

        else:
            st.warning("Por favor, selecione pelo menos uma variável contínua e uma categórica para a análise de regressão.")


# --- ABA 4: SOBRE O PROJETO ---
elif aba == "📘 Sobre o Projeto":
    st.header("📘 Sobre o Projeto e Autoria")
    st.markdown("""
    Este dashboard interativo foi desenvolvido como parte da disciplina de Sistemas de Informações em Engenharia de Produção, com o objetivo de analisar os fatores que influenciam o preço de imóveis na cidade de Ames, Iowa, utilizando técnicas de ANOVA e Regressão Linear Múltipla.
    
    *Autores:* Pedro Russo e Daniel Vianna
    """)
    
    st.markdown("---")
    st.subheader("📌 Funcionalidades e Requisitos Atendidos")
    st.markdown("""
    - ✔️ *Análise Exploratória e Comparativa com ANOVA (Etapa I)*
    - ✔️ *Modelagem Preditiva com Regressão Linear (Etapa II)*
    - ✔️ *Bônus de Inovação*: Dashboard interativo, filtros, gráficos e interpretações.
    """)