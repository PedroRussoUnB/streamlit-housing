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
    df_loaded = pd.read_csv("AmesHousing.csv") # Use um nome diferente para evitar conflito
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

# Certificar que OverallQual existe e tem valores únicos antes de usar no filtro
if 'OverallQual' in df_original.columns and not df_original['OverallQual'].empty:
    overall_qual_options = sorted(df_original['OverallQual'].unique())
    default_overall_qual = overall_qual_options
else:
    overall_qual_options = [0] # Placeholder
    default_overall_qual = [0] # Placeholder

qualidade_geral = st.sidebar.multiselect(
    'Filtre por Qualidade Geral do Imóvel:',
    options=overall_qual_options,
    default=default_overall_qual
)

if 'YearBuilt' in df_original.columns and not df_original['YearBuilt'].empty:
    ano_min_orig, ano_max_orig = int(df_original['YearBuilt'].min()), int(df_original['YearBuilt'].max())
else:
    ano_min_orig, ano_max_orig = 1900, 2020 # Placeholders

ano_range = st.sidebar.slider(
    'Filtre por Ano de Construção:',
    min_value=ano_min_orig,
    max_value=ano_max_orig,
    value=(ano_min_orig, ano_max_orig)
)

# Aplicar filtros ao DataFrame
# Renomeia a variável 'df' aqui para 'df_filtered' para clareza
df_filtered = df_original.copy() # Começa com uma cópia do original
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
            st.subheader("Preço de Venda vs. Área Construída")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_filtered, x='GrLivArea', y='SalePrice', alpha=0.5, ax=ax)
            ax.set_title("Preço de Venda vs. Área de Estar (GrLivArea)")
            ax.set_xlabel("Área de Estar (Pés Quadrados)")
            ax.set_ylabel("Preço de Venda ($)")
            st.pyplot(fig)
            
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

    # Seleção de variáveis
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

            # Visualização: Boxplot
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

            # Remover NaNs para o modelo ANOVA especificamente
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
                
                # a) Pressuposto de Normalidade dos Resíduos (Shapiro-Wilk)
                st.markdown("*a) Normalidade dos Resíduos*")
                if len(residuals) > 2: # Shapiro-Wilk precisa de pelo menos 3 amostras
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
                    p_valor_shapiro = 0 # Assume violado para forçar Kruskal-Wallis

                # b) Pressuposto de Homocedasticidade (Teste de Levene)
                st.markdown("*b) Homocedasticidade*")
                groups = [df_anova_current['SalePrice'][df_anova_current[var_cat] == g] for g in df_anova_current[var_cat].unique()]
                groups_for_levene = [g for g in groups if len(g) > 1] # Levene precisa de grupos com >1 amostra
                
                if len(groups_for_levene) > 1: # Levene precisa de pelo menos 2 grupos
                    levene_test = stats.levene(*groups_for_levene)
                    p_valor_levene = levene_test.pvalue
                    if p_valor_levene < 0.05:
                        st.warning(f"*Pressuposto violado:* As variâncias *não* são homogêneas entre os grupos (p-valor do teste de Levene = {p_valor_levene:.4f}).")
                    else:
                        st.success(f"*Pressuposto atendido:* As variâncias são homogêneas entre os grupos (p-valor do teste de Levene = {p_valor_levene:.4f}).")
                else:
                    st.warning("Não há grupos suficientes para realizar o teste de Levene.")
                    p_valor_levene = 0 # Assume violado

                st.markdown("---")
                st.subheader("3. Resultados do Teste Estatístico e Interpretação")
                
                # Decisão sobre qual teste usar
                if p_valor_shapiro >= 0.05 and p_valor_levene >= 0.05:
                    st.info("*Teste Aplicado: ANOVA* (pois os pressupostos foram atendidos).")
                    anova_table = sm.stats.anova_lm(model_anova, typ=2)
                    st.write("Tabela ANOVA:")
                    st.dataframe(anova_table)
                    p_valor_final = anova_table.iloc[0]['PR(>F)']
                    if p_valor_final < 0.05:
                        st.success(f"*Conclusão (ANOVA):* Existe uma diferença estatisticamente significativa nos preços médios de venda entre as diferentes categorias de '{var_cat}' (p-valor = {p_valor_final:.4f}). Categorias com médias mais altas podem indicar maior valorização.")
                    else:
                        st.warning(f"*Conclusão (ANOVA):* Não há evidência de uma diferença estatisticamente significativa nos preços médios de venda para '{var_cat}' (p-valor = {p_valor_final:.4f}).")

                else:
                    st.info("*Teste Aplicado: Kruskal-Wallis* (alternativa não paramétrica, pois um ou mais pressupostos da ANOVA foram violados).")
                    if len(groups) > 1 : # Kruskal-Wallis precisa de pelo menos 2 grupos
                        kruskal_test = stats.kruskal(*groups)
                        p_valor_kruskal = kruskal_test.pvalue
                        st.write(f"*Estatística H (Kruskal-Wallis):* {kruskal_test.statistic:.4f}")
                        st.write(f"*P-valor:* {p_valor_kruskal:.4f}")
                        if p_valor_kruskal < 0.05:
                            st.success(f"*Conclusão (Kruskal-Wallis):* Existe uma diferença estatisticamente significativa nas distribuições de preço de venda entre as diferentes categorias de '{var_cat}' (p-valor = {p_valor_kruskal:.4f}). Isso sugere que '{var_cat}' influencia o preço.")
                        else:
                            st.warning(f"*Conclusão (Kruskal-Wallis):* Não há evidência de uma diferença significativa nas distribuições de preço para '{var_cat}' (p-valor = {p_valor_kruskal:.4f}).")
                    else:
                        st.error("Não foi possível realizar o teste de Kruskal-Wallis devido ao número insuficiente de grupos.")
                
                st.markdown(f"*Orientação para Corretores/Investidores:* Se '{var_cat}' mostrou impacto significativo, foque nas categorias de maior valor para maximizar retornos ou comissões. Se não, esta característica pode não ser um diferencial de preço primário.")
        elif df_filtered.empty:
            st.warning("Nenhum dado disponível após a aplicação dos filtros para realizar a ANOVA.")


# --- ABA 3: REGRESSÃO LINEAR ---
elif aba == "📉 Etapa II – Regressão":
    st.header("📉 Modelagem Preditiva com Regressão Linear")
    st.markdown("""
    *Objetivo:* Construir um modelo para prever o SalePrice com base em múltiplas características do imóvel (4 a 6 variáveis, com pelo menos 1 contínua e 1 categórica).
    """)

    if df_filtered.empty:
        st.warning("Nenhum dado disponível após a aplicação dos filtros para a Regressão Linear.")
    else:
        st.subheader("1. Seleção de Variáveis e Transformação")
        
        num_cols_reg = df_filtered.select_dtypes(include=np.number).columns.tolist()
        # Remover 'PID' e 'SalePrice' das opções de variáveis explicativas numéricas
        num_cols_reg = [col for col in num_cols_reg if col not in ['PID', 'SalePrice', 'Id']]

        cat_cols_reg = [col for col in df_filtered.select_dtypes(include=['object', 'category']).columns if df_filtered[col].nunique() < 20 and df_filtered[col].nunique() > 1]
        
        # Garantir que as colunas padrão existam e sejam válidas
        desired_cat_defaults = ['MSZoning', 'HouseStyle', 'BldgType']
        actual_cat_defaults = [col for col in desired_cat_defaults if col in cat_cols_reg]
        
        desired_num_defaults = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'OverallQual']
        actual_num_defaults = [col for col in desired_num_defaults if col in num_cols_reg]

        col1_reg, col2_reg = st.columns(2)
        with col1_reg:
            vars_cont = st.multiselect(
                "*Escolha variáveis contínuas (numéricas) (1 ou mais):*",
                options=num_cols_reg,
                default=actual_num_defaults
            )
        with col2_reg:
            vars_cat_reg = st.multiselect( # Renomeado para vars_cat_reg para evitar conflito com a ANOVA
                "*Escolha variáveis categóricas (1 ou mais):*",
                options=cat_cols_reg,
                default=actual_cat_defaults
            )
            
        log_transform = st.checkbox("Aplicar transformação logarítmica em SalePrice e nas variáveis contínuas? (Modelo Log-Log)", value=True)

        if len(vars_cont) >= 1 and len(vars_cat_reg) >= 1 and (len(vars_cont) + len(vars_cat_reg)) >= 4 and (len(vars_cont) + len(vars_cat_reg)) <= 6 :
            st.markdown("---")
            st.subheader("2. Ajuste do Modelo e Resultados")
            
            # Preparação dos dados para o modelo
            cols_for_model = ['SalePrice'] + vars_cont + vars_cat_reg
            df_model = df_filtered[cols_for_model].copy().dropna() # Usar cópia e tratar NaNs

            if df_model.shape[0] < (len(vars_cont) + df_model.nunique().sum() + 1): # Heurística para dados suficientes
                 st.error(f"Dados insuficientes ({df_model.shape[0]} linhas) para o número de preditores após tratar NaNs. Tente outros filtros ou variáveis.")
            else:
                y = df_model['SalePrice']
                X_vars = df_model[vars_cont + vars_cat_reg]

                # Transformação logarítmica
                if log_transform:
                    y = np.log1p(y)
                    for col in vars_cont:
                        if pd.api.types.is_numeric_dtype(X_vars[col]):
                             # Adicionar pequeno valor para evitar log(0) se houver zeros e a coluna for estritamente positiva
                            if (X_vars[col] >= 0).all():
                                X_vars[col] = np.log1p(X_vars[col])
                            else: # Se pode ter negativos, não transformar ou usar outra estratégia
                                st.warning(f"Variável {col} não transformada com log1p pois contém valores negativos.")


                # Variáveis dummy para categóricas
                X_vars = pd.get_dummies(X_vars, columns=vars_cat_reg, drop_first=True, dtype=float)
                X_vars = sm.add_constant(X_vars) # Adicionar intercepto
                
                try:
                    model_reg = sm.OLS(y, X_vars).fit()
                    
                    st.write("*Resumo do Modelo (Statsmodels OLS):*")
                    st.text(model_reg.summary())
                
                    st.markdown("---")
                    st.subheader("3. Métricas de Desempenho do Modelo")
                    y_pred = model_reg.predict(X_vars)
                    
                    # Reverter o log para calcular o erro na escala original se a transformação foi aplicada
                    if log_transform:
                        y_true_orig = np.expm1(y)
                        y_pred_orig = np.expm1(y_pred)
                        # Tratar possíveis NaNs ou Inf em y_pred_orig se expm1 resultar em valores muito grandes
                        y_pred_orig = np.nan_to_num(y_pred_orig, nan=np.nanmedian(y_pred_orig), posinf=np.nanmax(y_true_orig[np.isfinite(y_true_orig)]))

                    else:
                        y_true_orig = y
                        y_pred_orig = y_pred

                    r2_adj = model_reg.rsquared_adj
                    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
                    mae = mean_absolute_error(y_true_orig, y_pred_orig)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric(label="R² Ajustado", value=f"{r2_adj:.4f}")
                    m2.metric(label="RMSE (Erro Médio Quadrático)", value=f"<span class="math-inline">\{rmse\:,\.2f\}"\)
m3\.metric\(label\="MAE \(Erro Médio Absoluto\)", value\=f"</span>{mae:,.2f}")
                    st.markdown(f"*Discussão do Ajuste:* O modelo explica aproximadamente *{r2_adj:.1%}* da variância no preço de venda (transformado, se aplicável). O MAE indica que, em média, as previsões do modelo (na escala original) erram em *<span class="math-inline">\{mae\:,\.2f\}\\*\."\)
st\.markdown\("\-\-\-"\)
st\.subheader\("4\. Diagnóstico dos Pressupostos do Modelo"\)
residuals\_reg \= model\_reg\.resid
diag1, diag2 \= st\.columns\(2\)
with diag1\:
\# a\) Linearidade e Homocedasticidade \(Visual\)
st\.markdown\("\\*a\) Linearidade e Homocedasticidade \(Visual\)\\*"\)
fig\_res\_fit, ax\_res\_fit \= plt\.subplots\(\)
sns\.scatterplot\(x\=model\_reg\.fittedvalues, y\=residuals\_reg, ax\=ax\_res\_fit, alpha\=0\.5\)
ax\_res\_fit\.axhline\(0, color\='red', linestyle\='\-\-'\)
ax\_res\_fit\.set\_xlabel\("Valores Ajustados"\)
ax\_res\_fit\.set\_ylabel\("Resíduos"\)
ax\_res\_fit\.set\_title\("Resíduos vs\. Valores Ajustados"\)
st\.pyplot\(fig\_res\_fit\)
st\.caption\("Ideal\: Pontos aleatoriamente dispersos em torno da linha horizontal em 0, sem padrões claros \(funil, curva\)\."\)
\# b\) Normalidade dos Resíduos \(Shapiro\-Wilk\)
st\.markdown\("\\*b\) Normalidade dos Resíduos\\*"\)
if len\(residuals\_reg\) \> 2\:
shapiro\_reg\_test \= stats\.shapiro\(residuals\_reg\)
if shapiro\_reg\_test\.pvalue < 0\.05\:
st\.warning\(f"P\-valor \(Shapiro\-Wilk\)\: \{shapiro\_reg\_test\.pvalue\:\.4f\}\. Os resíduos podem não ser normais\."\)
else\:
st\.success\(f"P\-valor \(Shapiro\-Wilk\)\: \{shapiro\_reg\_test\.pvalue\:\.4f\}\. Resíduos parecem normais\."\)
fig\_qq\_reg \= sm\.qqplot\(residuals\_reg, line\='s', fit\=True\)
plt\.title\("Q\-Q Plot dos Resíduos \(Regressão\)"\)
st\.pyplot\(fig\_qq\_reg\)
else\:
st\.warning\("Não há resíduos suficientes para o teste de Shapiro\-Wilk na regressão\."\)
with diag2\:
\# c\) Homocedasticidade \(Teste de Breusch\-Pagan\)
st\.markdown\("\\*c\) Homocedasticidade \(Teste Quantitativo\)\\*"\)
try\:
bp\_test \= het\_breuschpagan\(residuals\_reg, model\_reg\.model\.exog\)
if bp\_test\[1\] < 0\.05\: \# p\-valor do teste F
st\.warning\(f"P\-valor \(Breusch\-Pagan\)\: \{bp\_test\[1\]\:\.4f\}\. Há evidência de heterocedasticidade \(variância não constante dos resíduos\)\."\)
else\:
st\.success\(f"P\-valor \(Breusch\-Pagan\)\: \{bp\_test\[1\]\:\.4f\}\. Não há evidência significativa de heterocedasticidade\."\)
except Exception as e\_bp\:
st\.warning\(f"Não foi possível rodar o teste de Breusch\-Pagan\: \{e\_bp\}"\)
\# d\) Multicolinearidade \(VIF\)
st\.markdown\("\\*d\) Multicolinearidade \(VIF\)\\*"\)
X\_vif \= X\_vars\.drop\('const', axis\=1, errors\='ignore'\) \# Remover constante para VIF
if not X\_vif\.empty\:
vif\_data \= pd\.DataFrame\(\)
vif\_data\["Variável"\] \= X\_vif\.columns
vif\_data\["VIF"\] \= \[variance\_inflation\_factor\(X\_vif\.values, i\) for i in range\(X\_vif\.shape\[1\]\)\]
st\.dataframe\(vif\_data\[vif\_data\['VIF'\] \> 0\]\.style\.apply\( \# Mostrar apenas VIFs positivos
lambda x\: \['background\-color\: \#FF7F7F' if v \> 5 else '' for v in x\], subset\=\['VIF'\]\)\)
st\.caption\("VIF \> 5 pode indicar multicolinearidade\. VIF \> 10 é geralmente problemático\. Considere remover variáveis com VIF alto se os pressupostos forem afetados\."\)
else\:
st\.info\("Nenhuma variável para calcular VIF \(após remover constante\)\."\)
st\.markdown\("\-\-\-"\)
st\.subheader\("5\. Interpretação dos Coeficientes e Recomendações Práticas"\)
coef\_df \= pd\.DataFrame\(\{
'Coeficiente'\: model\_reg\.params,
'Erro Padrão'\: model\_reg\.bse,
'p\-valor'\: model\_reg\.pvalues
\}\)\.reset\_index\(\)\.rename\(columns\=\{'index'\: 'Variável'\}\)
st\.write\("Coeficientes do Modelo\:"\)
st\.dataframe\(coef\_df\)
coef\_significativos \= coef\_df\[\(coef\_df\['p\-valor'\] < 0\.05\) & \(coef\_df\['Variável'\] \!\= 'const'\)\]
if not coef\_significativos\.empty\:
st\.markdown\("\\*Recomendações e Insights \(baseado em variáveis com p\-valor < 0\.05\)\:\\*"\)
for \_, row in coef\_significativos\.iterrows\(\)\:
var, coef\_val \= row\['Variável'\], row\['Coeficiente'\]
\# Ajustar a interpretação para dummies \(se o nome da variável contém o nome de uma das vars\_cat\_reg originais\)
original\_cat\_var\_name \= next\(\(cat\_var for cat\_var in vars\_cat\_reg if cat\_var in var\), None\)
if log\_transform\:
impacto\_desc \= "aumenta" if coef\_val \> 0 else "reduz"
if original\_cat\_var\_name\: \# É uma dummy de uma variável categórica original
st\.markdown\(f"• Ser da categoria \\'\{var\.replace\(original\cat\_var\_name \+ '\', ''\)\}'\\ \(da variável '\{original\_cat\_var\_name\}'\), em comparação com a categoria base, \\\{impacto\_desc\}\\ o preço do imóvel em aproximadamente \\\{abs\(coef\_val\)\:\.2%\}\\, mantendo outras variáveis constantes\."\)
elif var in vars\_cont\: \# É uma variável contínua transformada
st\.markdown\(f"• Um aumento de 1% em \\'\{var\}'\\, mantendo outras variáveis constantes, \\\{impacto\_desc\}\\ o preço do imóvel em aproximadamente \\\{abs\(coef\_val\)\:\.2%\}\\\."\)
else\: \# Caso geral \(pode ser uma dummy não pega pela lógica acima\)
st\.markdown\(f"• \\\{var\}\\\: Impacto de elasticidade de \\\{coef\_val\:\.2f\}\\\. Um aumento de 1% está associado a uma mudança de \{coef\_val\:\.2%\} no preço\."\)
else\: \# Modelo linear \(não log\-log\)
impacto\_desc \= "aumenta" if coef\_val \> 0 else "reduz"
if original\_cat\_var\_name\:
st\.markdown\(f"• Ser da categoria \\'\{var\.replace\(original\cat\_var\_name \+ '\', ''\)\}'\\ \(da variável '\{original\_cat\_var\_name\}'\), em comparação com a categoria base, \\\{impacto\_desc\}\\ o preço do imóvel em \\</span>{abs(coef_val):,.0f}**, mantendo outras variáveis constantes.")
                                elif var in vars_cont:
                                    st.markdown(f"• Um aumento de uma unidade em *'{var}', mantendo outras variáveis constantes, *{impacto_desc}* o preço do imóvel em *<span class="math-inline">\{abs\(coef\_val\)\:,\.0f\}\\\."\)
else\:
st\.markdown\(f"• \\\{var\}\\\: Impacto de \\</span>{coef_val:,.0f}** no preço para cada unidade de aumento.")
                        st.caption("Interpretações de variáveis dummy são relativas à categoria base omitida. A transformação logarítmica permite interpretação percentual.")
                    else:
                        st.warning("Nenhuma variável selecionada apresentou impacto estatisticamente significativo no preço (com p-valor < 0.05).")
                
                except Exception as e_reg:
                    st.error(f"Erro ao ajustar o modelo de regressão ou em seus diagnósticos: {e_reg}")
        elif not ((len(vars_cont) + len(vars_cat_reg)) >= 4 and (len(vars_cont) + len(vars_cat_reg)) <= 6):
             st.warning("Por favor, selecione um total de 4 a 6 variáveis explicativas (entre contínuas e categóricas).")
        else:
            st.warning("Por favor, selecione pelo menos uma variável contínua e uma categórica para a análise de regressão.")


# --- ABA 4: SOBRE O PROJETO ---
elif aba == "📘 Sobre o Projeto":
    st.header("📘 Sobre o Projeto e Autoria")
    st.markdown("""
    Este dashboard interativo foi desenvolvido como parte da disciplina de Estatística Aplicada, com o objetivo de analisar os fatores que influenciam o preço de imóveis na cidade de Ames, Iowa, utilizando técnicas de ANOVA e Regressão Linear Múltipla.
    O projeto permite a exploração dinâmica dos dados, verificação de pressupostos estatísticos e geração de insights práticos para o mercado imobiliário.
    
    *Autores:* Pedro Russo e Daniel Vianna
    """)
    
    st.markdown("---")
    st.subheader("📌 Funcionalidades e Requisitos Atendidos")
    st.markdown("""
    - ✔️ *Análise Exploratória e Comparativa com ANOVA (Etapa I)*:
        - Seleção de 2-3 variáveis categóricas (sequencialmente).
        - Aplicação de ANOVA.
        - Verificação de pressupostos: Normalidade dos Resíduos (Shapiro-Wilk, Gráfico Q-Q) e Homocedasticidade (Teste de Levene).
        - Abordagem robusta (Kruskal-Wallis) se pressupostos não atendidos.
        - Interpretação dos resultados para tomada de decisão.
    - ✔️ *Modelagem Preditiva com Regressão Linear (Etapa II)*:
        - Seleção de 4-6 variáveis explicativas (≥1 contínua, ≥1 categórica com dummies).
        - Regressão Linear Múltipla (sem interações).
        - Opção de transformação logarítmica (modelo log-log) na dependente e contínuas.
        - Diagnóstico dos pressupostos: Linearidade (visual com resíduos vs. ajustados), Normalidade (Shapiro-Wilk, Q-Q plot), Homocedasticidade (Breusch-Pagan, visual), Multicolinearidade (VIF).
        - Métricas de desempenho: R² Ajustado, RMSE, MAE, com discussão.
        - Interpretação detalhada dos coeficientes (incluindo contexto log-log e significância).
        - Recomendações práticas baseadas no modelo.
    - ✔️ *Bônus de Inovação*:
        - Dashboard interativo em Streamlit.
        - Visualização de dados.
        - Execução dinâmica de análises com filtros.
        - Exibição de gráficos, diagnósticos e interpretações diretamente no app.
    """)