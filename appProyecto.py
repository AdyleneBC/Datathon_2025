#####################################################

#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


######################################################

#Nombre en el navegador
st.set_page_config(page_title="DataAnalyzer", page_icon="🌎", layout="wide")
######################################################

#Definimos la instancia
@st.cache_resource
######################################################

#Creamos la función de carga de datos
def load_data():
    df = pd.read_excel("casos6.xlsx")

###############################################################################

    # Columnas numéricas
    numeric_df = df.select_dtypes(include=['float', 'int'])
    numeric_cols = numeric_df.columns

    # Columnas de texto/categóricas
    text_df = df.select_dtypes(include=['object'])
    text_cols = text_df.columns
    
#####################################################################################################################################
#####################################################################################################################################
    return df, numeric_df, numeric_cols, text_df, text_cols

#Cargo los datos obtenidos de la función "load_data"
df, numeric_df, numeric_cols, text_df, text_cols  = load_data()
#######################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#************CREACIÓN DEL DASHBOARD***********CREACIÓN DEL DASHBOARD************CREACIÓN DEL DASHBOARD*****CREACIÓN DEL DASHBOARD

#LOGO DEL SIDEBAR
#st.sidebar.image("logo_albany.png", caption="Dashboard")
    # FONDO DEGRADADO BACKGROOUND
st.markdown("""
    <style>
    .stApp {
        background: black;
        color: white;
    }

    h3 {
        color: #E63946;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        padding-bottom: 0.5rem;
        border-bottom: 3px solid gray;
        margin-bottom: 1rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

#CREAMOS LOS BOTONES DEL SIDEBAR
# Inicializar estado si no existe
if "submenu" not in st.session_state:
    st.session_state.submenu = "Datos Duros"  # por default

# Botones del submenú
if st.sidebar.button("Datos Duros", use_container_width=True):
    st.session_state.submenu = "Datos Duros"

if st.sidebar.button("Violencias", use_container_width=True):
    st.session_state.submenu = "Violencias"
    
if st.sidebar.button("Analisis", use_container_width=True):
    st.session_state.submenu = "Analisis"

if st.sidebar.button("Propuesta", use_container_width=True):
    st.session_state.submenu = "Propuesta"
    
#BOTÓN DE INICIO/INDEX
if st.session_state.submenu == "Datos Duros":
    st.markdown(
    "<h1 style='text-align: center;'>Data Analyzer</h1>",
    unsafe_allow_html=True
    )

    #st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    
    #MOSTRAMOS LA POBLACIÓN
    col1, col2, col3 = st.columns(3) 
    
    with col1:
        st.markdown("""
            <div style='background-color: pink; padding: 5px; border-radius: 10px; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; height: 105px; margin:5px;'>
                <p style='margin: 0; color: #6c757d; text: 15px; font-weight: bold;'>Población</p>
                <p style='margin: 2px 0; color: white; text-size: 10px; font-weight: bold;'> 1,342,977</p>
                <p style='margin: 2px 0; color: white; text-size: 15px;'>TLAXCALA</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div style='background-color: pink; padding: 5px; border-radius: 10px; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; height: 105px; margin:5px;'>
                <p style='margin: 0; color: #6c757d; text: 15px; font-weight: bold;'>Mujeres</p>
                <p style='margin: 2px 0; color: white; text-size: 10px; font-weight: bold;'>51.6%</p>
                <p style='margin: 2px 0; color: white; text-size: 15px;'>MÉXICO</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div style='background-color: pink; padding: 5px; border-radius: 10px; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; height: 105px; margin:5px;'>
                <p style='margin: 0; color: #6c757d; text: 15px; font-weight: bold;'>Hombres</p>
                <p style='margin: 2px 0; color: white; text-size: 10px; font-weight: bold;'>48.4%</p>
                <p style='margin: 2px 0; color: white; text-size: 15px;'>MÉXICO</p>
            </div>
        """, unsafe_allow_html=True)
    
    
    ############################BARRA INTERACTIVA##############################33
    # ==========================
    import json, requests, unicodedata

    def _norm_str(x: str) -> str:
        x = unicodedata.normalize("NFKD", x or "")
        x = "".join(ch for ch in x if not unicodedata.combining(ch))
        return x.strip().casefold()

    @st.cache_data(show_spinner=False)
    def cargar_geojson_tlax(url: str = "https://raw.githubusercontent.com/angelnmara/geojson/master/Municipios/29_Tlaxcala.json"):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        gj = r.json()

        # Asegurar estructura básica
        if "features" not in gj or not gj["features"]:
            raise RuntimeError("GeoJSON sin 'features'.")

        # Detectar posibles nombres de municipio
        props_keys = list(gj["features"][0]["properties"].keys())
        candidatos_mun = ["NOMGEO", "NOM_MUN", "name", "NOMBRE", "MUN_NAME", "MUNICIPIO", "MUNICIP", "NAME_2"]
        campo_mun = next((k for k in candidatos_mun if k in props_keys), None)
        if not campo_mun:
            raise RuntimeError(f"No se encontró el nombre municipal en propiedades: {props_keys}")

        # Si trae todo el país, filtra al estado Tlaxcala si hay un campo de estado
        candidatos_estado = ["NOM_ENT", "ESTADO", "STATE_NAME", "NAME_1", "ENTIDAD", "NOMBRE_ENT"]
        campo_estado = next((k for k in candidatos_estado if k in props_keys), None)

        if campo_estado:
            feats = [f for f in gj["features"] if _norm_str(f["properties"].get(campo_estado, "")) == "tlaxcala"]
            if feats:  # sólo reemplaza si encontró Tlaxcala
                gj = {"type": "FeatureCollection", "features": feats}

        return gj, campo_mun

    def _norm_txt_series(s: pd.Series) -> pd.Series:
        s = s.astype("string").fillna("").str.strip().str.casefold()
        def strip_accents(x):
            x = unicodedata.normalize("NFKD", x)
            return "".join(ch for ch in x if not unicodedata.combining(ch))
        return s.map(strip_accents)

    # ==========================
    df_mapa = df 

    if "hecho_municipio" not in df_mapa.columns:
        st.warning("No se encontró la columna 'hecho_municipio' en el DataFrame.")
    else:
        df_mapa = df_mapa.copy()
        df_mapa["mun_norm"] = _norm_txt_series(df_mapa["hecho_municipio"])

        # Alias opcionales para empatar nombres abreviados/conocidos
        alias = {
            "ixtacuixtla": "ixtacuixtla de mariano matamoros",
            "contla": "contla de juan cuamatzi",
            "xicohtzinco": "san lorenzo xicohtzinco",
        }
        df_mapa["mun_norm"] = df_mapa["mun_norm"].replace(alias)

        # Conteo de casos por municipio
        conteo = df_mapa.groupby("mun_norm", dropna=False).size().rename("casos").reset_index()

        # Cargar geojson (ahora soporta NAME_2)
        geojson, campo_mun = cargar_geojson_tlax()

        # DataFrame de nombres oficiales desde el GeoJSON
        geo_df = pd.DataFrame({
            campo_mun: [f["properties"][campo_mun] for f in geojson["features"]]
        })
        geo_df["mun_norm"] = _norm_txt_series(geo_df[campo_mun])

        # Unir conteos a geometría
        mun_data = geo_df.merge(conteo, on="mun_norm", how="left").fillna({"casos": 0})
        mun_data["casos"] = mun_data["casos"].astype(int)
        total = int(mun_data["casos"].sum())
        mun_data["porcentaje"] = (mun_data["casos"] / total * 100).round(1) if total > 0 else 0

        # Choropleth con tooltip (muestra total de casos)
        fig_map = px.choropleth(
            mun_data,
            geojson=geojson,
            locations=campo_mun,
            featureidkey=f"properties.{campo_mun}",
            color="casos",
            hover_name=campo_mun,
            hover_data={"casos": ":,d", campo_mun: False, "mun_norm": False},
            color_continuous_scale="Reds",
            title=f"Casos por municipio — Tlaxcala (Total: {total:,})",
            template="plotly_dark",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

        # Depuración opcional
        with st.expander("Ver llaves de propiedades y tabla de conteos"):
            # Muestra las claves disponibles por si quieres elegir otra manualmente
            props_keys = list(geojson["features"][0]["properties"].keys())
            #st.write("Claves en properties():", props_keys)
            st.dataframe(mun_data[[campo_mun, "casos", "porcentaje"]].sort_values("casos", ascending=False), use_container_width=True)
            
  ###############################GRAFICA DE BARRAS#########################
    # Agrupar y contar casos
    conteo = df['victima_sexo'].value_counts().reset_index()
    conteo.columns = ['victima_sexo', 'cantidad']

    #Crear la gráfica interactiva
    fig = px.bar(
        conteo,
        x='victima_sexo',
        y='cantidad',
        color='victima_sexo',
        title='Número de víctimas por sexo',
        text='cantidad',
        color_discrete_sequence=['#f76f8e', '#6fa8dc', '#f4b183']
    )

    # Ajustar diseño
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Sexo',
        yaxis_title='Número de víctimas',
        template='plotly_white'
    )

    # Mostrar
    st.plotly_chart(fig)
    
    #################################3

    # Fechas
    df["hecho_fecha"] = pd.to_datetime(df["hecho_fecha"], errors="coerce", dayfirst=True)

    # Casos desde 2020
    df_2020 = df[df["hecho_fecha"] >= pd.Timestamp(2020, 1, 1)].copy()

    # Serie mensual
    serie_mensual = (
        df_2020.dropna(subset=["hecho_fecha"])
            .set_index("hecho_fecha")
            .assign(caso=1)
            .resample("M")["caso"].sum()        # por mes
            .rename("casos")
            .reset_index()
    )

    # Casos por mes
    fig_mes = px.line(
        serie_mensual, x="hecho_fecha", y="casos",
        markers=True, title="Casos por mes (desde 2020)"
    )
    st.plotly_chart(fig_mes, use_container_width=True)

    # Crecimiento acumulado
    serie_mensual["acumulado"] = serie_mensual["casos"].cumsum()
    fig_acum = px.line(
        serie_mensual, x="hecho_fecha", y="acumulado",
        markers=True, title="Crecimiento acumulado de casos (desde 2020)"
    )
    st.plotly_chart(fig_acum, use_container_width=True)
#######################################################################################################
elif st.session_state.submenu == "Violencias":
    
    st.title("Matriz de correlación de variables")
    
    ######################################INICIO DE TABS##########################
    tab1, tab2, tab3 = st.tabs(["Matriz de Correlación", "Regresión Lineal Simple", "Regresión Logística"])
    with tab1:

        # ========================
        # Selecciona solo columnas numéricas
        # ========================
        num_cols = df.select_dtypes(include=["number"]).columns

        # ========================
        # Calcular matriz de correlación
        # ========================
        corr = df[num_cols].corr()

        # ========================
        # Mostrar tabla en Streamlit
        # ========================
        st.subheader("Tabla de correlación")
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)

        # ========================
        # Mapa de calor (estático)
        # ========================
        st.subheader("Mapa de calor (Seaborn)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="RdYlGn", linewidths=0.5, fmt=".2f", ax=ax)
        st.pyplot(fig)

        # ========================
        # Mapa de calor interactivo (Plotly)
        # ========================
        st.subheader("Mapa de calor interactivo")
        fig_px = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            title="Matriz de correlación (interactiva)",
            aspect="auto"
        )
        st.plotly_chart(fig_px, use_container_width=True)
        
    with tab2:

        ##############################################################################################33
        col_x_global = st.sidebar.selectbox("Variable Independiente (X)", list(set(numeric_cols)), key="global_rls_x", index=0)
        col_y_global = st.sidebar.selectbox("Variable Dependiente (Y)", list(set(numeric_cols)), key="global_rls_y", index=2)
        
        vistaLineal = st.sidebar.multiselect("País", ["Regresión lineal"], default=["Regresión lineal"], key="vistaLineal")
        
        if "Regresión lineal" in vistaLineal:
                    st.subheader("México")

                    X_M = df[[col_x_global]].values.reshape(-1, 1)
                    y_M = df[col_y_global].values.reshape(-1, 1)

                    model_M = LinearRegression()
                    model_M.fit(X_M, y_M)
                    y_pred_M = model_M.predict(X_M)

                    st.write(f"**Coeficiente:** {model_M.coef_[0][0]:.4f}")
                    st.write(f"**Intercepto:** {model_M.intercept_[0]:.4f}")

                    scatter_M = px.scatter(df, x=col_x_global, y=col_y_global, color_discrete_sequence=['green'])
                    scatter_M.add_scatter(x=df[col_x_global], y=y_pred_M.flatten(), mode='lines', name='Línea de Regresión',
                                        line=dict(color='red', width=3))
                    st.plotly_chart(scatter_M, use_container_width=True)
    ####################################################################################   
    
    with tab3:  
                   

            
        
#elif st.session_state.submenu == "Analisis":
         
    #elif st.session_state.submenu == "Propuesta":
        st.write("#### Regresión Logística (México)")

        # ===== 1) Variables dependientes binarias (solo las de TU dataset y que EXISTAN en df) =====
        y_opciones_base = [
            "economica","patrimonial","sexual","fisica","psicologica","vicaria","feminicida","otra_violencia",
            "hechos_victima_delincuencia_organizada","hechos_relacionada_orientacion_identidad",
            "quemadura_corrosion","embarazo","intento_suicida","ideacion_suicida","arma_fuego",
            "victima_pueblo_indigena","victima_discapacidad","victima_adiccion","victima_servicio_medico",
            "hecho_domicilio_victima"
        ]
        y_opciones = [c for c in y_opciones_base if c in df.columns]

        if not y_opciones:
            st.error("No encontré variables binarias esperadas en el DataFrame.")
            st.stop()

        # ===== 2) Selección de Y (dependiente) y X (independientes) =====
        y_col = st.sidebar.selectbox("Variable Dependiente (binaria)", y_opciones, index=0, key="logit_y")

        # Usa tus numéricas para X (ya tienes numeric_cols)
        x_opciones = sorted(list(set(numeric_cols)))  # asegúrate de tener numeric_cols definido antes
        if not x_opciones:
            st.error("No hay columnas numéricas disponibles para X.")
            st.stop()

        x_cols = st.sidebar.multiselect(
            "Variables Independientes (X)",
            x_opciones,
            default=[x_opciones[0]],
            key="logit_x"
        )

        if not x_cols:
            st.warning("Selecciona al menos una X.")
            st.stop()

        # ===== 3) Preparar datos (básico) =====
        base = df.copy()
        # Asegura que Y sea 0/1 (por si quedó como texto)
        base[y_col] = pd.to_numeric(base[y_col], errors="coerce")

        # Asegura numéricas en X
        for c in x_cols:
            base[c] = pd.to_numeric(base[c], errors="coerce")

        base = base.dropna(subset=[y_col] + x_cols)

        if base[y_col].nunique() < 2:
            st.error("La variable dependiente tiene una sola clase con los datos filtrados. Elige otra Y o ajusta datos.")
            st.stop()

        X = base[x_cols].values
        y = base[y_col].values.astype(int)

        # ===== 4) Split estratificado + escalado =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ===== 5) Modelo (básico) =====
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ===== 6) Métricas y salidas =====
        st.write(f"**Precisión:** {accuracy_score(y_test, y_pred)*100:.2f}%")
        st.dataframe(pd.DataFrame({"Variable": x_cols, "Coeficiente": model.coef_[0]}))

        matriz = confusion_matrix(y_test, y_pred)
        fig = ff.create_annotated_heatmap(
            z=matriz,
            annotation_text=matriz.astype(str),
            colorscale="Greens",
            showscale=True
        )
        fig.update_layout(title=f"Matriz de Confusión — Y: {y_col}")
        for ann in fig["layout"]["annotations"]:
            ann["font"] = dict(size=20, color="black")
        st.plotly_chart(fig, use_container_width=True)

        rep = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)
