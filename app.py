import streamlit as st
import pandas as pd
from curl_cffi import requests
fromimport streamlit as st
import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup
import json
import re
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(page_title="Gestor de Inversiones Inmobiliarias", page_icon="🏢", layout="wide")

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = st.secrets.get("DATASET_REPO", "")
ARCHIVO_CSV = "Avisos propiedades en venta.csv"

def cargar_datos():
    columnas_base = ["Título", "Barrio", "Piso", "Ambientes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", "USD/m2 Promedio", "Link", "Notas Personales", "Historial Precio"]
    if HF_TOKEN and REPO_ID:
        try:
            ruta_local = hf_hub_download(repo_id=REPO_ID, filename=ARCHIVO_CSV, repo_type="dataset", token=HF_TOKEN)
            df = pd.read_csv(ruta_local)
        except:
            df = pd.DataFrame(columns=columnas_base)
    else:
        if os.path.exists(ARCHIVO_CSV):
            df = pd.read_csv(ARCHIVO_CSV)
        else:
            df = pd.DataFrame(columns=columnas_base)
    
    # Asegurar que todas las columnas existan y no haya duplicados
    df = df.loc[:, ~df.columns.duplicated()]
    for col in columnas_base:
        if col not in df.columns:
            df[col] = ""
            
    for col in ["Barrio", "Piso", "Notas Personales", "Historial Precio", "Título", "Link"]:
        df[col] = df[col].astype(object).fillna("")
        
    for col in ["Precio (USD)", "USD/m2 Promedio", "Ambientes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados"]:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

def guardar_datos(df):
    df = df.loc[:, ~df.columns.duplicated()]
    if "M2 Totales" in df.columns and "M2 Cubiertos" in df.columns and "Precio (USD)" in df.columns:
        m2_descubiertos = df["M2 Totales"] - df["M2 Cubiertos"]
        m2_descubiertos = m2_descubiertos.apply(lambda x: x if x > 0 else 0)
        df["M2 Ponderados"] = df["M2 Cubiertos"] + (m2_descubiertos * 0.5)
        df["USD/m2 Promedio"] = df.apply(lambda row: round(row["Precio (USD)"] / row["M2 Ponderados"]) if row["M2 Ponderados"] > 0 and row["Precio (USD)"] > 0 else 0, axis=1)

    df.to_csv(ARCHIVO_CSV, index=False)
    if HF_TOKEN and REPO_ID:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=ARCHIVO_CSV,
                path_in_repo=ARCHIVO_CSV,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
        except Exception as e:
            st.error(f"Error al sincronizar con Hugging Face: {e}")

def extraer_datos_web(url):
    try:
        respuesta = requests.get(url, impersonate="chrome", timeout=12)
        if respuesta.status_code != 200: 
            return None
            
        sopa = BeautifulSoup(respuesta.text, 'html.parser')
        
        titulo_texto = "Propiedad Zonaprop"
        precio = 0
        m2_tot = 0
        m2_cub = 0
        ambientes = 0
        barrio = ""
        piso = ""

        next_data_tag = sopa.find("script", id="__NEXT_DATA__")
        if next_data_tag:
            try:
                data_json = json.loads(next_data_tag.string)
                props = data_json.get("props", {}).get("pageProps", {}).get("posting", {})
                
                if props:
                    titulo_texto = props.get("title", titulo_texto)
                    precio_val = props.get("priceOperations", [{}])
                    if precio_val:
                        precios_list = precio_val[0].get("prices", [])
                        if precios_list:
                            precio = int(precios_list[0].get("amount", 0))
                    
                    features = props.get("mainFeatures", [])
                    for feat in features:
                        f_text = str(feat).upper()
                        if "M²" in f_text or "METROS" in f_text:
                            m2_match = re.search(r'(\d+)', f_text)
                            if m2_match:
                                val = int(m2_match.group(1))
                                if "CUB" in f_text: m2_cub = val
                                elif "TOT" in f_text: m2_tot = val
                                elif m2_tot == 0: m2_tot = val
                        if "AMB" in f_text:
                            amb_match = re.search(r'(\d+)', f_text)
                            if amb_match: ambientes = int(amb_match.group(1))

                    location = props.get("location", {})
                    barrio = location.get("parent", {}).get("name", "") or location.get("name", "")
            except Exception as e:
                pass

        texto_completo = f"{titulo_texto} {sopa.get_text(separator=' ')}".upper()
        
        if precio == 0:
            precio_match = re.search(r'(?:USD|U\$S|US\$)\s*([\d\.]+)', texto_completo)
            if precio_match: precio = int(precio_match.group(1).replace('.', ''))

        if m2_tot == 0:
            m2_t_match = re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*TOT', texto_completo)
            if m2_t_match: m2_tot = int(m2_t_match.group(1))
            else:
                m2_gen = re.search(r'(\d+)\s*(?:M2|M²|METROS)', texto_completo)
                if m2_gen: m2_tot = int(m2_gen.group(1))

        if m2_cub == 0:
            m2_c_match = re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*CUB', texto_completo)
            if m2_c_match: m2_cub = int(m2_c_match.group(1))
            else: m2_cub = m2_tot

        if m2_cub > m2_tot: m2_cub = m2_tot
        if m2_tot > 0 and m2_cub == 0: m2_cub = m2_tot

        # Detección exhaustiva de piso
        if re.search(r'\b(?:PB|PLANTA\s*BAJA)\b', texto_completo):
            piso = "PB"
        else:
            piso_match = re.search(r'PISO\s*(\d+)', texto_completo)
            if not piso_match: 
                piso_match = re.search(r'(\d+)\s*(?:ER|RO|TO|MO|VO|NO|°|ER\.)?\s*PISO', texto_completo)
            if piso_match: 
                piso = str(piso_match.group(1))
            else:
                mapa_pisos = {"PRIMER": "1", "SEGUNDO": "2", "TERCER": "3", "CUARTO": "4", "QUINTO": "5", "SEXTO": "6", "SEPTIMO": "7", "SÉPTIMO": "7", "OCTAVO": "8", "NOVENO": "9", "DECIMO": "10", "DÉCIMO": "10"}
                for k, v in mapa_pisos.items():
                    if f"{k} PISO" in texto_completo or f"{k}°" in texto_completo:
                        piso = v
                        break

        m2_descubiertos = m2_tot - m2_cub if m2_tot > m2_cub else 0
        m2_pond = m2_cub + (m2_descubiertos * 0.5)
        usd_m2 = round(precio / m2_pond) if m2_pond > 0 and precio > 0 else 0

        if ambientes == 0:
            if "monoambiente" in url.lower(): ambientes = 1
            else:
                amb_match = re.search(r'(\d+)\s*AMB', texto_completo)
                if amb_match: ambientes = int(amb_match.group(1))

        if not barrio:
            barrios_caba = ["VILLA URQUIZA", "BELGRANO", "PALERMO", "CABALLITO", "RECOLETA", "NUÑEZ", "SAAVEDRA", "COGHLAN", "VILLA CRESPO", "ALMAGRO", "COLEGIALES", "CHACARITA", "DEVOTO", "VILLA DEL PARQUE"]
            for b in barrios_caba:
                if b in texto_completo or b.replace(" ", "-") in url.lower():
                    barrio = b.title()
                    break
        
        return {
            "Título": titulo_texto, "Ambientes": ambientes, "Barrio": barrio if barrio else "CABA", "Piso": piso, 
            "M2 Totales": m2_tot, "M2 Cubiertos": m2_cub, "M2 Ponderados": m2_pond, 
            "Precio (USD)": precio, "USD/m2 Promedio": usd_m2, "Link": url, "Notas Personales": "", "Historial Precio": ""
        }
    except Exception as e:
        return None

st.title("🏢 Gestor de Inversiones Inmobiliarias")

df = cargar_datos()

# Inicializar estado para el input de texto y limpiarlo correctamente
if "url_input" not in st.session_state:
    st.session_state["url_input"] = ""

col_input, col_btn = st.columns([4, 1])
with col_input:
    url_input = st.text_input("Link de Zonaprop", placeholder="Pegá el link acá...", key="url_input")

with col_btn:
    st.write("") # Espaciador visual
    st.write("")
    btn_agregar = st.button("Agregar Propiedad", type="primary")

if btn_agregar:
    if url_input:
        with st.spinner("Extrayendo datos del aviso..."):
            datos = extraer_datos_web(url_input)
            if datos:
                nuevo_registro = pd.DataFrame([datos])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.success("¡Propiedad agregada con éxito!")
                st.session_state["url_input"] = "" # Limpiar input
                st.rerun()
            else:
                st.error("No se pudo extraer información del link.")
    else:
        st.warning("Por favor, ingresá un link válido.")

# Botón de Actualización Automática de Precios
if not df.empty:
    if st.button("🔄 Actualizar Precios Automáticamente"):
        with st.spinner("Verificando cambios de precios online..."):
            hoy = datetime.now().strftime("%Y-%m-%d")
            cambios_detectados = 0
            for idx, row in df.iterrows():
                link = row["Link"]
                precio_viejo = row["Precio (USD)"]
                if link and str(link).startswith("http"):
                    datos_nuevos = extraer_datos_web(link)
                    if datos_nuevos and datos_nuevos["Precio (USD)"] > 0:
                        precio_nuevo = datos_nuevos["Precio (USD)"]
                        if precio_nuevo != precio_viejo:
                            cambios_detectados += 1
                            historial_previo = str(row["Historial Precio"]) if pd.notna(row["Historial Precio"]) and row["Historial Precio"] != "" else ""
                            nuevo_cambio = f"{hoy}: USD {precio_viejo} ➔ USD {precio_nuevo}"
                            df.at[idx, "Historial Precio"] = f"{historial_previo} | {nuevo_cambio}".strip(" | ")
                            df.at[idx, "Precio (USD)"] = precio_nuevo
            guardar_datos(df)
            st.success(f"¡Proceso finalizado! Se actualizaron los precios de {cambios_detectados} inmuebles.")
            st.rerun()

st.subheader("Propiedades Registradas (Editables)")
if not df.empty:
    df_editado = st.data_editor(df, use_container_width=True, key="tabla_props")
    if st.button("Guardar Cambios en la Tabla"):
        guardar_datos(df_editado)
        st.success("¡Cambios y recálculos guardados en tu dataset de Hugging Face!")
        st.rerun()
        
    st.markdown("---")
    st.subheader("🗑️ Eliminar Propiedad")
    propiedades_a_borrar = st.multiselect("Seleccioná las propiedades que querés eliminar de la base de datos:", options=df["Título"].tolist())
    if propiedades_a_borrar:
        if st.button("Eliminar Seleccionadas", type="secondary"):
            df = df[~df["Título"].isin(propiedades_a_borrar)]
            guardar_datos(df)
            st.success("Propiedades eliminadas correctamente.")
            st.rerun()
else:
    st.info("No hay propiedades cargadas todavía.") bs4 import BeautifulSoup
import json
import re
import os
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(page_title="Gestor de Inversiones Inmobiliarias", page_icon="🏢", layout="wide")

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = st.secrets.get("DATASET_REPO", "")
ARCHIVO_CSV = "Avisos propiedades en venta.csv"

def cargar_datos():
    if HF_TOKEN and REPO_ID:
        try:
            ruta_local = hf_hub_download(repo_id=REPO_ID, filename=ARCHIVO_CSV, repo_type="dataset", token=HF_TOKEN)
            df = pd.read_csv(ruta_local)
        except:
            df = pd.DataFrame(columns=["Título", "Barrio", "Piso", "Ambientes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", "USD/m2 Promedio", "Link", "Notas Personales", "Historial Precio"])
    else:
        if os.path.exists(ARCHIVO_CSV):
            df = pd.read_csv(ARCHIVO_CSV)
        else:
            df = pd.DataFrame(columns=["Título", "Barrio", "Piso", "Ambientes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", "USD/m2 Promedio", "Link", "Notas Personales", "Historial Precio"])
    
    for col in ["Barrio", "Piso", "Notas Personales", "Historial Precio"]:
        if col not in df.columns: df[col] = ""
        df[col] = df[col].astype(object)
    
    for col in ["Precio (USD)", "USD/m2 Promedio", "Ambientes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def guardar_datos(df):
    # Recalcular valores ponderados y m2 antes de guardar por seguridad
    if "M2 Totales" in df.columns and "M2 Cubiertos" in df.columns and "Precio (USD)" in df.columns:
        m2_descubiertos = df["M2 Totales"] - df["M2 Cubiertos"]
        m2_descubiertos = m2_descubiertos.apply(lambda x: x if x > 0 else 0)
        df["M2 Ponderados"] = df["M2 Cubiertos"] + (m2_descubiertos * 0.5)
        df["USD/m2 Promedio"] = df.apply(lambda row: round(row["Precio (USD)"] / row["M2 Ponderados"]) if row["M2 Ponderados"] > 0 and row["Precio (USD)"] > 0 else 0, axis=1)

    df.to_csv(ARCHIVO_CSV, index=False)
    if HF_TOKEN and REPO_ID:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=ARCHIVO_CSV,
                path_in_repo=ARCHIVO_CSV,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
        except Exception as e:
            st.error(f"Error al sincronizar con Hugging Face: {e}")

def extraer_datos_web(url):
    try:
        respuesta = requests.get(url, impersonate="chrome", timeout=12)
        if respuesta.status_code != 200: 
            return None
            
        sopa = BeautifulSoup(respuesta.text, 'html.parser')
        
        titulo_texto = "Propiedad Zonaprop"
        precio = 0
        m2_tot = 0
        m2_cub = 0
        ambientes = 0
        barrio = ""
        piso = ""

        next_data_tag = sopa.find("script", id="__NEXT_DATA__")
        if next_data_tag:
            try:
                data_json = json.loads(next_data_tag.string)
                props = data_json.get("props", {}).get("pageProps", {}).get("posting", {})
                
                if props:
                    titulo_texto = props.get("title", titulo_texto)
                    precio_val = props.get("priceOperations", [{}])
                    if precio_val:
                        precios_list = precio_val[0].get("prices", [])
                        if precios_list:
                            precio = int(precios_list[0].get("amount", 0))
                    
                    features = props.get("mainFeatures", [])
                    for feat in features:
                        f_text = str(feat).upper()
                        if "M²" in f_text or "METROS" in f_text:
                            m2_match = re.search(r'(\d+)', f_text)
                            if m2_match:
                                val = int(m2_match.group(1))
                                if "CUB" in f_text: m2_cub = val
                                elif "TOT" in f_text: m2_tot = val
                                elif m2_tot == 0: m2_tot = val
                        if "AMB" in f_text:
                            amb_match = re.search(r'(\d+)', f_text)
                            if amb_match: ambientes = int(amb_match.group(1))

                    location = props.get("location", {})
                    barrio = location.get("parent", {}).get("name", "") or location.get("name", "")
            except Exception as e:
                pass

        texto_completo = sopa.get_text(separator=' ').upper()
        
        if precio == 0:
            precio_match = re.search(r'(?:USD|U\$S|US\$)\s*([\d\.]+)', texto_completo)
            if precio_match: precio = int(precio_match.group(1).replace('.', ''))

        if m2_tot == 0:
            m2_t_match = re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*TOT', texto_completo)
            if m2_t_match: m2_tot = int(m2_t_match.group(1))
            else:
                m2_gen = re.search(r'(\d+)\s*(?:M2|M²|METROS)', texto_completo)
                if m2_gen: m2_tot = int(m2_gen.group(1))

        if m2_cub == 0:
            m2_c_match = re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*CUB', texto_completo)
            if m2_c_match: m2_cub = int(m2_c_match.group(1))
            else: m2_cub = m2_tot # Si no especifica cubiertos, asumimos total

        if m2_cub > m2_tot: m2_cub = m2_tot
        if m2_tot > 0 and m2_cub == 0: m2_cub = m2_tot

        # Aplicación estricta de tu fórmula de metros descubiertos (50%)
        m2_descubiertos = m2_tot - m2_cub if m2_tot > m2_cub else 0
        m2_pond = m2_cub + (m2_descubiertos * 0.5)

        usd_m2 = round(precio / m2_pond) if m2_pond > 0 and precio > 0 else 0

        if ambientes == 0:
            if "monoambiente" in url.lower(): ambientes = 1
            else:
                amb_match = re.search(r'(\d+)\s*AMB', texto_completo)
                if amb_match: ambientes = int(amb_match.group(1))

        if not barrio:
            barrios_caba = ["VILLA URQUIZA", "BELGRANO", "PALERMO", "CABALLITO", "RECOLETA", "NUÑEZ", "SAAVEDRA", "COGHLAN", "VILLA CRESPO", "ALMAGRO", "COLEGIALES", "CHACARITA", "DEVOTO", "VILLA DEL PARQUE"]
            for b in barrios_caba:
                if b in texto_completo or b.replace(" ", "-") in url.lower():
                    barrio = b.title()
                    break
        
        return {
            "Titulo": titulo_texto, "Ambientes": ambientes, "Barrio": barrio if barrio else "CABA", "Piso": piso, 
            "M2 Totales": m2_tot, "M2 Cubiertos": m2_cub, "M2 Ponderados": m2_pond, 
            "Precio (USD)": precio, "USD/m2 Promedio": usd_m2, "Link": url
        }
    except Exception as e:
        return None

st.title("🏢 Gestor de Inversiones Inmobiliarias")

df = cargar_datos()

url_input = st.text_input("Link de Zonaprop", placeholder="Pegá el link acá...")
if st.button("Agregar Propiedad", type="primary"):
    if url_input:
        with st.spinner("Extrayendo y aplicando cálculo de $M^2$ ponderados..."):
            datos = extraer_datos_web(url_input)
            if datos:
                nuevo_registro = pd.DataFrame([datos])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.success("¡Propiedad agregada con la fórmula de $M^2$ aplicada correctamente!")
                st.rerun()
    else:
        st.warning("Por favor, ingresá un link válido.")

st.subheader("Propiedades Registradas (Editables)")
if not df.empty:
    # Ahora podés editar metros o precios a mano y se recalculará automáticamente al guardar
    df_editado = st.data_editor(df, use_container_width=True, key="tabla_props")
    if st.button("Guardar Cambios y Recalcular"):
        guardar_datos(df_editado)
        st.success("¡Cambios y recálculo de $M^2$ Ponderados guardados en la nube!")
        st.rerun()
else:
    st.info("No hay propiedades cargadas todavía.")
