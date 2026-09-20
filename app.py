import streamlit as st
import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup
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
    
    if "Precio (USD)" in df.columns: df["Precio (USD)"] = pd.to_numeric(df["Precio (USD)"], errors='coerce').fillna(0)
    if "USD/m2 Promedio" in df.columns: df["USD/m2 Promedio"] = pd.to_numeric(df["USD/m2 Promedio"], errors='coerce').fillna(0)
    if "Ambientes" in df.columns: df["Ambientes"] = pd.to_numeric(df["Ambientes"], errors='coerce').fillna(0)
    
    return df

def guardar_datos(df):
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

        # Intentar extraer desde el JSON interno de Next.js (__NEXT_DATA__)
        next_data_tag = sopa.find("script", id="__NEXT_DATA__")
        if next_data_tag:
            try:
                data_json = json.loads(next_data_tag.string)
                # Navegar por la estructura típica de Zonaprop
                props = data_json.get("props", {}).get("pageProps", {}).get("posting", {})
                
                if props:
                    titulo_texto = props.get("title", titulo_texto)
                    precio_val = props.get("priceOperations", [{}])
                    if precio_val:
                        precios_list = precio_val[0].get("prices", [])
                        if precios_list:
                            precio = int(precios_list[0].get("amount", 0))
                    
                    # Metros y ambientes
                    features = props.get("mainFeatures", [])
                    for feat in features:
                        f_text = str(feat).upper()
                        if "M²" in f_text or "METROS" in f_text:
                            m2_match = re.search(r'(\d+)', f_text)
                            if m2_match and m2_tot == 0: m2_tot = int(m2_match.group(1))
                        if "AMB" in f_text:
                            amb_match = re.search(r'(\d+)', f_text)
                            if amb_match: ambientes = int(amb_match.group(1))

                    # Ubicación
                    location = props.get("location", {})
                    barrio = location.get("parent", {}).get("name", "") or location.get("name", "")
            except Exception as e:
                pass

        # Respaldo por texto general y URL si el JSON no trajo todo
        texto_completo = sopa.get_text(separator=' ').upper()
        
        if precio == 0:
            precio_match = re.search(r'(?:USD|U\$S|US\$)\s*([\d\.]+)', texto_completo)
            if precio_match: precio = int(precio_match.group(1).replace('.', ''))

        if m2_tot == 0:
            m2_gen = re.search(r'(\d+)\s*(?:M2|M²|METROS)', texto_completo)
            if m2_gen: m2_tot = int(m2_gen.group(1))

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

        m2_cub = m2_tot
        m2_pond = m2_cub if m2_tot > 0 else 30
        usd_m2 = round(precio / m2_pond) if m2_pond > 0 and precio > 0 else 0
        
        return {
            "Titulo": titulo_texto, "Ambientes": ambientes, "Barrio": barrio if barrio else "CABA", "Piso": piso, 
            "M2_Totales": m2_tot, "M2_Cubiertos": m2_cub, "M2_Ponderados": m2_pond, 
            "Precio": precio, "USD_m2": usd_m2, "Link": url
        }
    except Exception as e:
        return None

st.title("🏢 Gestor de Inversiones Inmobiliarias")

df = cargar_datos()

url_input = st.text_input("Link de Zonaprop", placeholder="Pegá el link acá...")
if st.button("Agregar Propiedad", type="primary"):
    if url_input:
        with st.spinner("Extrayendo datos estructurados del aviso..."):
            datos = extraer_datos_web(url_input)
            if datos:
                nuevo_registro = pd.DataFrame([{
                    "Título": datos["Titulo"], "Barrio": datos["Barrio"], "Piso": datos["Piso"], 
                    "Ambientes": datos["Ambientes"], "M2 Totales": datos["M2_Totales"], 
                    "M2 Cubiertos": datos["M2_Cubiertos"], "M2 Ponderados": datos["M2_Ponderados"], 
                    "Precio (USD)": datos["Precio"], "USD/m2 Promedio": datos["USD_m2"], 
                    "Link": datos["Link"], "Notas Personales": "", "Historial Precio": ""
                }])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.success("¡Datos extraídos y guardados con éxito!")
                st.rerun()
    else:
        st.warning("Por favor, ingresá un link válido.")

st.subheader("Propiedades Registradas (Editables)")
if not df.empty:
    df_editado = st.data_editor(df, use_container_width=True, key="tabla_props")
    if st.button("Guardar Cambios en la Tabla"):
        guardar_datos(df_editado)
        st.success("¡Cambios guardados en tu dataset de Hugging Face!")
else:
    st.info("No hay propiedades cargadas todavía.")
