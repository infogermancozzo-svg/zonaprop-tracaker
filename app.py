import streamlit as st
import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup
import json
import re
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from groq import Groq

st.set_page_config(page_title="Gestor de Inversiones Inmobiliarias", page_icon="🏢", layout="wide")

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = st.secrets.get("DATASET_REPO", "")
ARCHIVO_CSV = "Avisos propiedades en venta.csv"
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

def resumir_con_ia(texto):
    if not GROQ_API_KEY or not texto.strip():
        return {"antiguedad": "", "resumen": ""}
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""Actuá como un tasador inmobiliario. Analizá el texto y devolvé ÚNICAMENTE un objeto JSON válido con las claves "antiguedad" y "resumen". No agregues texto antes ni después del JSON.

Instrucciones para "antiguedad":
- Si el texto dice a estrenar, poné: "A estrenar"
- Si dice que es pozo, poné: "Pozo"
- Si dice en construcción o da fecha de entrega, poné: "En construcción"
- Si dice los años (ej. 10 años, 50 años), poné: "X años"
- Si no dice nada, dejalo vacío: ""

Instrucciones para "resumen":
- 2 o 3 renglones fluidos sobre las características físicas y ventajas.
- Ignorá textos legales, inmobiliarias y matrículas.

Texto original:
{texto}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b", # <-- MODELO SOLICITADO
            temperature=0.1
        )
        
        respuesta = chat_completion.choices[0].message.content.strip()
        
        # Limpiar si la IA agregó comillas de bloque de código Markdown
        respuesta = re.sub(r'^```json\s*', '', respuesta)
        respuesta = re.sub(r'^```\s*', '', respuesta)
        respuesta = re.sub(r'\s*```$', '', respuesta)
        
        try:
            data = json.loads(respuesta)
            return {"antiguedad": data.get("antiguedad", ""), "resumen": data.get("resumen", "")}
        except:
            return {"antiguedad": "", "resumen": respuesta[:150] + "..."}
            
    except Exception as e:
        return {"antiguedad": "", "resumen": f"⚠️ [Error Groq]: {str(e)}"}

def cargar_datos():
    # Agregamos Baños y Toilettes a las columnas base
    columnas_base = ["Barrio", "Piso", "Ambientes", "Baños", "Toilettes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", "USD/m2 Promedio", "Antigüedad", "Link", "Notas Personales", "Descripción Completa", "Historial Precio"]
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
    
    if "Título" in df.columns: df = df.drop(columns=["Título"])
    if "Titulo" in df.columns: df = df.drop(columns=["Titulo"])
    if "Borrar" in df.columns: df = df.drop(columns=["Borrar"])
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    for col in columnas_base:
        if col not in df.columns:
            df[col] = ""
            
    df = df[[col for col in columnas_base if col in df.columns]]
            
    for col in ["Barrio", "Piso", "Antigüedad", "Notas Personales", "Descripción Completa", "Historial Precio", "Link"]:
        df[col] = df[col].astype(object).fillna("")
        
    for col in ["Precio (USD)", "USD/m2 Promedio", "Ambientes", "Baños", "Toilettes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados"]:
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
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8,en;q=0.7",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        respuesta = requests.get(url, impersonate="chrome110", headers=headers, timeout=15)
        if respuesta.status_code != 200: return None
            
        sopa = BeautifulSoup(respuesta.text, 'html.parser')
        
        titulo_texto = "Propiedad Zonaprop"
        descripcion_aviso = ""
        precio = m2_tot = m2_cub = ambientes = banos = toilettes = 0
        barrio = piso = antiguedad_web = ""

        next_data_tag = sopa.find("script", id="__NEXT_DATA__")
        if next_data_tag:
            try:
                data_json = json.loads(next_data_tag.string)
                page_props = data_json.get("props", {}).get("pageProps", {})
                props = page_props.get("posting", {}) or page_props.get("initialPosting", {})
                
                if props:
                    titulo_texto = props.get("title", titulo_texto)
                    descripcion_aviso = props.get("plainDescription", "") or props.get("description", "")
                    
                    precio_val = props.get("priceOperations", [{}])
                    if precio_val:
                        precios_list = precio_val[0].get("prices", [])
