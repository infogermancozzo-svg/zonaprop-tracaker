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

REGLA VITAL: DEBES RESPONDER ESTRICTAMENTE EN ESPAÑOL (CASTELLANO). NO USES INGLÉS.

Instrucciones para "antiguedad":
- Si el texto dice a estrenar, poné: "A estrenar"
- Si dice que es pozo, poné: "Pozo"
- Si dice en construcción o da fecha de entrega, poné: "En construcción"
- Si dice los años (ej. 10 años, 50 años), poné: "X años"
- Si no dice nada, dejalo vacío: ""

Instrucciones para "resumen":
- 2 o 3 renglones fluidos sobre las características físicas y ventajas (todo en ESPAÑOL).
- Ignorá textos legales, inmobiliarias y matrículas.

Texto original:
{texto}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
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
    columnas_base = [
        "Barrio", "Piso", "Ambientes", "Baños", "Toilettes", 
        "Disposición", "Orientación", "Balcón",
        "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", 
        "USD/m2 Promedio", "Antigüedad", "Link", "Notas Personales", 
        "Descripción Completa", "Historial Precio"
    ]
    if HF_TOKEN and REPO_ID:
        try:
            ruta_local = hf_hub_download(repo_id=REPO_
