import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(page_title="Gestor de Inversiones Inmobiliarias", page_icon="🏢", layout="wide")

# Configuración de credenciales desde los Secrets de Streamlit Cloud
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        respuesta = requests.get(url, headers=headers, timeout=10)
        if respuesta.status_code != 200: return None
            
        sopa = BeautifulSoup(respuesta.text, 'html.parser')
        
        titulo = sopa.find("meta", property="og:title")
        descripcion = sopa.find("meta", property="og:description")
        texto_titulo = titulo["content"] if titulo else "Propiedad"
        texto_desc = descripcion["content"] if descripcion else ""
        texto_completo = f"{texto_titulo} {texto_desc} {sopa.get_text(separator=' ')}".upper()
        
        barrios_caba = ["VILLA URQUIZA", "BELGRANO", "PALERMO", "CABALLITO", "RECOLETA", "NUÑEZ", "NUNEZ", "SAAVEDRA", "COGHLAN", "VILLA CRESPO", "ALMAGRO", "COLEGIALES", "CHACARITA", "DEVOTO", "VILLA DEL PARQUE", "PATERNAL", "FLORES", "MICROCENTRO", "SAN TELMO", "PUERTO MADERO", "RETIRO", "BARRACAS", "BOEDO", "BALVANERA", "LINIERS", "MATADEROS", "PARQUE CHACABUCO", "PARQUE PATRICIOS", "SAN CRISTOBAL", "VILLA LURO", "VILLA PUEYRREDON", "VILLA ORTUZAR"]
        barrio = ""
        for b in barrios_caba:
            if b in texto_completo:
                barrio = b.title()
                break

        piso = ""
        if re.search(r'\b(?:PB|PLANTA\s*BAJA)\b', texto_completo): piso = "PB"
        else:
            piso_match = re.search(r'PISO\s*(\d+)', texto_completo)
            if not piso_match: piso_match = re.search(r'(\d+)\s*(?:ER|RO|TO|MO|VO|NO|°|ER\.)?\s*PISO', texto_completo)
            if piso_match: piso = piso_match.group(1)
            else:
                mapa_pisos = {"PRIMER": "1", "SEGUNDO": "2", "TERCER": "3", "CUARTO": "4", "QUINTO": "5", "SEXTO": "6", "SEPTIMO": "7", "OCTAVO": "8", "NOVENO": "9", "DECIMO": "10", "UNO": "1", "DOS": "2", "TRES": "3", "CUATRO": "4", "CINCO": "5"}
                letras_match = re.search(r'(PRIMER|SEGUNDO|TERCER|CUARTO|QUINTO|SEXTO|SEPTIMO|SÉPTIMO|OCTAVO|NOVENO|DECIMO|DÉCIMO)\s*PISO', texto_completo)
                if letras_match: piso = mapa_pisos.get(letras_match.group(1).replace('É', 'E'), "")

        precio_match = re.search(r'(?:USD|U\$S|US\$)\s*([\d\.]+)', texto_completo)
        precio = int(precio_match.group(1).replace('.', '')) if precio_match else 0
        
        m2_tot = int(re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*TOT', texto_completo).group(1)) if re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*TOT', texto_completo) else (int(re.search(r'(\d+)\s*(?:M2|M²|METROS)', texto_completo).group(1)) if re.search(r'(\d+)\s*(?:M2|M²|METROS)', texto_completo) else 0)
        m2_cub = int(re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*CUB', texto_completo).group(1)) if re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*CUB', texto_completo) else (int(re.search(r'CUBIERTA?S?\D*(\d+)', texto_completo).group(1)) if re.search(r'CUBIERTA?S?\D*(\d+)', texto_completo) else 0)
        ambientes = int(re.search(r'(\d+)\s*AMB', texto_completo).group(1)) if re.search(r'(\d+)\s*AMB', texto_completo) else 0

        if m2_cub > 0 and m2_tot == 0: m2_tot = m2_cub
        if m2_tot > 0 and m2_cub == 0: m2_cub = m2_tot
        if m2_cub > m2_tot: m2_cub = m2_tot

        m2_desc = m2_tot - m2_cub
        m2_pond = m2_cub + (m2_desc * 0.5)
        usd_m2 = round(precio / m2_pond) if m2_pond > 0 and precio > 0 else 0
        
        return {"Titulo": texto_titulo, "Ambientes": ambientes, "Barrio": barrio, "Piso": piso, "M2_Totales": m2_tot, "M2_Cubiertos": m2_cub, "M2_Ponderados": m2_pond, "Precio": precio, "USD_m2": usd_m2, "Link": url}
    except Exception as e:
        return None

st.title("🏢 Gestor de Inversiones Inmobiliarias")

df = cargar_datos()

url_input = st.text_input("Link de Zonaprop", placeholder="Pegá el link acá...")
if st.button("Agregar Propiedad", type="primary"):
    if url_input:
        with st.spinner("Extrayendo datos de la propiedad..."):
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
                st.success("¡Propiedad agregada y guardada en la nube con éxito!")
                st.rerun()
            else:
                st.error("No se pudo extraer información de este link.")
    else:
        st.warning("Por favor, ingresá un link válido.")

st.subheader("Propiedades Registradas")
if not df.empty:
    st.dataframe(df, use_container_width=True)
else:
    st.info("No hay propiedades cargadas todavía.")
