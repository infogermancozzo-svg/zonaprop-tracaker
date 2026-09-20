import streamlit as st
import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup
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
        # Usamos curl_cffi simulando un navegador Chrome para saltar Cloudflare
        respuesta = requests.get(url, impersonate="chrome", timeout=12)
        if respuesta.status_code != 200: 
            return None
            
        sopa = BeautifulSoup(respuesta.text, 'html.parser')
        
        # Extraer metadatos y texto completo de la página
        titulo = sopa.find("meta", property="og:title")
        descripcion = sopa.find("meta", property="og:description")
        texto_titulo = titulo["content"] if titulo else "Propiedad"
        texto_desc = descripcion["content"] if descripcion else ""
        texto_completo = f"{texto_titulo} {texto_desc} {sopa.get_text(separator=' ')}".upper()
        
        # Detección de Barrio CABA
        barrios_caba = ["VILLA URQUIZA", "BELGRANO", "PALERMO", "CABALLITO", "RECOLETA", "NUÑEZ", "NUNEZ", "SAAVEDRA", "COGHLAN", "VILLA CRESPO", "ALMAGRO", "COLEGIALES", "CHACARITA", "DEVOTO", "VILLA DEL PARQUE", "PATERNAL", "FLORES", "MICROCENTRO", "SAN TELMO", "PUERTO MADERO", "RETIRO", "BARRACAS", "BOEDO", "BALVANERA", "LINIERS", "MATADEROS", "PARQUE CHACABUCO", "PARQUE PATRICIOS", "SAN CRISTOBAL", "VILLA LURO", "VILLA PUEYRREDON", "VILLA ORTUZAR"]
        barrio = ""
        for b in barrios_caba:
            if b in texto_completo:
                barrio = b.title()
                break

        # Si el scraping de texto no encuentra el barrio, respaldamos con la URL
        if not barrio:
            url_lower = url.lower()
            for b in barrios_caba:
                if b.replace(" ", "-") in url_lower:
                    barrio = b.title()
                    break

        # Detección de Piso
        piso = ""
        if re.search(r'\b(?:PB|PLANTA\s*BAJA)\b', texto_completo): 
            piso = "PB"
        else:
            piso_match = re.search(r'PISO\s*(\d+)', texto_completo)
            if not piso_match: 
                piso_match = re.search(r'(\d+)\s*(?:ER|RO|TO|MO|VO|NO|°|ER\.)?\s*PISO', texto_completo)
            if piso_match: 
                piso = piso_match.group(1)

        # Precio en USD
        precio_match = re.search(r'(?:USD|U\$S|US\$)\s*([\d\.]+)', texto_completo)
        precio = int(precio_match.group(1).replace('.', '')) if precio_match else 0
        
        # Metros y Ambientes
        m2_tot_match = re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*TOT', texto_completo)
        m2_tot = int(m2_tot_match.group(1)) if m2_tot_match else 0
        
        if m2_tot == 0:
            m2_gen = re.search(r'(\d+)\s*(?:M2|M²|METROS)', texto_completo)
            m2_tot = int(m2_gen.group(1)) if m2_gen else 0

        m2_cub_match = re.search(r'(\d+)\s*(?:M2|M²|METROS)\s*CUB', texto_completo)
        m2_cub = int(m2_cub_match.group(1)) if m2_cub_match else m2_tot

        if m2_cub > m2_tot: m2_cub = m2_tot
        if m2_tot > 0 and m2_cub == 0: m2_cub = m2_tot

        amb_match = re.search(r'(\d+)\s*AMB', texto_completo)
        ambientes = int(amb_match.group(1)) if amb_match else (1 if "monoambiente" in url.lower() else 0)

        m2_desc = m2_tot - m2_cub
        m2_pond = m2_cub + (m2_desc * 0.5) if m2_tot > 0 else 0
        usd_m2 = round(precio / m2_pond) if m2_pond > 0 and precio > 0 else 0
        
        return {
            "Titulo": texto_titulo, "Ambientes": ambientes, "Barrio": barrio, "Piso": piso, 
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
        with st.spinner("Extrayendo datos reales del aviso..."):
            datos = extraer_datos_web(url_input)
            if datos and datos["Precio"] > 0:
                nuevo_registro = pd.DataFrame([{
                    "Título": datos["Titulo"], "Barrio": datos["Barrio"], "Piso": datos["Piso"], 
                    "Ambientes": datos["Ambientes"], "M2 Totales": datos["M2_Totales"], 
                    "M2 Cubiertos": datos["M2_Cubiertos"], "M2 Ponderados": datos["M2_Ponderados"], 
                    "Precio (USD)": datos["Precio"], "USD/m2 Promedio": datos["USD_m2"], 
                    "Link": datos["Link"], "Notas Personales": "", "Historial Precio": ""
                }])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.success("¡Propiedad extraída y guardada con éxito en la nube!")
                st.rerun()
            else:
                st.warning("No se pudo extraer automáticamente el precio. Podés ingresarlo manualmente abajo en la tabla.")
                # Igualmente agregamos la fila vacía para que puedas completarla a mano
                nuevo_registro = pd.DataFrame([{
                    "Título": "Propiedad Zonaprop", "Barrio": "", "Piso": "", 
                    "Ambientes": 0, "M2 Totales": 0, "M2 Cubiertos": 0, "M2 Ponderados": 0, 
                    "Precio (USD)": 0, "USD/m2 Promedio": 0, "Link": url_input, "Notas Personales": "", "Historial Precio": ""
                }])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.rerun()
    else:
        st.warning("Por favor, ingresá un link válido.")

st.subheader("Propiedades Registradas (Editables)")
if not df.empty:
    df_editado = st.data_editor(df, use_container_width=True, key="tabla_props")
    if st.button("Guardar Cambios en la Tabla"):
        guardar_datos(df_editado)
        st.success("¡Cambios y correcciones guardados en tu dataset de Hugging Face!")
else:
    st.info("No hay propiedades cargadas todavía.")
