import streamlit as st
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
    columnas_base = ["Barrio", "Piso", "Ambientes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", "USD/m2 Promedio", "Link", "Notas Personales", "Historial Precio"]
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
    
    # Limpieza de columnas viejas si las hubiera
    if "Título" in df.columns: df = df.drop(columns=["Título"])
    if "Titulo" in df.columns: df = df.drop(columns=["Titulo"])
    if "Borrar" in df.columns: df = df.drop(columns=["Borrar"])
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    for col in columnas_base:
        if col not in df.columns:
            df[col] = ""
            
    df = df[[col for col in columnas_base if col in df.columns]]
            
    for col in ["Barrio", "Piso", "Notas Personales", "Historial Precio", "Link"]:
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
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        respuesta = requests.get(url, impersonate="chrome110", headers=headers, timeout=15)
        
        if respuesta.status_code != 200: 
            return None
            
        sopa = BeautifulSoup(respuesta.text, 'html.parser')
        
        titulo_texto = "Propiedad Zonaprop"
        descripcion_aviso = ""
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
                    descripcion_aviso = props.get("description", "")
                    
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
                    b_name = location.get("parent", {}).get("name", "") or location.get("name", "")
                    if b_name: barrio = b_name.title()
            except Exception:
                pass

        texto_completo = f"{titulo_texto} {descripcion_aviso} {sopa.get_text(separator=' ')}".upper()
        
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
            "Barrio": barrio if barrio else "CABA", "Piso": piso, "Ambientes": ambientes,
            "M2 Totales": m2_tot, "M2 Cubiertos": m2_cub, "M2 Ponderados": m2_pond, 
            "Precio (USD)": precio, "USD/m2 Promedio": usd_m2, "Link": url, "Notas Personales": "", "Historial Precio": ""
        }
    except Exception:
        return None

# --- UI PRINCIPAL ---
st.title("🏢 Gestor de Inversiones Inmobiliarias")

df = cargar_datos()

# Formulario para agregar propiedades
with st.container(border=True):
    with st.form("form_agregar", clear_on_submit=True):
        st.subheader("➕ Agregar Inmueble")
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            url_input = st.text_input("Link de Zonaprop", placeholder="Pegá el link acá...", label_visibility="collapsed")
        with col_btn:
            btn_agregar = st.form_submit_button("Analizar y Agregar", type="primary", use_container_width=True)

if btn_agregar:
    if url_input:
        with st.spinner("Extrayendo datos del aviso y su descripción..."):
            datos = extraer_datos_web(url_input)
            if datos:
                if datos["Precio (USD)"] > 0:
                    hoy = datetime.now().strftime("%d/%m/%Y")
                    datos["Historial Precio"] = f"{hoy}: USD {datos['Precio (USD)']}"
                    
                nuevo_registro = pd.DataFrame([datos])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.success("¡Propiedad agregada y datos extraídos con éxito!")
                st.rerun()
            else:
                st.error("No se pudo extraer información del link. Zonaprop bloqueó la lectura desde este servidor.")
    else:
        st.warning("Por favor, ingresá un link válido.")

# Botón de actualización de precios general
if not df.empty:
    if st.button("🔄 Verificador de Precios (Actualizar toda la base)", use_container_width=True):
        with st.spinner("Verificando precios online y registrando en el historial..."):
            hoy = datetime.now().strftime("%d/%m/%Y")
            propiedades_actualizadas = 0
            for idx, row in df.iterrows():
                link = row["Link"]
                if link and str(link).startswith("http"):
                    datos_nuevos = extraer_datos_web(link)
                    if datos_nuevos and datos_nuevos["Precio (USD)"] > 0:
                        precio_nuevo = datos_nuevos["Precio (USD)"]
                        historial_previo = str(row["Historial Precio"]) if pd.notna(row["Historial Precio"]) and str(row["Historial Precio"]).strip() != "" else ""
                        
                        registro_hoy = f"{hoy}: USD {precio_nuevo}"
                        
                        if registro_hoy not in historial_previo:
                            df.at[idx, "Precio (USD)"] = precio_nuevo
                            if historial_previo == "":
                                df.at[idx, "Historial Precio"] = registro_hoy
                            else:
                                df.at[idx, "Historial Precio"] = f"{historial_previo} | {registro_hoy}"
                            propiedades_actualizadas += 1
            
            guardar_datos(df)
            st.success(f"¡Proceso finalizado! Se actualizaron los precios de {propiedades_actualizadas} inmuebles.")
            st.rerun()

st.divider()
st.subheader(f"🏠 Propiedades en Seguimiento ({len(df)})")

# --- GRID DE TARJETAS (2 COLUMNAS EN PC) ---
if not df.empty:
    columnas_grid = st.columns(2)
    
    for idx, row in df.iterrows():
        # Alterna entre la columna izquierda (0) y derecha (1)
        col_actual = columnas_grid[idx % 2]
        
        with col_actual:
            with st.container(border=True):
                # Encabezado principal de la tarjeta
                barrio = row['Barrio'] if row['Barrio'] else "Barrio a confirmar"
                ambientes = int(row['Ambientes']) if row['Ambientes'] else "?"
                st.markdown(f"### {barrio} • {ambientes} Amb.")
                
                # Resumen de métricas financieras clave
                precio = int(row['Precio (USD)'])
                m2_pond = row['M2 Ponderados']
                usd_m2 = int(row['USD/m2 Promedio'])
                
                st.markdown(f"**💰 Precio:** USD {precio}")
                st.markdown(f"**📐 Superficie:** {m2_pond} M² Pond. (Tot: {int(row['M2 Totales'])} / Cub: {int(row['M2 Cubiertos'])})")
                st.markdown(f"**📊 Ratio:** USD {usd_m2} / M²")
                
                # Mostrar piso actual si está cargado
                if row['Piso']:
                    st.markdown(f"**🏢 Piso:** {row['Piso']}")
                
                # Notas actuales visibles directo en la tarjeta
                if row['Notas Personales']:
                    st.info(f"**Notas:** {row['Notas Personales']}")

                st.link_button("🔗 Ver Publicación", row['Link'], use_container_width=True)
                
                # Acordeón 1: Historial de precios
                with st.expander("📉 Ver historial de precios"):
                    historial = str(row["Historial Precio"])
                    if historial and historial.strip() != "":
                        for item in historial.split("|"):
                            st.write(f"• {item.strip()}")
                    else:
                        st.write("Sin cambios registrados.")
                
                # Acordeón 2: Edición rápida
                with st.expander("✏️ Editar Piso, Notas o Eliminar"):
                    with st.form(f"form_edicion_{idx}"):
                        nuevo_piso = st.text_input("Piso del inmueble:", value=str(row["Piso"]), key=f"piso_{idx}")
                        nuevas_notas = st.text_area("Notas / Observaciones:", value=str(row["Notas Personales"]), key=f"notas_{idx}")
                        
                        st.markdown("---")
                        borrar = st.checkbox("🗑️ Eliminar definitivamente", key=f"borrar_{idx}")
                        
                        if st.form_submit_button("Guardar Cambios", use_container_width=True):
                            if borrar:
                                df = df.drop(idx)
                                guardar_datos(df)
                                st.rerun()
                            else:
                                df.at[idx, "Piso"] = nuevo_piso
                                df.at[idx, "Notas Personales"] = nuevas_notas
                                guardar_datos(df)
                                st.rerun()
else:
    st.info("No tenés propiedades cargadas. Pegá un link arriba para empezar.")
