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

def format_precio(num):
    try:
        if pd.isna(num) or num == "" or num == 0: return "$0"
        return f"${int(num):,}".replace(',', '.')
    except:
        return "$0"

def parse_num_seguro(valor):
    try:
        if valor is None or str(valor).strip() == "": return 0
        val_str = str(valor).strip()
        val_str = re.sub(r'[.,]\d{2}$', '', val_str)
        numeros = re.sub(r'[^\d]', '', val_str)
        return int(numeros) if numeros else 0
    except:
        return 0

def safe_str(valor):
    # Elimina los "nan" de la base de datos para que la UI quede limpia
    if pd.isna(valor): return ""
    s = str(valor).strip()
    if s.lower() in ["nan", "none", "null"]: return ""
    return s

def extraer_todo_con_ia(texto_crudo):
    if not GROQ_API_KEY or not texto_crudo.strip():
        return {}, "⚠️ Faltan datos o API Key."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""Actuá como un detective y tasador inmobiliario de élite. Te paso el texto en bruto de una publicación web. Extraé la información estructurada en JSON.

¡ATENCIÓN A LOS TEXTOS PEGADOS!: Al extraer datos de la web, a veces el texto se junta. Si ves algo como '36 m² tot.18 m² cub.1 amb.1 baño6 años', separá mentalmente los conceptos (Totales: 36, Cubiertos: 18, Ambientes: 1, Baños: 1, Antigüedad: 6). No te confundas por la falta de espacios.

¡ESFUÉRZATE EN DEDUCIR! Especialmente el PISO y la ANTIGÜEDAD. Buscá pistas ocultas:
- PISO: Si dice "patio", suele ser Planta Baja (PB) o 1er piso. Si tiene "terraza propia" o "vista panorámica libre", suele ser el último piso. Si menciona "escaleras", puede ser un 1er o 2do piso por escalera. Cruzá información.
- ANTIGÜEDAD: Si solo hay renders (imágenes digitales) o habla de plazos, es "En construcción" (indicá fecha estimada si está). Si dice recién terminado, es "A estrenar".

REGLA VITAL: Devolvé ÚNICAMENTE un objeto JSON válido. NO escribas texto afuera de las llaves.

Estructura JSON requerida:
{{
    "barrio": "Barrio específico (ej. Villa Urquiza). No pongas CABA ni Provincia.",
    "direccion": "Dirección exacta o aproximada si la dice. Si no, dejalo vacío.",
    "operacion": "Venta o Alquiler",
    "precio_num": Número entero (ej: 120000). Si no hay precio, 0.,
    "ambientes": Número entero. Si no hay, 0.,
    "dormitorios": Número entero. Si no hay, 0.,
    "m2_totales": Número entero. Asegurate de leer bien si está pegado a otra palabra. Si no hay, 0.,
    "m2_cubiertos": Número entero. Asegurate de leer bien si está pegado. Si no hay, 0.,
    "m2_descubiertos": Número entero (totales menos cubiertos). Si no hay, 0.,
    "banos_toilettes": "Texto breve, ej: '1 Baño, 1 Toilette'. Si no dice, vacío.",
    "piso": "Piso de la unidad (ej: 'PB', '3', 'Último'). Si es imposible saberlo, vacío.",
    "disposicion": "Frente, Contrafrente, Lateral, o vacío.",
    "orientacion": "Norte, Sur, Este, Oeste... o vacío.",
    "balcon_patio": "Texto breve, ej: 'Balcón corrido' o 'Patio'. Si no tiene, vacío.",
    "antiguedad": "Ej: '10 años', 'A estrenar', 'En construcción'. Si no hay datos, vacío.",
    "resumen_ia": "2 renglones destacando los puntos fuertes que no entran en los datos fríos (ej: amenities, luminosidad)."
}}

--- TEXTO COMPLETO EXTRAÍDO ---
{texto_crudo[:8000]}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            temperature=0.1
        )
        
        respuesta = chat_completion.choices[0].message.content.strip()
        
        json_limpio = re.sub(r'^```json\s*', '', respuesta)
        json_limpio = re.sub(r'^```\s*', '', json_limpio)
        json_limpio = re.sub(r'\s*```$', '', json_limpio)
        
        try:
            data = json.loads(json_limpio)
            return data, respuesta
        except Exception as e:
            return {}, f"❌ ERROR PARSEANDO JSON:\n{respuesta}\n\nDetalle técnico: {str(e)}"
            
    except Exception as e:
        return {}, f"❌ ERROR API GROQ: {str(e)}"

def cargar_datos():
    columnas_base = [
        "Link", "Precio (USD)", "Historial Precio", "Notas Personales", "Respuesta Cruda IA", "Descripción Completa",
        "Barrio", "Direccion", "Operacion", "Ambientes", "Dormitorios", 
        "M2 Totales", "M2 Cubiertos", "M2 Descubiertos", "Banos", "Piso", 
        "Disposicion", "Orientacion", "Balcon Patio", "Antiguedad", "Resumen IA"
    ]
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
    
    for col_vieja in ["Título", "Titulo", "Borrar", "Ficha Limpia"]:
        if col_vieja in df.columns: df = df.drop(columns=[col_vieja])
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    for col in columnas_base:
        if col not in df.columns:
            df[col] = ""
            
    df = df[[col for col in columnas_base if col in df.columns]]
            
    for col in ["Precio (USD)", "Ambientes", "Dormitorios", "M2 Totales", "M2 Cubiertos", "M2 Descubiertos"]:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    df.reset_index(drop=True, inplace=True)
    return df

def guardar_datos(df):
    df = df.loc[:, ~df.columns.duplicated()]
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
        
        for element in sopa(["script", "style", "noscript", "nav", "footer"]):
            element.extract()
            
        texto_visible = sopa.get_text(separator='\n', strip=True)
        texto_visible = re.sub(r'\n+', '\n', texto_visible)

        descripcion_limpia = ""
        div_desc = sopa.find(attrs={"data-qa": "posting-description"}) or sopa.find(id=re.compile("description", re.I))
        if div_desc:
            descripcion_limpia = div_desc.get_text(separator="\n").strip()
        if not descripcion_limpia:
            descripcion_limpia = "Ver publicación original."

        ia_data, raw_ia_response = extraer_todo_con_ia(texto_visible[:8000])
        if not ia_data: ia_data = {}

        return {
            "Link": url, 
            "Descripción Completa": descripcion_limpia,
            "Respuesta Cruda IA": raw_ia_response,
            "Precio (USD)": parse_num_seguro(ia_data.get("precio_num", 0)),
            "Barrio": str(ia_data.get("barrio", "")).strip().title(),
            "Direccion": str(ia_data.get("direccion", "")).strip(),
            "Operacion": str(ia_data.get("operacion", "Venta")).strip(),
            "Ambientes": parse_num_seguro(ia_data.get("ambientes", 0)),
            "Dormitorios": parse_num_seguro(ia_data.get("dormitorios", 0)),
            "M2 Totales": parse_num_seguro(ia_data.get("m2_totales", 0)),
            "M2 Cubiertos": parse_num_seguro(ia_data.get("m2_cubiertos", 0)),
            "M2 Descubiertos": parse_num_seguro(ia_data.get("m2_descubiertos", 0)),
            "Banos": str(ia_data.get("banos_toilettes", "")).strip(),
            "Piso": str(ia_data.get("piso", "")).strip(),
            "Disposicion": str(ia_data.get("disposicion", "")).strip().title(),
            "Orientacion": str(ia_data.get("orientacion", "")).strip().title(),
            "Balcon Patio": str(ia_data.get("balcon_patio", "")).strip(),
            "Antiguedad": str(ia_data.get("antiguedad", "")).strip(),
            "Resumen IA": str(ia_data.get("resumen_ia", "")).strip(),
            "Historial Precio": "",
            "Notas Personales": ""
        }
    except Exception:
        return None

# --- UI PRINCIPAL ---
st.title("🏢 Gestor de Inversiones Inmobiliarias")

df = cargar_datos()

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
        with st.spinner("Leyendo página y razonando (IA Detective)..."):
            datos = extraer_datos_web(url_input)
            if datos:
                if datos["Precio (USD)"] > 0:
                    hoy = datetime.now().strftime("%d/%m/%Y")
                    datos["Historial Precio"] = f"{hoy}: {format_precio(datos['Precio (USD)'])}"
                    
                nuevo_registro = pd.DataFrame([datos])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.success("¡Propiedad agregada y Ficha generada con éxito!")
                st.rerun()
            else:
                st.error("No se pudo extraer información del link.")
    else:
        st.warning("Por favor, ingresá un link válido.")

if not df.empty:
    if st.button("🔄 Verificador de Precios (Actualizar toda la base)", use_container_width=True):
        with st.spinner("Verificando precios online y registrando en el historial..."):
            hoy = datetime.now().strftime("%d/%m/%Y")
            propiedades_actualizadas = 0
            
            for idx, row in df.iterrows():
                link = row["Link"]
                if link and str(link).startswith("http"):
                    try:
                        headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"}
                        respuesta = requests.get(link, impersonate="chrome110", headers=headers, timeout=12)
                        if respuesta.status_code == 200:
                            sopa = BeautifulSoup(respuesta.text, 'html.parser')
                            for element in sopa(["script", "style", "noscript", "nav", "footer"]):
                                element.extract()
                            texto_visible = sopa.get_text(separator='\n', strip=True)
                            texto_visible = re.sub(r'\n+', '\n', texto_visible)
                            
                            ia_rapida, _ = extraer_todo_con_ia(texto_visible[:5000])
                            if ia_rapida:
                                precio_nuevo = parse_num_seguro(ia_rapida.get("precio_num", 0))
                                precio_viejo = int(row["Precio (USD)"]) if pd.notna(row["Precio (USD)"]) else 0
                                
                                if precio_nuevo > 0 and precio_nuevo != precio_viejo:
                                    historial_previo = str(row["Historial Precio"]) if pd.notna(row["Historial Precio"]) else ""
                                    registro_hoy = f"{hoy}: {format_precio(precio_nuevo)}"
                                    df.at[idx, "Precio (USD)"] = precio_nuevo
                                    df.at[idx, "Historial Precio"] = registro_hoy if historial_previo == "" else f"{historial_previo} | {registro_hoy}"
                                    propiedades_actualizadas += 1
                    except Exception:
                        pass
            
            guardar_datos(df)
            st.success(f"¡Proceso finalizado! Se actualizaron los precios de {propiedades_actualizadas} inmuebles.")
            st.rerun()

st.divider()
st.subheader(f"🏠 Propiedades en Seguimiento ({len(df)})")

if not df.empty:
    columnas_grid = st.columns(3)
    
    for idx, row in df.iterrows():
        col_actual = columnas_grid[idx % 3]
        
        with col_actual:
            with st.container(border=True):
                # 1. BOTÓN ELIMINAR Y PRECIO
                c_precio, c_borrar = st.columns([5, 1])
                precio_actual = int(row['Precio (USD)']) if pd.notna(row['Precio (USD)']) else 0
                historial_str = str(row.get("Historial Precio", ""))
                primer_precio = precio_actual
                
                if historial_str:
                    primer_registro = historial_str.split("|")[0]
                    match_primer_precio = re.search(r'(?:USD|\$)\s*([\d\.]+)', primer_registro.replace('.', ''))
                    if match_primer_precio:
                        primer_precio = int(match_primer_precio.group(1))

                with c_precio:
                    st.markdown(f"<h2 style='margin:0; padding:0; color:#1E88E5;'>{format_precio(precio_actual)}</h2>", unsafe_allow_html=True)
                with c_borrar:
                    if st.button("🗑️", key=f"btn_del_{idx}", help="Eliminar"):
                        df = df.drop(idx).reset_index(drop=True)
                        guardar_datos(df)
                        st.rerun()

                if 0 < precio_actual < primer_precio:
                    st.markdown("<div style='background:#ffebee; color:#c62828; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold; display:inline-block; margin-top:4px;'>🔥 BAJÓ DE PRECIO</div>", unsafe_allow_html=True)

                # 2. SUBTÍTULO
                barrio = safe_str(row.get('Barrio', ''))
                direccion = safe_str(row.get('Direccion', ''))
                ambientes = int(row.get('Ambientes', 0)) if pd.notna(row.get('Ambientes')) else 0
                
                if not barrio: barrio = "Ubicación a confirmar"
                amb_str = f" • {ambientes} Amb." if ambientes > 0 else ""
                dir_str = f"<br><span style='font-size:13px; font-weight:normal;'>📍 {direccion}</span>" if direccion else ""
                
                st.markdown(f"<div style='margin-top:8px; margin-bottom:12px; color:#333; font-weight:bold; font-size:15px;'>{barrio}{amb_str}{dir_str}</div>", unsafe_allow_html=True)
                
                # 3. GRILLA DE ÍCONOS
                m2_tot = int(row.get('M2 Totales', 0)) if pd.notna(row.get('M2 Totales')) else 0
                m2_cub = int(row.get('M2 Cubiertos', 0)) if pd.notna(row.get('M2 Cubiertos')) else 0
                m2_desc = int(row.get('M2 Descubiertos', 0)) if pd.notna(row.get('M2 Descubiertos')) else 0
                
                sup_parts = []
                if m2_tot > 0: sup_parts.append(f"{m2_tot}m² Tot")
                if m2_cub > 0: sup_parts.append(f"{m2_cub}m² Cub")
                if m2_desc > 0: sup_parts.append(f"{m2_desc}m² Desc")
                sup_str = " | ".join(sup_parts) if sup_parts else "Sin datos"
                
                dormitorios = int(row.get('Dormitorios', 0)) if pd.notna(row.get('Dormitorios')) else 0
                banos = safe_str(row.get('Banos', ''))
                
                dist_parts = []
                if dormitorios > 0: dist_parts.append(f"{dormitorios} Dorm.")
                if banos: dist_parts.append(banos)
                dist_str = " | ".join(dist_parts) if dist_parts else "Sin datos"
                
                piso = safe_str(row.get('Piso', ''))
                disposicion = safe_str(row.get('Disposicion', ''))
                orientacion = safe_str(row.get('Orientacion', ''))
                
                ubic_parts = []
                if piso: ubic_parts.append(f"Piso {piso}")
                if disposicion: ubic_parts.append(disposicion)
                if orientacion: ubic_parts.append(orientacion)
                ubic_str = " | ".join(ubic_parts) if ubic_parts else "Sin datos"

                antiguedad = safe_str(row.get('Antiguedad', ''))
                balcon = safe_str(row.get('Balcon Patio', ''))
                
                ext_parts = []
                if antiguedad: ext_parts.append(antiguedad)
                if balcon: ext_parts.append(balcon)
                ext_str = " | ".join(ext_parts) if ext_parts else "Sin datos"

                grilla_html = f"""
                <div style='font-size: 13px; color: #444; line-height: 1.6; margin-bottom: 12px; background-color: #fcfcfc; padding: 10px; border-radius: 6px; border: 1px solid #eee;'>
                    <div style='margin-bottom: 4px;'>📐 <b>Superficie:</b> {sup_str}</div>
                    <div style='margin-bottom: 4px;'>🛏️ <b>Distribución:</b> {dist_str}</div>
                    <div style='margin-bottom: 4px;'>🧭 <b>Ubicación:</b> {ubic_str}</div>
                    <div>🏗️ <b>Características:</b> {ext_str}</div>
                </div>
                """
                st.markdown(grilla_html, unsafe_allow_html=True)
                
                # 4. RESUMEN DE LA IA
                resumen_ia = safe_str(row.get('Resumen IA', ''))
                if resumen_ia == "":
                    st.warning("⚠️ Faltan datos generados por IA.")
                else:
                    st.markdown(f"<div style='font-size: 12.5px; color: #1e3a5f; background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin-bottom: 12px; border-left: 3px solid #1E88E5; line-height: 1.5;'>✨ {resumen_ia}</div>", unsafe_allow_html=True)

                # 5. NOTAS PERSONALES
                nuevas_notas = st.text_area("Notas", value=safe_str(row.get('Notas Personales')), height=68, key=f"notas_{idx}", label_visibility="collapsed", placeholder="📝 Escribí tus notas personales acá...")
                if nuevas_notas != safe_str(row.get('Notas Personales')):
                    df.at[idx, "Notas Personales"] = nuevas_notas
                    guardar_datos(df)
                    st.rerun()

                # 6. EXPANDERS
                with st.expander("📖 Detalles e Historial"):
                    st.markdown("**📉 Historial de Precios**")
                    if historial_str and historial_str.strip() != "":
                        for item in historial_str.split("|"):
                            item_limpio = item.strip()
                            match = re.search(r'(?:USD|\$)\s*([\d\.]+)', item_limpio)
                            if match:
                                val = int(match.group(1).replace('.', ''))
                                item_limpio = item_limpio.replace(match.group(0), format_precio(val))
                            st.markdown(f"<div style='font-size: 13px;'>• {item_limpio}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size: 13px;'>Sin cambios registrados.</div>", unsafe_allow_html=True)
                        
                    st.markdown("<br><b>📝 Descripción Original</b>", unsafe_allow_html=True)
                    desc_completa = safe_str(row.get('Descripción Completa'))
                    if desc_completa:
                        st.markdown(f"<div style='font-size: 12px; color: #666; max-height: 150px; overflow-y: auto;'>{desc_completa}</div>", unsafe_allow_html=True)
                    else:
                        st.write("No se encontró texto original.")
                        
                with st.expander("🤖 Ver razonamiento de la IA (Debug)"):
                    raw_ia = safe_str(row.get("Respuesta Cruda IA"))
                    if raw_ia == "":
                        raw_ia = "No hay datos de IA guardados. Actualizá la propiedad."
                    st.code(raw_ia, language="json")

                # 7. BOTONES
                col_links, col_acts = st.columns(2)
                with col_links:
                    st.link_button("🔗 Ver Aviso", row['Link'], use_container_width=True)
                with col_acts:
                    if st.button("🔄 Actualizar", key=f"btn_act_{idx}", use_container_width=True):
                        with st.spinner("Leyendo web y aplicando IA Detective..."):
                            link_actual = row["Link"]
                            if link_actual and str(link_actual).startswith("http"):
                                datos_frescos = extraer_datos_web(link_actual)
                                if datos_frescos:
                                    df.at[idx, "Descripción Completa"] = datos_frescos["Descripción Completa"]
                                    df.at[idx, "Respuesta Cruda IA"] = datos_frescos["Respuesta Cruda IA"]
                                    
                                    df.at[idx, "Barrio"] = datos_frescos["Barrio"]
                                    df.at[idx, "Direccion"] = datos_frescos["Direccion"]
                                    df.at[idx, "Operacion"] = datos_frescos["Operacion"]
                                    df.at[idx, "Ambientes"] = datos_frescos["Ambientes"]
                                    df.at[idx, "Dormitorios"] = datos_frescos["Dormitorios"]
                                    df.at[idx, "M2 Totales"] = datos_frescos["M2 Totales"]
                                    df.at[idx, "M2 Cubiertos"] = datos_frescos["M2 Cubiertos"]
                                    df.at[idx, "M2 Descubiertos"] = datos_frescos["M2 Descubiertos"]
                                    df.at[idx, "Banos"] = datos_frescos["Banos"]
                                    df.at[idx, "Piso"] = datos_frescos["Piso"]
                                    df.at[idx, "Disposicion"] = datos_frescos["Disposicion"]
                                    df.at[idx, "Orientacion"] = datos_frescos["Orientacion"]
                                    df.at[idx, "Balcon Patio"] = datos_frescos["Balcon Patio"]
                                    df.at[idx, "Antiguedad"] = datos_frescos["Antiguedad"]
                                    df.at[idx, "Resumen IA"] = datos_frescos["Resumen IA"]
                                    
                                    if int(datos_frescos["Precio (USD)"]) > 0:
                                        precio_nuevo = datos_frescos["Precio (USD)"]
                                        precio_viejo = int(row["Precio (USD)"]) if pd.notna(row["Precio (USD)"]) else 0
                                        
                                        if precio_nuevo > 0 and precio_nuevo != precio_viejo:
                                            hoy = datetime.now().strftime("%d/%m/%Y")
                                            historial_previo = str(row.get("Historial Precio", "")) if pd.notna(row.get("Historial Precio", "")) else ""
                                            registro_hoy = f"{hoy}: {format_precio(precio_nuevo)}"
                                            
                                            df.at[idx, "Precio (USD)"] = precio_nuevo
                                            if historial_previo == "":
                                                df.at[idx, "Historial Precio"] = registro_hoy
                                            else:
                                                df.at[idx, "Historial Precio"] = f"{historial_previo} | {registro_hoy}"
                                        
                                    guardar_datos(df)
                                    st.rerun()
else:
    st.info("No tenés propiedades cargadas. Pegá un link arriba para empezar.")
