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

def extraer_todo_con_ia(texto_crudo):
    if not GROQ_API_KEY or not texto_crudo.strip():
        return {}, "⚠️ Faltan datos o API Key."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # --- NUEVO PROMPT DE DIAGNÓSTICO PROFUNDO ---
        prompt = f"""Actuá como un analista inmobiliario. Te voy a pasar todo el texto en bruto extraído de un link de Zonaprop.
Tu objetivo es analizarlo profundamente y devolverme ÚNICAMENTE un objeto JSON.

REGLAS VITALES:
- RESPONDER SÓLO CON EL JSON. NADA DE TEXTO EXTRA AFUERA DE LAS LLAVES.
- Los campos numéricos dejamelos en 0 si no los encontrás, pero el campo "resumen" debe ser súper exhaustivo.

Estructura JSON requerida:
{{
    "precio": Número entero (ej. 95000),
    "m2_totales": Número entero,
    "m2_cubiertos": Número entero,
    "ambientes": Número entero,
    "resumen": "REPORTE EXHAUSTIVO: Hacé un resumen detallado destacando obligatoriamente lo siguiente: Barrio, Precio, Metros totales, Metros cubiertos, Metros descubiertos, Cantidad de baños y/o toilettes, si tiene balcón, patio o terraza, tu deducción rápida del piso en que se encuentra, antigüedad, orientación, y si es frente o contrafrente. REGLA DE ORO: Si no podés deducir o encontrar alguna de esta información en el texto, debés explicitarlo claramente (ej: 'No se especifica en qué piso se encuentra' o 'No menciona la antigüedad'). Todo este reporte redactado debe ir en este único campo."
}}

--- TEXTO COMPLETO EXTRAÍDO DEL LINK DEL AVISO ---
{texto_crudo[:8000]}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/GPT-OSS 120B",
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
        "Barrio", "Piso", "Ambientes", "Baños", "Toilettes", 
        "Disposición", "Orientación", "Balcón",
        "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", 
        "USD/m2 Promedio", "Antigüedad", "Link", "Resumen IA", "Notas Personales", 
        "Descripción Completa", "Historial Precio", "Respuesta Cruda IA"
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
    
    if "Título" in df.columns: df = df.drop(columns=["Título"])
    if "Titulo" in df.columns: df = df.drop(columns=["Titulo"])
    if "Borrar" in df.columns: df = df.drop(columns=["Borrar"])
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    for col in columnas_base:
        if col not in df.columns:
            df[col] = ""
            
    df = df[[col for col in columnas_base if col in df.columns]]
            
    for col in ["Barrio", "Piso", "Antigüedad", "Disposición", "Orientación", "Balcón", "Resumen IA", "Notas Personales", "Descripción Completa", "Historial Precio", "Link", "Respuesta Cruda IA"]:
        df[col] = df[col].astype(object).fillna("")
        
    for col in ["Precio (USD)", "USD/m2 Promedio", "Ambientes", "Baños", "Toilettes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados"]:
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

        precio = parse_num_seguro(ia_data.get("precio", 0))
        m2_tot = parse_num_seguro(ia_data.get("m2_totales", 0))
        m2_cub = parse_num_seguro(ia_data.get("m2_cubiertos", 0))
        ambientes = parse_num_seguro(ia_data.get("ambientes", 0))
        
        # Dejamos que la app haga la matemática básica si la IA encontró los metros y el precio
        if m2_cub > m2_tot: m2_cub = m2_tot
        if m2_tot > 0 and m2_cub == 0: m2_cub = m2_tot
        m2_descubiertos = m2_tot - m2_cub if m2_tot > m2_cub else 0
        m2_pond = m2_cub + (m2_descubiertos * 0.5)
        usd_m2 = round(precio / m2_pond) if m2_pond > 0 and precio > 0 else 0

        resumen_ia = str(ia_data.get("resumen", "La IA no devolvió un resumen.")).strip()
        
        # Conservamos las variables vacías para que no se rompan las cajas editables de la interfaz
        return {
            "Barrio": "CABA", "Piso": "no menciona", "Ambientes": ambientes,
            "Baños": 0, "Toilettes": 0,
            "Disposición": "", "Orientación": "", "Balcón": "",
            "M2 Totales": m2_tot, "M2 Cubiertos": m2_cub, "M2 Ponderados": m2_pond, 
            "Precio (USD)": precio, "USD/m2 Promedio": usd_m2, "Antigüedad": "no menciona", 
            "Link": url, "Resumen IA": resumen_ia, "Notas Personales": "", "Descripción Completa": descripcion_limpia, 
            "Historial Precio": "", "Respuesta Cruda IA": raw_ia_response
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
        with st.spinner("Leyendo página y redactando reporte de IA..."):
            datos = extraer_datos_web(url_input)
            if datos:
                if datos["Precio (USD)"] > 0:
                    hoy = datetime.now().strftime("%d/%m/%Y")
                    datos["Historial Precio"] = f"{hoy}: {format_precio(datos['Precio (USD)'])}"
                    
                nuevo_registro = pd.DataFrame([datos])
                df = pd.concat([df, nuevo_registro], ignore_index=True)
                guardar_datos(df)
                st.success("¡Propiedad agregada y datos extraídos con éxito!")
                st.rerun()
            else:
                st.error("No se pudo extraer información del link. Zonaprop bloqueó la lectura desde este servidor.")
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
                                precio_nuevo = parse_num_seguro(ia_rapida.get("precio", 0))
                                precio_viejo = int(row["Precio (USD)"]) if pd.notna(row["Precio (USD)"]) else 0
                                
                                if precio_nuevo > 0 and precio_nuevo != precio_viejo:
                                    historial_previo = str(row["Historial Precio"]) if pd.notna(row["Historial Precio"]) else ""
                                    registro_hoy = f"{hoy}: {format_precio(precio_nuevo)}"
                                    
                                    df.at[idx, "Precio (USD)"] = precio_nuevo
                                    if historial_previo == "":
                                        df.at[idx, "Historial Precio"] = registro_hoy
                                    else:
                                        df.at[idx, "Historial Precio"] = f"{historial_previo} | {registro_hoy}"
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
                precio_actual = int(row['Precio (USD)']) if pd.notna(row['Precio (USD)']) else 0
                historial_str = str(row.get("Historial Precio", ""))
                primer_precio = precio_actual
                
                if historial_str:
                    primer_registro = historial_str.split("|")[0]
                    match_primer_precio = re.search(r'(?:USD|\$)\s*([\d\.]+)', primer_registro.replace('.', ''))
                    if match_primer_precio:
                        primer_precio = int(match_primer_precio.group(1))

                c_titulo, c_borrar = st.columns([5, 1])
                with c_titulo:
                    st.markdown(f"<h3 style='margin:0; padding:0; color:#1E88E5;'>{format_precio(precio_actual)}</h3>", unsafe_allow_html=True)
                with c_borrar:
                    if st.button("🗑️", key=f"btn_del_{idx}", help="Eliminar"):
                        df = df.drop(idx).reset_index(drop=True)
                        guardar_datos(df)
                        st.rerun()
                
                barrio = row['Barrio'] if row['Barrio'] else "Barrio a confirmar"
                ambientes = int(row['Ambientes']) if pd.notna(row['Ambientes']) and row['Ambientes'] != 0 else "?"
                badge = "<span style='background:#ffebee; color:#c62828; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold; margin-left:6px; vertical-align:middle;'>🔥 BAJÓ</span>" if (0 < precio_actual < primer_precio) else ""
                
                st.markdown(f"<div style='margin-top:4px; margin-bottom:8px;'><b>{barrio}</b> • {ambientes} Amb.{badge}</div>", unsafe_allow_html=True)
                
                piso_val = str(row.get('Piso', '')).strip()
                if piso_val.lower() == 'nan' or piso_val == '': piso_val = 'no menciona'
                elif piso_val.endswith('.0'): piso_val = piso_val[:-2]
                if piso_val == '0': piso_val = 'PB'

                antig_val = str(row.get('Antigüedad', '')).strip()
                if antig_val.lower() == 'nan' or antig_val == '' or antig_val.lower() == 'contactar agente':
                    antig_val = 'no menciona'

                valor_mostrar_piso = piso_val if str(piso_val).lower().startswith("piso") else f"Piso: {piso_val}"
                valor_mostrar_ant = antig_val if str(antig_val).lower().startswith("antig") else f"Antigüedad: {antig_val}"
                
                c_piso, c_ant = st.columns(2)
                with c_piso:
                    nuevo_piso = st.text_input("Piso", value=valor_mostrar_piso, key=f"piso_{idx}", label_visibility="collapsed")
                    if nuevo_piso != valor_mostrar_piso:
                        dato_limpio = nuevo_piso.replace("Piso: ", "").replace("Piso:", "").strip()
                        df.at[idx, "Piso"] = dato_limpio
                        guardar_datos(df)
                        st.rerun()
                with c_ant:
                    nuevo_ant = st.text_input("Antigüedad", value=valor_mostrar_ant, key=f"ant_{idx}", label_visibility="collapsed")
                    if nuevo_ant != valor_mostrar_ant:
                        dato_limpio = nuevo_ant.replace("Antigüedad: ", "").replace("Antigüedad:", "").strip()
                        df.at[idx, "Antigüedad"] = dato_limpio
                        guardar_datos(df)
                        st.rerun()

                m2_tot = int(row['M2 Totales'])
                m2_cub = int(row['M2 Cubiertos'])
                m2_desc = m2_tot - m2_cub if m2_tot > m2_cub else 0
                usd_m2 = int(row['USD/m2 Promedio'])
                banos = int(row['Baños']) if 'Baños' in row and pd.notna(row['Baños']) else 0
                toilettes = int(row['Toilettes']) if 'Toilettes' in row and pd.notna(row['Toilettes']) else 0
                
                detalles_html = f"📐 {m2_tot}m² Tot ({m2_cub}m² Cub) <br>"
                detalles_html += f"📊 Ratio: <b>{format_precio(usd_m2)}/m²</b> <br>"
                
                textos_sanitarios = []
                if banos > 0: textos_sanitarios.append(f"{banos} Baño{'s' if banos > 1 else ''}")
                if toilettes > 0: textos_sanitarios.append(f"{toilettes} Toil.")
                if textos_sanitarios:
                    detalles_html += f"🚿 {' | '.join(textos_sanitarios)} <br>"

                extras = []
                if str(row.get('Disposición', '')) != "": extras.append(str(row['Disposición']))
                if str(row.get('Balcón', '')) != "": extras.append(str(row['Balcón']))
                if str(row.get('Orientación', '')) != "": extras.append(f"Orientación: {str(row['Orientación'])}")
                if extras:
                    detalles_html += f"🧭 {' | '.join(extras)}"
                
                st.markdown(f"<div style='font-size: 13px; color: #555; line-height: 1.5; margin-bottom: 12px;'>{detalles_html}</div>", unsafe_allow_html=True)

                resumen_ia = str(row.get('Resumen IA', ''))
                if resumen_ia != "":
                    # Eliminamos el height límite para que se lea todo el reporte
                    st.markdown(f"<div style='font-size: 13px; color: #1e3a5f; background-color: #e8f4fd; padding: 12px; border-radius: 5px; margin-bottom: 8px; border-left: 3px solid #1E88E5; line-height: 1.6;'>✨ <b>Reporte IA:</b><br>{resumen_ia}</div>", unsafe_allow_html=True)

                nuevas_notas = st.text_area("Notas", value=str(row.get('Notas Personales', '')), height=68, key=f"notas_{idx}", label_visibility="collapsed", placeholder="📝 Escribí tus notas personales acá...")
                if nuevas_notas != str(row.get('Notas Personales', '')):
                    df.at[idx, "Notas Personales"] = nuevas_notas
                    guardar_datos(df)
                    st.rerun()

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
                    desc_completa = str(row['Descripción Completa'])
                    if desc_completa.strip():
                        st.markdown(f"<div style='font-size: 12px; color: #666; max-height: 150px; overflow-y: auto;'>{desc_completa}</div>", unsafe_allow_html=True)
                    else:
                        st.write("No se encontró texto original.")
                        
                with st.expander("🤖 Ver razonamiento de la IA (Debug)"):
                    raw_ia = str(row.get("Respuesta Cruda IA", "No hay datos de IA para esta propiedad."))
                    if raw_ia.strip() == "":
                        raw_ia = "No hay datos de IA guardados."
                    st.code(raw_ia, language="json")

                col_links, col_acts = st.columns(2)
                with col_links:
                    st.link_button("🔗 Ver Aviso", row['Link'], use_container_width=True)
                with col_acts:
                    if st.button("🔄 Actualizar", key=f"btn_act_{idx}", use_container_width=True):
                        with st.spinner("Generando nuevo reporte con IA..."):
                            link_actual = row["Link"]
                            if link_actual and str(link_actual).startswith("http"):
                                datos_frescos = extraer_datos_web(link_actual)
                                if datos_frescos:
                                    df.at[idx, "Descripción Completa"] = datos_frescos["Descripción Completa"]
                                    df.at[idx, "Resumen IA"] = datos_frescos["Resumen IA"]
                                    df.at[idx, "Respuesta Cruda IA"] = datos_frescos["Respuesta Cruda IA"]
                                    
                                    if int(datos_frescos["Precio (USD)"]) > 0:
                                        precio_nuevo = datos_frescos["Precio (USD)"]
                                        precio_viejo = int(row["Precio (USD)"]) if pd.notna(row["Precio (USD)"]) else 0
                                        
                                        if precio_nuevo > 0 and precio_nuevo != precio_viejo:
                                            hoy = datetime.now().strftime("%d/%m/%Y")
                                            historial_previo = str(row["Historial Precio"]) if pd.notna(row["Historial Precio"]) else ""
                                            registro_hoy = f"{hoy}: {format_precio(precio_nuevo)}"
                                            
                                            df.at[idx, "Precio (USD)"] = precio_nuevo
                                            if historial_previo == "":
                                                df.at[idx, "Historial Precio"] = registro_hoy
                                            else:
                                                df.at[idx, "Historial Precio"] = f"{historial_previo} | {registro_hoy}"
                                        
                                        df.at[idx, "M2 Totales"] = datos_frescos["M2 Totales"]
                                        df.at[idx, "M2 Cubiertos"] = datos_frescos["M2 Cubiertos"]
                                        df.at[idx, "M2 Ponderados"] = datos_frescos["M2 Ponderados"]
                                        df.at[idx, "USD/m2 Promedio"] = datos_frescos["USD/m2 Promedio"]
                                        df.at[idx, "Ambientes"] = datos_frescos["Ambientes"]
                                        
                                    guardar_datos(df)
                                    st.rerun()
else:
    st.info("No tenés propiedades cargadas. Pegá un link arriba para empezar.")
