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

# --- NUEVO CEREBRO: LA IA EXTRAE ABSOLUTAMENTE TODO ---
def extraer_todo_con_ia(texto_crudo):
    if not GROQ_API_KEY or not texto_crudo.strip():
        return {}
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""Actuá como un extractor de datos inmobiliarios experto. Tu objetivo es leer los datos en bruto de un aviso y extraer toda la información en un ÚNICO objeto JSON válido.

REGLA VITAL: RESPONDER ESTRICTAMENTE EN ESPAÑOL Y ÚNICAMENTE CON EL JSON. NO ESCRIBAS NADA FUERA DE LAS LLAVES {{ }}.

Estructura JSON requerida y reglas estrictas:
{{
    "barrio": "Nombre del barrio (ej. 'Palermo'). Si no dice, poné 'CABA'.",
    "precio": Número entero del precio en USD (ej. 95000). Si no dice, poné 0,
    "m2_totales": Número entero. Si no dice, poné 0,
    "m2_cubiertos": Número entero. Si no dice, poné 0,
    "ambientes": Número entero. Si no dice, poné 0,
    "banos": Número entero de baños. Si no dice, poné 0,
    "toilettes": Número entero de toilettes. Si no dice, poné 0,
    "piso": "Piso del departamento (ej. '3', 'PB'). PROHIBIDO adivinar por la cantidad total de pisos del edificio. Si no dice en qué piso está la unidad, poné 'no menciona'.",
    "antiguedad": "Años de antigüedad (ej. '10 años'). Si dice a estrenar, poné 'A estrenar'. Si es pozo, 'Pozo'. Si no dice nada, poné 'no menciona'.",
    "disposicion": "'Frente', 'Contrafrente', 'Lateral', o 'no menciona'.",
    "orientacion": "Punto cardinal ('N', 'S', 'E', 'O', 'NE', 'NO', 'SE', 'SO') o 'no menciona'.",
    "balcon": "'Balcón' si tiene, sino 'no menciona'.",
    "resumen": "2 o 3 renglones sobre características físicas y ventajas del inmueble."
}}

Datos en bruto del aviso a analizar:
{texto_crudo}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            temperature=0.1
        )
        
        respuesta = chat_completion.choices[0].message.content.strip()
        
        # Limpieza de formato markdown si la IA lo incluyó
        respuesta = re.sub(r'^```json\s*', '', respuesta)
        respuesta = re.sub(r'^```\s*', '', respuesta)
        respuesta = re.sub(r'\s*```$', '', respuesta)
        
        return json.loads(respuesta)
            
    except Exception as e:
        print(f"Error IA: {e}")
        return {}

def cargar_datos():
    columnas_base = [
        "Barrio", "Piso", "Ambientes", "Baños", "Toilettes", 
        "Disposición", "Orientación", "Balcón",
        "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "Precio (USD)", 
        "USD/m2 Promedio", "Antigüedad", "Link", "Resumen IA", "Notas Personales", 
        "Descripción Completa", "Historial Precio"
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
            
    for col in ["Barrio", "Piso", "Antigüedad", "Disposición", "Orientación", "Balcón", "Resumen IA", "Notas Personales", "Descripción Completa", "Historial Precio", "Link"]:
        df[col] = df[col].astype(object).fillna("")
        
    for col in ["Precio (USD)", "USD/m2 Promedio", "Ambientes", "Baños", "Toilettes", "M2 Totales", "M2 Cubiertos", "M2 Ponderados"]:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    df.reset_index(drop=True, inplace=True)
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
        
        descripcion_limpia = ""
        texto_para_ia = ""

        # Buscamos los datos ocultos de Zonaprop para pasárselos completos a la IA
        next_data_tag = sopa.find("script", id="__NEXT_DATA__")
        if next_data_tag:
            try:
                data_json = json.loads(next_data_tag.string)
                page_props = data_json.get("props", {}).get("pageProps", {})
                props = page_props.get("posting", {}) or page_props.get("initialPosting", {})
                
                if props:
                    descripcion_aviso = props.get("plainDescription", "") or props.get("description", "")
                    descripcion_limpia = BeautifulSoup(descripcion_aviso.replace("<br>", "\n").replace("<br/>", "\n").replace("</p>", "\n"), "html.parser").get_text(separator="\n").strip()
                    
                    # Armamos un super-bloque de texto con todo masticado para que la IA extraiga
                    texto_para_ia = f"TÍTULO: {props.get('title', '')}\n"
                    texto_para_ia += f"DATOS DE PRECIO: {props.get('priceOperations', [])}\n"
                    texto_para_ia += f"CARACTERÍSTICAS PRINCIPALES: {props.get('mainFeatures', [])}\n"
                    texto_para_ia += f"UBICACIÓN: {props.get('location', {})}\n"
                    texto_para_ia += f"DESCRIPCIÓN COMPLETA: {descripcion_limpia}"
            except:
                pass

        # Respaldo por si no encontramos el script oculto
        if not descripcion_limpia:
            div_desc = sopa.find(attrs={"data-qa": "posting-description"}) or sopa.find(id=re.compile("description", re.I))
            if div_desc:
                for br in div_desc.find_all("br"): br.replace_with("\n")
                descripcion_limpia = div_desc.get_text(separator="\n").strip()
                texto_para_ia = f"TÍTULO: {sopa.title.string}\nDESCRIPCIÓN COMPLETA: {descripcion_limpia}"

        # Limitamos a 6000 caracteres para no romper el límite de Groq
        texto_para_ia = texto_para_ia[:6000]

        # --- AQUÍ SUCEDE LA MAGIA: Delegamos TODA la extracción a la IA ---
        ia_data = extraer_todo_con_ia(texto_para_ia)
        if not ia_data: ia_data = {}

        # Mapeo seguro de datos devueltos por la IA
        try: precio = int(ia_data.get("precio", 0))
        except: precio = 0
        try: m2_tot = int(ia_data.get("m2_totales", 0))
        except: m2_tot = 0
        try: m2_cub = int(ia_data.get("m2_cubiertos", 0))
        except: m2_cub = 0
        try: ambientes = int(ia_data.get("ambientes", 0))
        except: ambientes = 0
        try: banos = int(ia_data.get("banos", 0))
        except: banos = 0
        try: toilettes = int(ia_data.get("toilettes", 0))
        except: toilettes = 0

        # Autocorrección matemática base
        if m2_cub > m2_tot: m2_cub = m2_tot
        if m2_tot > 0 and m2_cub == 0: m2_cub = m2_tot

        m2_descubiertos = m2_tot - m2_cub if m2_tot > m2_cub else 0
        m2_pond = m2_cub + (m2_descubiertos * 0.5)
        usd_m2 = round(precio / m2_pond) if m2_pond > 0 and precio > 0 else 0

        # Datos de texto devueltos por la IA
        barrio = str(ia_data.get("barrio", "CABA")).title()
        piso = str(ia_data.get("piso", "no menciona")).strip()
        antiguedad = str(ia_data.get("antiguedad", "no menciona")).strip()
        disposicion = str(ia_data.get("disposicion", "")).strip()
        orientacion = str(ia_data.get("orientacion", "")).strip()
        balcon = str(ia_data.get("balcon", "")).strip()
        resumen_ia = str(ia_data.get("resumen", "")).strip()

        # Limpieza visual final
        if piso.lower() in ["none", "", "null"]: piso = "no menciona"
        if antiguedad.lower() in ["none", "", "null"]: antiguedad = "no menciona"
        if disposicion.lower() in ["no menciona", "none", "null"]: disposicion = ""
        if orientacion.lower() in ["no menciona", "none", "null"]: orientacion = ""
        if balcon.lower() in ["no menciona", "none", "null"]: balcon = ""
        
        return {
            "Barrio": barrio, "Piso": piso, "Ambientes": ambientes,
            "Baños": banos, "Toilettes": toilettes,
            "Disposición": disposicion, "Orientación": orientacion, "Balcón": balcon,
            "M2 Totales": m2_tot, "M2 Cubiertos": m2_cub, "M2 Ponderados": m2_pond, 
            "Precio (USD)": precio, "USD/m2 Promedio": usd_m2, "Antigüedad": antiguedad, 
            "Link": url, "Resumen IA": resumen_ia, "Notas Personales": "", "Descripción Completa": descripcion_limpia, "Historial Precio": ""
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
        with st.spinner("Extrayendo datos y analizando con Inteligencia Artificial..."):
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
                            next_data_tag = sopa.find("script", id="__NEXT_DATA__")
                            precio_nuevo = 0
                            if next_data_tag:
                                data_json = json.loads(next_data_tag.string)
                                page_props = data_json.get("props", {}).get("pageProps", {})
                                props = page_props.get("posting", {}) or page_props.get("initialPosting", {})
                                precio_val = props.get("priceOperations", [{}])
                                if precio_val:
                                    precios_list = precio_val[0].get("prices", [])
                                    if precios_list:
                                        precio_nuevo = int(precios_list[0].get("amount", 0))
                            
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
                
                piso_val = str(row.get('Piso', '')).strip()
                if piso_val.lower() == 'nan' or piso_val == '': piso_val = 'no menciona'
                elif piso_val.endswith('.0'): piso_val = piso_val[:-2]
                if piso_val == '0': piso_val = 'PB'
                
                barrio = row['Barrio'] if row['Barrio'] else "Barrio a confirmar"
                ambientes = int(row['Ambientes']) if pd.notna(row['Ambientes']) and row['Ambientes'] != 0 else "?"
                badge = "<span style='background:#ffebee; color:#c62828; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold; margin-left:6px; vertical-align:middle;'>🔥 BAJÓ</span>" if (0 < precio_actual < primer_precio) else ""
                
                valor_mostrar_piso = piso_val if str(piso_val).lower().startswith("piso") else f"Piso: {piso_val}"
                
                c_info, c_piso = st.columns([6, 4])
                with c_info:
                    st.markdown(f"<div style='margin-top:6px; margin-bottom:8px;'><b>{barrio}</b> • {ambientes} Amb.{badge}</div>", unsafe_allow_html=True)
                with c_piso:
                    nuevo_piso = st.text_input("Piso", value=valor_mostrar_piso, key=f"piso_{idx}", label_visibility="collapsed")
                    if nuevo_piso != valor_mostrar_piso:
                        dato_limpio = nuevo_piso.replace("Piso: ", "").replace("Piso:", "").strip()
                        df.at[idx, "Piso"] = dato_limpio
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
                
                antig_val = str(row.get('Antigüedad', '')).strip()
                if antig_val.lower() == 'nan' or antig_val == '' or antig_val.lower() == 'contactar agente':
                    antig_val = 'no menciona'

                valor_mostrar_ant = antig_val if str(antig_val).lower().startswith("antig") else f"Antigüedad: {antig_val}"
                
                c_ant_lbl, c_ant_inp = st.columns([4, 6])
                with c_ant_lbl:
                    st.markdown("<div style='margin-top:7px; font-size:13px; font-weight:bold; color:#555;'>🏗️ Antigüedad:</div>", unsafe_allow_html=True)
                with c_ant_inp:
                    nuevo_ant = st.text_input("Antigüedad", value=valor_mostrar_ant, key=f"ant_{idx}", label_visibility="collapsed")
                    if nuevo_ant != valor_mostrar_ant:
                        dato_limpio = nuevo_ant.replace("Antigüedad: ", "").replace("Antigüedad:", "").strip()
                        df.at[idx, "Antigüedad"] = dato_limpio
                        guardar_datos(df)
                        st.rerun()

                resumen_ia = str(row.get('Resumen IA', ''))
                if resumen_ia != "":
                    st.markdown(f"<div style='font-size: 12px; color: #1e3a5f; background-color: #e8f4fd; padding: 8px; border-radius: 5px; margin-bottom: 8px; border-left: 3px solid #1E88E5;'>✨ <b>Resumen IA:</b> {resumen_ia}</div>", unsafe_allow_html=True)

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

                col_links, col_acts = st.columns(2)
                with col_links:
                    st.link_button("🔗 Ver Aviso", row['Link'], use_container_width=True)
                with col_acts:
                    if st.button("🔄 Actualizar", key=f"btn_act_{idx}", use_container_width=True):
                        with st.spinner("Descargando..."):
                            link_actual = row["Link"]
                            if link_actual and str(link_actual).startswith("http"):
                                datos_frescos = extraer
