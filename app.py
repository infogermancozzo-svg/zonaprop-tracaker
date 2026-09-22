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
        
        prompt = f"""Actuá como un analista inmobiliario de élite. Analizá el texto bruto de un aviso de Zonaprop y estructurá la información en una 'Ficha Limpia'.

REGLA VITAL: Devolvé ÚNICAMENTE un objeto JSON válido con dos claves: "precio_usd" y "ficha_limpia". NO escribas nada fuera de las llaves {{ }}.

Estructura JSON requerida:
{{
    "precio_usd": Número entero con el precio de la propiedad (ej: 120000). Si no hay precio, poné 0.,
    "ficha_limpia": "Texto en formato Markdown siguiendo ESTRICTAMENTE el orden y las reglas de abajo."
}}

REGLAS PARA EL TEXTO DE 'ficha_limpia':
Redactá el texto usando este orden exacto. Usá negritas para los títulos (ej: **Barrio:**). Si un dato no está y no podés deducirlo por contexto, OMITÍ EL ÍTEM COMPLETO (no escribas 'no menciona' ni dejes el título vacío, simplemente saltá al siguiente punto). Utilizá saltos de línea (\\n) entre cada ítem.

ORDEN OBLIGATORIO:
1. **Barrio:** (Debe ser el barrio específico, ej: Villa Urquiza. NUNCA pongas CABA, Capital Federal, GBA o Provincia).
2. **Dirección:** (Exacta o aproximada).
3. **Piso:** (Intentá deducirlo de la información bruta. Si es imposible saberlo, omití este ítem).
4. **Operación:** (Ej: Venta - USD 120.000 o Alquiler - ARS 400.000).
5. **Ambientes:** (Cantidad).
6. **Dormitorios:** (Solo si lo explicita).
7. **Superficie:** (Ej: 50m² Totales, 45m² Cubiertos, 5m² Descubiertos. Incluí solo los que encuentres).
8. **Baños y Toilettes:** (Ej: 1 Baño, 1 Toilette).
9. **Orientación:** (Norte, Sur, Este, Oeste, etc. Si no lo dice, omitilo).
10. **Disposición:** (Frente o Contrafrente. Si es otra o no dice, omitilo).
11. **Antigüedad:** (Años, A estrenar, o En construcción. Deducilo de la info bruta: si el aviso tiene fotos de renders y no reales, o si habla de cuotas y plazos, suele ser 'En construcción'. Si es en construcción, agregá la fecha aproximada de entrega. Si no podés deducirlo de ninguna forma, omití este ítem).

--- TEXTO COMPLETO EXTRAÍDO DEL LINK DEL AVISO ---
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
    # Mantenemos las columnas base para que no se rompa tu CSV anterior, pero agregamos "Ficha Limpia"
    columnas_base = [
        "Link", "Precio (USD)", "Historial Precio", "Ficha Limpia", "Notas Personales", "Respuesta Cruda IA",
        "Barrio", "Piso", "Ambientes", "Baños", "Toilettes", "Disposición", "Orientación", "Balcón",
        "M2 Totales", "M2 Cubiertos", "M2 Ponderados", "USD/m2 Promedio", "Antigüedad", "Resumen IA", "Descripción Completa"
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
            
    for col in ["Ficha Limpia", "Notas Personales", "Historial Precio", "Link", "Respuesta Cruda IA"]:
        df[col] = df[col].astype(object).fillna("")
        
    for col in ["Precio (USD)"]:
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

        ia_data, raw_ia_response = extraer_todo_con_ia(texto_visible[:8000])
        if not ia_data: ia_data = {}

        precio = parse_num_seguro(ia_data.get("precio_usd", 0))
        ficha_limpia = str(ia_data.get("ficha_limpia", "")).strip()
        
        return {
            "Link": url, 
            "Precio (USD)": precio, 
            "Ficha Limpia": ficha_limpia, 
            "Notas Personales": "", 
            "Historial Precio": "", 
            "Respuesta Cruda IA": raw_ia_response
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
        with st.spinner("Generando Ficha Limpia con IA..."):
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
                                precio_nuevo = parse_num_seguro(ia_rapida.get("precio_usd", 0))
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
                # Botón de eliminar arriba a la derecha
                c_vacio, c_borrar = st.columns([5, 1])
                with c_borrar:
                    if st.button("🗑️", key=f"btn_del_{idx}", help="Eliminar"):
                        df = df.drop(idx).reset_index(drop=True)
                        guardar_datos(df)
                        st.rerun()

                # Mostrar historial y alertar si bajó de precio
                precio_actual = int(row['Precio (USD)']) if pd.notna(row['Precio (USD)']) else 0
                historial_str = str(row.get("Historial Precio", ""))
                primer_precio = precio_actual
                
                if historial_str:
                    primer_registro = historial_str.split("|")[0]
                    match_primer_precio = re.search(r'(?:USD|\$)\s*([\d\.]+)', primer_registro.replace('.', ''))
                    if match_primer_precio:
                        primer_precio = int(match_primer_precio.group(1))

                if 0 < precio_actual < primer_precio:
                    st.markdown("<div style='background:#ffebee; color:#c62828; padding:4px 8px; border-radius:4px; font-size:12px; font-weight:bold; margin-bottom:10px; display:inline-block;'>🔥 BAJÓ DE PRECIO</div>", unsafe_allow_html=True)

                # --- RENDERIZADO DE LA FICHA LIMPIA ---
                ficha_limpia = str(row.get("Ficha Limpia", "")).strip()
                
                if ficha_limpia == "":
                    # Si es una propiedad vieja del CSV que no tiene ficha, armamos un aviso
                    st.warning("Ficha no generada aún. Hacé clic en 'Actualizar' abajo.")
                else:
                    st.markdown(f"<div style='font-size: 14px; color: #333; line-height: 1.6;'>{ficha_limpia}</div>", unsafe_allow_html=True)
                
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

                # --- NOTAS PERSONALES ---
                nuevas_notas = st.text_area("Notas", value=str(row.get('Notas Personales', '')), height=68, key=f"notas_{idx}", label_visibility="collapsed", placeholder="📝 Escribí tus notas personales acá...")
                if nuevas_notas != str(row.get('Notas Personales', '')):
                    df.at[idx, "Notas Personales"] = nuevas_notas
                    guardar_datos(df)
                    st.rerun()

                # --- EXPANDERS (Historial y Debug) ---
                with st.expander("📉 Historial de Precios"):
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
                        
                with st.expander("🤖 Ver razonamiento de la IA (Debug)"):
                    raw_ia = str(row.get("Respuesta Cruda IA", "No hay datos de IA para esta propiedad."))
                    if raw_ia.strip() == "":
                        raw_ia = "No hay datos de IA guardados."
                    st.code(raw_ia, language="json")

                # --- BOTONES DE ACCIÓN ---
                col_links, col_acts = st.columns(2)
                with col_links:
                    st.link_button("🔗 Ver Aviso", row['Link'], use_container_width=True)
                with col_acts:
                    if st.button("🔄 Actualizar", key=f"btn_act_{idx}", use_container_width=True):
                        with st.spinner("Regenerando Ficha con IA..."):
                            link_actual = row["Link"]
                            if link_actual and str(link_actual).startswith("http"):
                                datos_frescos = extraer_datos_web(link_actual)
                                if datos_frescos:
                                    df.at[idx, "Ficha Limpia"] = datos_frescos["Ficha Limpia"]
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
                                        
                                    guardar_datos(df)
                                    st.rerun()
else:
    st.info("No tenés propiedades cargadas. Pegá un link arriba para empezar.")
