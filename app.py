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
            model="mixtral-8x7b-32768",
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
                        if precios_list: precio = int(precios_list[0].get("amount", 0))
                    
                    # Extracción profunda de Features (Baños, Toilettes, Antigüedad)
                    features = props.get("mainFeatures", [])
                    if isinstance(features, dict): features = list(features.values())
                    
                    for feat in features:
                        f_str = str(feat).upper()
                        
                        # Metros y Ambientes
                        if "M²" in f_str or "METROS" in f_str:
                            m2_match = re.search(r'(\d+)', f_str)
                            if m2_match:
                                val = int(m2_match.group(1))
                                if "CUB" in f_str: m2_cub = val
                                elif "TOT" in f_str: m2_tot = val
                                elif m2_tot == 0: m2_tot = val
                        if "AMB" in f_str:
                            amb_match = re.search(r'(\d+)', f_str)
                            if amb_match: ambientes = int(amb_match.group(1))
                            
                        # Baños y Toilettes
                        if "BAÑO" in f_str or "BATHROOM" in f_str:
                            num = re.search(r'(\d+)', f_str)
                            if num: banos = int(num.group(1))
                        if "TOILET" in f_str:
                            num = re.search(r'(\d+)', f_str)
                            if num: toilettes = int(num.group(1))
                            
                        # Antigüedad en la etiqueta web
                        if "A ESTRENAR" in f_str: antiguedad_web = "A estrenar"
                        elif "POZO" in f_str: antiguedad_web = "Pozo"
                        elif "CONSTRUCCI" in f_str: antiguedad_web = "En construcción"
                        elif "AÑO" in f_str:
                            a_match = re.search(r'(\d+)', f_str)
                            if a_match: antiguedad_web = f"{a_match.group(1)} años"

                    location = props.get("location", {})
                    b_name = location.get("parent", {}).get("name", "") or location.get("name", "")
                    if b_name: barrio = b_name.title()
            except Exception:
                pass

        if not descripcion_aviso:
            div_desc = sopa.find(attrs={"data-qa": "posting-description"}) or sopa.find(id=re.compile("description", re.I))
            if div_desc:
                for br in div_desc.find_all("br"): br.replace_with("\n")
                for p in div_desc.find_all("p"): p.insert_after("\n")
                descripcion_aviso = div_desc.get_text(separator=" ").strip()

        descripcion_limpia = ""
        if descripcion_aviso:
            descripcion_aviso = descripcion_aviso.replace("<br>", "\n").replace("<br/>", "\n").replace("</p>", "\n")
            descripcion_limpia = BeautifulSoup(descripcion_aviso, "html.parser").get_text(separator="\n")
            descripcion_limpia = re.sub(r'\n+', '\n', descripcion_limpia).strip()

        texto_completo = f"{titulo_texto} {descripcion_limpia} {sopa.get_text(separator=' ')}".upper()

        # Búsqueda de rescate en todo el texto visible si los campos json fallaron
        if banos == 0:
            b_match = re.search(r'(\d+)\s*BAÑO', texto_completo)
            if b_match: banos = int(b_match.group(1))
        if toilettes == 0:
            t_match = re.search(r'(\d+)\s*TOILET', texto_completo)
            if t_match: toilettes = int(t_match.group(1))
            
        if not antiguedad_web:
            if re.search(r'\bA ESTRENAR\b', texto_completo): antiguedad_web = "A estrenar"
            elif re.search(r'\b(?:EN POZO|POZO)\b', texto_completo): antiguedad_web = "Pozo"
            elif re.search(r'\b(?:EN CONSTRUCCI[OÓ]N)\b', texto_completo): antiguedad_web = "En construcción"
            else:
                a_match = re.search(r'(\d+)\s*AÑO', texto_completo)
                if a_match: antiguedad_web = f"{a_match.group(1)} años"

        # LLAMADA A LA IA CON REGLAS JSON ESTRICTAS
        ia_data = resumir_con_ia(descripcion_limpia)
        resumen_ia = ia_data["resumen"]
        
        # Priorizar la antigüedad extraída directo de la web. Si no hay, usar la deducción de la IA.
        antiguedad_final = antiguedad_web if antiguedad_web else ia_data["antiguedad"]
        if not antiguedad_final: 
            antiguedad_final = "Contactar agente"

        if not descripcion_limpia:
            descripcion_limpia = "No se pudo extraer la descripción."
            resumen_ia = "Sin descripción disponible."

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
            "Baños": banos, "Toilettes": toilettes, # <-- NUEVAS COLUMNAS
            "M2 Totales": m2_tot, "M2 Cubiertos": m2_cub, "M2 Ponderados": m2_pond, 
            "Precio (USD)": precio, "USD/m2 Promedio": usd_m2, "Antigüedad": antiguedad_final, 
            "Link": url, "Notas Personales": resumen_ia, "Descripción Completa": descripcion_limpia, "Historial Precio": ""
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
                            
                            if precio_nuevo > 0:
                                historial_previo = str(row["Historial Precio"]) if pd.notna(row["Historial Precio"]) else ""
                                registro_hoy = f"{hoy}: USD {precio_nuevo}"
                                
                                if registro_hoy not in historial_previo:
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
    columnas_grid = st.columns(2)
    
    for idx, row in df.iterrows():
        col_actual = columnas_grid[idx % 2]
        
        with col_actual:
            with st.container(border=True):
                col_titulo, col_borrar = st.columns([6, 1])
                with col_titulo:
                    barrio = row['Barrio'] if row['Barrio'] else "Barrio a confirmar"
                    ambientes = int(row['Ambientes']) if row['Ambientes'] else "?"
                    banos = int(row['Baños']) if 'Baños' in row and pd.notna(row['Baños']) else 0
                    toilettes = int(row['Toilettes']) if 'Toilettes' in row and pd.notna(row['Toilettes']) else 0
                    
                    # Título dinámico que suma baños y toilettes solo si existen
                    texto_titulo = f"### {barrio} • {ambientes} Amb."
                    if banos > 0: texto_titulo += f" • {banos} Baños"
                    if toilettes > 0: texto_titulo += f" • {toilettes} Toil."
                    
                    st.markdown(texto_titulo)
                    
                with col_borrar:
                    if st.button("🗑️", key=f"btn_del_{idx}", help="Eliminar definitivamente"):
                        df = df.drop(idx)
                        guardar_datos(df)
                        st.rerun()
                
                precio = int(row['Precio (USD)'])
                m2_tot = int(row['M2 Totales'])
                m2_cub = int(row['M2 Cubiertos'])
                m2_desc = m2_tot - m2_cub if m2_tot > m2_cub else 0
                usd_m2 = int(row['USD/m2 Promedio'])
                
                st.markdown(f"**💰 Precio:** USD {precio} | **📊 Precio Ponderado:** USD {usd_m2} / m²")
                st.markdown(f"**📐 Superficie:** {m2_tot} m² Totales | {m2_cub} m² Cub. | {m2_desc} m² Desc.")
                
                piso_val = str(row['Piso']).strip()
                if piso_val.endswith('.0'): piso_val = piso_val[:-2]
                if piso_val == '0': piso_val = 'PB'
                
                antig_val = str(row.get('Antigüedad', 'Contactar agente'))
                if antig_val == "": antig_val = "Contactar agente"
                
                col_piso, col_ant = st.columns([1, 2])
                with col_piso:
                    nuevo_piso = st.text_input("🏢 Piso", value=piso_val, key=f"piso_{idx}")
                    if nuevo_piso != piso_val:
                        df.at[idx, "Piso"] = nuevo_piso
                        guardar_datos(df)
                        st.rerun()
                with col_ant:
                    nuevo_ant = st.text_input("🏗️ Estado / Antigüedad", value=antig_val, key=f"ant_{idx}")
                    if nuevo_ant != antig_val:
                        df.at[idx, "Antigüedad"] = nuevo_ant
                        guardar_datos(df)
                        st.rerun()

                nuevas_notas = st.text_area("✨ Resumen (IA)", value=str(row['Notas Personales']), height=100, key=f"notas_{idx}")
                if nuevas_notas != str(row['Notas Personales']):
                    df.at[idx, "Notas Personales"] = nuevas_notas
                    guardar_datos(df)
                    st.rerun()

                with st.expander("📖 Ver descripción original completa"):
                    desc_completa = str(row['Descripción Completa'])
                    if desc_completa.strip():
                        st.write(desc_completa)
                    else:
                        st.write("No se encontró texto original.")

                col_links, col_acts = st.columns(2)
                with col_links:
                    st.link_button("🔗 Ver Publicación Original", row['Link'], use_container_width=True)
                with col_acts:
                    if st.button("🔄 Actualizar Info / IA", key=f"btn_act_{idx}", use_container_width=True):
                        with st.spinner("Reintentando descargar datos e IA..."):
                            link_actual = row["Link"]
                            if link_actual and str(link_actual).startswith("http"):
                                datos_frescos = extraer_datos_web(link_actual)
                                if datos_frescos:
                                    # Actualizamos todo el bloque para que incorpore los baños y antigüedad
                                    df.at[idx, "Descripción Completa"] = datos_frescos["Descripción Completa"]
                                    df.at[idx, "Notas Personales"] = datos_frescos["Notas Personales"]
                                    df.at[idx, "Antigüedad"] = datos_frescos["Antigüedad"]
                                    df.at[idx, "Baños"] = datos_frescos["Baños"]
                                    df.at[idx, "Toilettes"] = datos_frescos["Toilettes"]
                                    
                                    precio_nuevo = datos_frescos["Precio (USD)"]
                                    if precio_nuevo > 0 and precio_nuevo != row["Precio (USD)"]:
                                        hoy = datetime.now().strftime("%d/%m/%Y")
                                        historial_previo = str(row["Historial Precio"]) if pd.notna(row["Historial Precio"]) else ""
                                        registro_hoy = f"{hoy}: USD {precio_nuevo}"
                                        
                                        if registro_hoy not in historial_previo:
                                            df.at[idx, "Precio (USD)"] = precio_nuevo
                                            if historial_previo == "":
                                                df.at[idx, "Historial Precio"] = registro_hoy
                                            else:
                                                df.at[idx, "Historial Precio"] = f"{historial_previo} | {registro_hoy}"
                                    
                                    guardar_datos(df)
                                    st.rerun()
                
                with st.expander("📉 Ver historial de precios"):
                    historial = str(row["Historial Precio"])
                    if historial and historial.strip() != "":
                        for item in historial.split("|"):
                            st.write(f"• {item.strip()}")
                    else:
                        st.write("Sin cambios registrados.")
else:
    st.info("No tenés propiedades cargadas. Pegá un link arriba para empezar.")
