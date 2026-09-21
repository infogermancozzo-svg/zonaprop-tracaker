def resumir_con_gemini(texto):
    if not GEMINI_API_KEY or not texto.strip():
        return ""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""Actuá como un experto tasador inmobiliario. Tu tarea es leer TODA la descripción del aviso y crear un resumen de máximo 2 o 3 renglones. 
        
        REGLAS ESTRICTAS:
        1. Enfocate ÚNICAMENTE en las características físicas y ventajas del inmueble (distribución, luminosidad, estado de conservación, amenities, ubicación).
        2. IGNORÁ por completo "disclosures", textos legales, leyes de accesibilidad (ej. Ley 5115), matrículas de corredores (CUCICBA, CPI), avisos de medidas aproximadas, horarios de atención o información de la inmobiliaria.
        3. Redactá un solo párrafo fluido, directo al grano y sin usar viñetas.
        
        Descripción original del aviso:
        {texto}"""
        
        respuesta = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return respuesta.text.strip()
    except Exception as e:
        # Esto va a mostrar el error de Google directo en la pantalla de la app
        return f"[Error IA]: {str(e)}"
