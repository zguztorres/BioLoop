import re
import streamlit as st
import pandas as pd
import os
import json
import urllib.request
import urllib.parse
from io import StringIO
from openai import OpenAI
from Bio import SeqIO
from dnachisel import DnaOptimizationProblem, CodonOptimize
from dnachisel.biotools import reverse_translate
from dotenv import load_dotenv

# Cargar las variables secretas desde el archivo .env
load_dotenv() 

# 1. CONFIGURACIÓN VISUAL (Estética Sci-Fi)
# 1. CONFIGURACIÓN VISUAL (Estética Sci-Fi)
st.set_page_config(page_title="BioLoop Engine | ZFG", page_icon="🧬", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0a0a0f; color: #d1d5db; }
    h1, h2, h3 { color: #00ff41; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { color: #00ff41; border: 1px solid #00ff41; background-color: transparent; width: 100%; }
    .stButton>button:hover { background-color: #00ff41; color: #0a0a0f; }
    
    /* NUEVO: Ocultar el texto de Press Enter */
    div[data-testid="InputInstructions"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# 2. CONFIGURACIÓN DEL BACKEND (SEGURO 🔒)
token_seguro = os.getenv("GITHUB_TOKEN")

if not token_seguro:
    st.error("🚨 ALERTA DE SEGURIDAD: Falta el token de API. Revisa el archivo .env.")
    st.stop()

cliente_ia = OpenAI(
    base_url="https://models.inference.ai.azure.com", 
    api_key=token_seguro
)

MAPEO_CHASIS = {
    "pseudomonas putida": "p_putida",
    "escherichia coli": "e_coli",
    "saccharomyces cerevisiae": "s_cerevisiae"
}

@st.cache_data
def cargar_datos():
    return pd.read_csv("ZFG_bio_routes.csv")

df = cargar_datos()
lista_residuos = df['Residue'].tolist()

# 3. SISTEMA DE IDIOMAS (DICCIONARIO)
# 3. SISTEMA DE IDIOMAS (DICCIONARIO)
LANG = {
    "ES": {
        "title_main": "Bioparts",
        "title_sub": "🧬 BioLoop: Circular BioParts Engine",
        "sub": "### Plataforma *In Silico* para Valorización de Residuos",
        "desc": "Genera estrategias de biología sintética y sintetiza el código genético a partir de biomasa.",
        "dropdown": "🔍 Busca un residuo conocido (Autocompletar):",
        "other": "✍️ + Escribir otro residuo nuevo...",
        "manual": "Escribe el nuevo residuo:",
        "btn": "INICIAR PROTOCOLO DE INFERENCIA",
        "found": "🟢 Ruta encontrada en base de datos local.",
        "inferring": "🟡 Residuo no catalogado. Iniciando agente predictivo y conectando a UniProt...",
        "justification": "Justificación Científica:",
        "download": "📥 Descargar Bio-Parte (.fasta)",
        "warn": "⚠️ Por favor, introduce un residuo.",
        "ai_prompt": "Responde estrictamente en ESPAÑOL.",
        "tip": "💡 **Tip:** Si tu residuo no está en la lista principal, selecciona la última opción ('✍️ + Escribir otro...') para analizarlo con nuestro Agente de IA."
    },
    "EN": {
        "title_main": "Bioparts",
        "title_sub": "🧬 BioLoop: Circular BioParts Engine",
        "sub": "### *In Silico* Platform for Waste Valorization",
        "desc": "Generate synthetic biology strategies and synthesize genetic code from biomass.",
        "dropdown": "🔍 Search a known residue (Autocomplete):",
        "other": "✍️ + Type a new custom residue...",
        "manual": "Type the new residue:",
        "btn": "INITIATE INFERENCE PROTOCOL",
        "found": "🟢 Pathway found in local database.",
        "inferring": "🟡 Uncatalogued residue. Starting predictive agent and connecting to UniProt...",
        "justification": "Scientific Justification:",
        "download": "📥 Download Bio-Part (.fasta)",
        "warn": "⚠️ Please, enter a residue.",
        "ai_prompt": "Respond strictly in ENGLISH.",
        "tip": "💡 **Tip:** If your residue is not in the main list, select the last option ('✍️ + Type a new custom residue...') to analyze it with our AI Agent."
    }
}
idioma_seleccionado = st.radio("Language / Idioma", ["EN", "ES"], horizontal=True)
t = LANG[idioma_seleccionado]

# --- FUNCIONES BIOLÓGICAS Y DE HERRAMIENTAS ---
def obtener_secuencia_uniprot(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        with urllib.request.urlopen(url) as response:
            record = SeqIO.read(StringIO(response.read().decode('utf-8')), "fasta")
            return str(record.seq), record.description
    except:
        return None, None

def optimizar_con_dnachisel(secuencia_proteina, chasis_nombre):
    chasis_codigo = MAPEO_CHASIS.get(chasis_nombre.lower(), "e_coli")
    try:
        adn_inicial = reverse_translate(secuencia_proteina)
        problema = DnaOptimizationProblem(sequence=adn_inicial, objectives=[CodonOptimize(species=chasis_codigo)])
        problema.optimize()
        return problema.sequence
    except:
        return None

def buscar_herramienta_uniprot(compuesto):
    query_segura = urllib.parse.quote(f"{compuesto} AND reviewed:true AND existence:1")
    url = f"https://rest.uniprot.org/uniprotkb/search?query={query_segura}&format=json&size=3"
    
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            resultados = []
            for entry in data.get('results', []):
                nombre_enzima = "Desconocido"
                try:
                    nombre_enzima = entry['proteinDescription']['recommendedName']['fullName']['value']
                except KeyError:
                    pass
                
                resultados.append({
                    "UniProt_ID": entry['primaryAccession'],
                    "Nombre_Enzima": nombre_enzima,
                    "Organismo": entry.get('organism', {}).get('scientificName', 'Unknown')
                })
            
            return json.dumps(resultados) if resultados else "No se encontraron enzimas en UniProt para este compuesto."
    except Exception as e:
        return f"Error al consultar UniProt: {str(e)}"

# --- SEGURIDAD E IA ---
def sanitizar_input(texto):
    texto = texto[:50]
    return re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', '', texto).strip()

def inferir_con_ia(residuo, regla_idioma):
    residuo_seguro = sanitizar_input(residuo)
    if not residuo_seguro: return None

    instrucciones_sistema = f"""
    Eres un experto en biología sintética. Tu tarea es analizar residuos agroindustriales y proponer una enzima real y un chasis.
    {regla_idioma}
    Si no conoces el residuo, DEBES usar la herramienta 'buscar_herramienta_uniprot' para extraer una enzima real antes de responder.
    Devuelve estrictamente un objeto JSON con esta estructura exacta, sin texto adicional:
    {{"Enzyme": "Nombre", "UniProt_ID": "ID_Real", "Chassis": "Escherichia coli", "Justification": "Razón corta"}}
    """
    
    herramientas = [{
        "type": "function",
        "function": {
            "name": "buscar_herramienta_uniprot",
            "description": "Busca enzimas reales en UniProt para un compuesto específico. Devuelve IDs y nombres.",
            "parameters": {
                "type": "object",
                "properties": {
                    "compuesto": {"type": "string", "description": "El compuesto clave a degradar en inglés (ej. 'pectin')."}
                },
                "required": ["compuesto"]
            }
        }
    }]

    mensajes_historial = [
        {"role": "system", "content": instrucciones_sistema},
        {"role": "user", "content": f"Analiza este residuo: ### {residuo_seguro} ###"}
    ]

    try:
        respuesta = cliente_ia.chat.completions.create(
            model="gpt-4o", messages=mensajes_historial, tools=herramientas, tool_choice="auto", temperature=0.1
        )
        mensaje_respuesta = respuesta.choices[0].message
        
        if mensaje_respuesta.tool_calls:
            mensajes_historial.append(mensaje_respuesta)
            for tool_call in mensaje_respuesta.tool_calls:
                if tool_call.function.name == "buscar_herramienta_uniprot":
                    argumentos = json.loads(tool_call.function.arguments)
                    resultados_reales = buscar_herramienta_uniprot(argumentos.get("compuesto", "cellulose"))
                    mensajes_historial.append({
                        "role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": resultados_reales
                    })
            
            respuesta_final = cliente_ia.chat.completions.create(model="gpt-4o", messages=mensajes_historial, temperature=0.1)
            resultado_texto = respuesta_final.choices[0].message.content
        else:
            resultado_texto = mensaje_respuesta.content

        return json.loads(resultado_texto.strip().replace("```json", "").replace("```", ""))
    except Exception as e:
        print(f"Error en IA: {e}")
        return None

# 4. CONSTRUCCIÓN DE LA INTERFAZ
st.markdown(f"<h1 style='color: #deff9a; font-size: 48px; margin-bottom: 0;'>{t['title_main']}</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='font-size: 24px; margin-top: 0;'>{t['title_sub']}</h3>", unsafe_allow_html=True)
st.markdown(t["sub"])
st.write(t["desc"])
st.write("---")

# --- SISTEMA DE AUTOCOMPLETADO Y UX ---
TRADUCCION_RESIDUOS_EN = {
    "Cáscara de Naranja": "Orange Peel", "Orujo de Uva": "Grape Pomace",
    "Bagazo de Caña": "Sugarcane Bagasse", "Suero Lácteo": "Dairy Whey", "Cascarilla de Café": "Coffee Husk"
}
TRADUCCION_INVERSA = {v: k for k, v in TRADUCCION_RESIDUOS_EN.items()}

lista_visual = [TRADUCCION_RESIDUOS_EN.get(res, res) for res in lista_residuos] if idioma_seleccionado == "EN" else lista_residuos

st.caption("💡 **Tip:** Si tu residuo no está en la lista principal, selecciona la última opción ('✍️ + Escribir otro...') para analizarlo con nuestro Agente de IA.")
opcion_elegida = st.selectbox(t["dropdown"], lista_visual + [t["other"]], label_visibility="collapsed")

residuo_input = ""
if opcion_elegida == t["other"]:
    residuo_input = st.text_input(t["manual"], placeholder="Ej: Piña / Pineapple...")
else:
    residuo_input = TRADUCCION_INVERSA.get(opcion_elegida, opcion_elegida) if idioma_seleccionado == "EN" else opcion_elegida

# --- LÓGICA DE EJECUCIÓN ---
st.caption(t["tip"])

if st.button(t["btn"]):
    if not residuo_input.strip():
        st.warning(t["warn"])
    else:
        resultado_local = df[df['Residue'].str.lower() == residuo_input.lower()]
        
        if not resultado_local.empty:
            st.success(t["found"])
            df_mostrar = resultado_local.copy()

            if idioma_seleccionado == "EN":
                df_mostrar['Residue'] = df_mostrar['Residue'].map(TRADUCCION_RESIDUOS_EN).fillna(df_mostrar['Residue'])
                df_mostrar = df_mostrar.rename(columns={
                    "Key_Component": "Key Component", "Suggested_Enzyme": "Suggested Enzyme",
                    "Target_Chassis": "Target Chassis", "Final_Product": "Final Product", "Justification": "Justification"
                })
                TRAD_CELDAS = {
                    "Pectina": "Pectin", "Pectinasa": "Pectinase", "Limoneno": "Limonene",
                    "Celulosa": "Cellulose", "Celulasa": "Cellulase", "Etanol": "Ethanol"
                }
                df_mostrar = df_mostrar.replace(TRAD_CELDAS)
                
                JUSTIFICACIONES_EN = {
                    "Orange Peel": "High conversion rate to limonene due to pectin richness.",
                    "Grape Pomace": "Optimal lignocellulosic profile for synthesizing advanced biofuels.",
                    "Sugarcane Bagasse": "High cellulose content for efficient enzymatic degradation.",
                    "Dairy Whey": "Lactose-rich medium perfect for engineered yeast chassis.",
                    "Coffee Husk": "Abundant source for valuable organic acid recovery."
                }
                df_mostrar['Justification'] = df_mostrar['Residue'].map(JUSTIFICACIONES_EN).fillna(df_mostrar['Justification'])
            
            st.dataframe(df_mostrar, hide_index=True)
            
        else:
            with st.spinner(t["inferring"]):
                respuesta_ia = inferir_con_ia(residuo_input, t["ai_prompt"])
                
                if respuesta_ia:
                    st.json(respuesta_ia)
                    st.info(f"**{t['justification']}** {respuesta_ia.get('Justification', '')}")
                    
                    uniprot_id = respuesta_ia.get("UniProt_ID")
                    chasis = respuesta_ia.get("Chassis", "Escherichia coli")
                    
                    if uniprot_id and uniprot_id not in ["ID_Real", "Desconocido"]:
                        st.write("---")
                        st.write("🧬 **Sintetizando Bio-Parte...**")
                        
                        secuencia_prot, desc = obtener_secuencia_uniprot(uniprot_id)
                        
                        if secuencia_prot:
                            secuencia_adn = optimizar_con_dnachisel(secuencia_prot, chasis)
                            
                            if secuencia_adn:
                                fasta_content = f">{uniprot_id} | Optimized for {chasis} | BioLoop Engine\n{secuencia_adn}"
                                st.success("✅ Secuencia genética generada y optimizada.")
                                st.download_button(
                                    label=t["download"],
                                    data=fasta_content,
                                    file_name=f"BioLoop_{uniprot_id}.fasta",
                                    mime="text/plain"
                                )
                            else:
                                st.error("❌ Error en la optimización de codones.")
                        else:
                            st.error("❌ No se pudo descargar la secuencia desde UniProt.")
                else:
                    st.error("❌ Error de comunicación con el Agente. Intenta de nuevo.")
