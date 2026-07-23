"""
BioLoop Engine - Circular BioParts Engine for Waste Valorization
Synthetic biology platform for generating genetic strategies from biomass.
"""

import json
import logging
import re
import urllib.parse
import urllib.request
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from Bio import SeqIO
from dnachisel import CodonOptimize, DnaOptimizationProblem
from dnachisel.biotools import reverse_translate
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MAX_INPUT_LENGTH = 50
UNIPROT_SEARCH_SIZE = 3
DEFAULT_CHASSIS = "Escherichia coli"
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.1

# Chassis mapping for codon optimization
CHASSIS_MAPPING = {
    "pseudomonas putida": "p_putida",
    "escherichia coli": "e_coli",
    "saccharomyces cerevisiae": "s_cerevisiae"
}

# Translation dictionaries
RESIDUE_TRANSLATIONS = {
    "Cascara de Naranja": "Orange Peel",
    "Orujo de Uva": "Grape Pomace",
    "Bagazo de Cana": "Sugarcane Bagasse",
    "Suero Lacteo": "Dairy Whey",
    "Cascarilla de Cafe": "Coffee Husk"
}

REVERSE_TRANSLATIONS = {v: k for k, v in RESIDUE_TRANSLATIONS.items()}

JUSTIFICATIONS_EN = {
    "Orange Peel": "High conversion rate to limonene due to pectin richness.",
    "Grape Pomace": "Optimal lignocellulosic profile for synthesizing advanced biofuels.",
    "Sugarcane Bagasse": "High cellulose content for efficient enzymatic degradation.",
    "Dairy Whey": "Lactose-rich medium perfect for engineered yeast chassis.",
    "Coffee Husk": "Abundant source for valuable organic acid recovery."
}

CELL_TRANSLATIONS = {
    "Pectina": "Pectin",
    "Pectinasa": "Pectinase",
    "Limoneno": "Limonene",
    "Celulosa": "Cellulose",
    "Celulasa": "Cellulase",
    "Etanol": "Ethanol"
}


class LocalizationManager:
    """Manages multilingual support for the application."""

    TEXTS = {
        "ES": {
            "title_main": "Bioparts",
            "title_sub": "BioLoop: Circular BioParts Engine",
            "sub": "### Plataforma *In Silico* para Valorizacion de Residuos",
            "desc": "Genera estrategias de biologia sintetica y sintetiza el codigo genetico a partir de biomasa.",
            "dropdown": "Busca un residuo conocido (Autocompletar):",
            "other": "+ Escribir otro residuo nuevo...",
            "manual": "Escribe el nuevo residuo:",
            "btn": "INICIAR PROTOCOLO DE INFERENCIA",
            "found": "Ruta encontrada en base de datos local.",
            "inferring": "Residuo no catalogado. Iniciando agente predictivo y conectando a UniProt...",
            "justification": "Justificacion Cientifica:",
            "download": "Descargar Bio-Parte (.fasta)",
            "warn": "Por favor, introduce un residuo.",
            "ai_prompt": "Responde estrictamente en ESPANOL.",
            "tip": "Tip: Si tu residuo no esta en la lista principal, selecciona la ultima opcion para analizarlo con nuestro Agente de IA.",
            "synth": "Sintetizando Bio-Parte...",
            "success_dna": "Secuencia genetica generada y optimizada.",
            "err_codon": "Error en la optimizacion de codones.",
            "err_uniprot": "No se pudo descargar la secuencia desde UniProt.",
            "err_agent": "Error de comunicacion con el Agente. Intenta de nuevo."
        },
        "EN": {
            "title_main": "Bioparts",
            "title_sub": "BioLoop: Circular BioParts Engine",
            "sub": "### *In Silico* Platform for Waste Valorization",
            "desc": "Generate synthetic biology strategies and synthesize genetic code from biomass.",
            "dropdown": "Search a known residue (Autocomplete):",
            "other": "+ Type a new custom residue...",
            "manual": "Type the new residue:",
            "btn": "INITIATE INFERENCE PROTOCOL",
            "found": "Pathway found in local database.",
            "inferring": "Uncatalogued residue. Starting predictive agent and connecting to UniProt...",
            "justification": "Scientific Justification:",
            "download": "Download Bio-Part (.fasta)",
            "warn": "Please, enter a residue.",
            "ai_prompt": "Respond strictly in ENGLISH.",
            "tip": "Tip: If your residue is not in the main list, select the last option to analyze it with our AI Agent.",
            "synth": "Synthesizing Bio-Part...",
            "success_dna": "Genetic sequence generated and optimized.",
            "err_codon": "Error in codon optimization.",
            "err_uniprot": "Could not download the sequence from UniProt.",
            "err_agent": "Communication error with the Agent. Please try again."
        }
    }

    @classmethod
    def get_text(cls, key: str, language: str) -> str:
        """Get localized text for a given key and language."""
        return cls.TEXTS.get(language, cls.TEXTS["EN"]).get(key, key)


class UniProtClient:
    """Client for interacting with UniProt REST API."""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    @staticmethod
    def fetch_sequence(uniprot_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch protein sequence from UniProt by ID.

        Args:
            uniprot_id: UniProt identifier

        Returns:
            Tuple of (sequence, description) or (None, None) on failure
        """
        url = f"{UniProtClient.BASE_URL}/{uniprot_id}.fasta"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                record = SeqIO.read(
                    StringIO(response.read().decode('utf-8')),
                    "fasta"
                )
                return str(record.seq), record.description
        except Exception as error:
            logger.error("Error fetching sequence for %s: %s", uniprot_id, error)
            return None, None

    @staticmethod
    def search_enzymes(compound: str) -> str:
        """
        Search for enzymes related to a compound in UniProt.

        Args:
            compound: Compound name to search for

        Returns:
            JSON string with search results
        """
        query = f"{compound} AND reviewed:true AND existence:1"
        encoded_query = urllib.parse.quote(query)
        url = f"{UniProtClient.BASE_URL}/search?query={encoded_query}&format=json&size={UNIPROT_SEARCH_SIZE}"

        try:
            request = urllib.request.Request(
                url,
                headers={'Accept': 'application/json'}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []
            for entry in data.get('results', []):
                enzyme_name = "Desconocido"
                try:
                    enzyme_name = entry['proteinDescription']['recommendedName']['fullName']['value']
                except KeyError:
                    pass

                results.append({
                    "UniProt_ID": entry['primaryAccession'],
                    "Nombre_Enzima": enzyme_name,
                    "Organismo": entry.get('organism', {}).get('scientificName', 'Unknown')
                })

            return json.dumps(results) if results else "No se encontraron enzimas en UniProt para este compuesto."
        except Exception as error:
            logger.error("Error searching UniProt for %s: %s", compound, error)
            return f"Error al consultar UniProt: {str(error)}"


class CodonOptimizer:
    """Handles codon optimization for different chassis organisms."""

    @staticmethod
    def optimize(protein_sequence: str, chassis_name: str) -> Optional[str]:
        """
        Optimize codons for a given protein sequence and chassis.

        Args:
            protein_sequence: Amino acid sequence
            chassis_name: Target chassis organism name

        Returns:
            Optimized DNA sequence or None on failure
        """
        chassis_code = CHASSIS_MAPPING.get(chassis_name.lower(), "e_coli")
        try:
            initial_dna = reverse_translate(protein_sequence)
            problem = DnaOptimizationProblem(
                sequence=initial_dna,
                objectives=[CodonOptimize(species=chassis_code)]
            )
            problem.optimize()
            return problem.sequence
        except Exception as error:
            logger.error(
                "Codon optimization failed for chassis %s: %s",
                chassis_name, error
            )
            return None


class AIInferenceEngine:
    """Handles AI-powered inference for unknown residues."""

    SYSTEM_PROMPT = """
    Eres un experto en biologia sintetica. Tu tarea es analizar residuos agroindustriales 
    y proponer una enzima real y un chasis.
    {language_rule}
    Si no conoces el residuo, DEBES usar la herramienta 'buscar_herramienta_uniprot' 
    para extraer una enzima real antes de responder.
    Devuelve estrictamente un objeto JSON con esta estructura exacta, sin texto adicional:
    {{"Enzyme": "Nombre", "UniProt_ID": "ID_Real", "Chassis": "Escherichia coli", "Justification": "Razon corta"}}
    """

    def __init__(self, client: OpenAI):
        """
        Initialize the AI inference engine.

        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.tools = self._build_tools()

    @staticmethod
    def _build_tools() -> List[Dict]:
        """Build the tools configuration for function calling."""
        return [{
            "type": "function",
            "function": {
                "name": "buscar_herramienta_uniprot",
                "description": "Busca enzimas reales en UniProt para un compuesto especifico. Devuelve IDs y nombres.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "compuesto": {
                            "type": "string",
                            "description": "El compuesto clave a degradar en ingles (ej. 'pectin')."
                        }
                    },
                    "required": ["compuesto"]
                }
            }
        }]

    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitize user input to prevent injection.

        Args:
            text: Raw input text

        Returns:
            Sanitized text
        """
        text = text[:MAX_INPUT_LENGTH]
        return re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', '', text).strip()

    def infer(self, residue: str, language_rule: str) -> Optional[Dict]:
        """
        Perform AI inference for a given residue.

        Args:
            residue: Residue name to analyze
            language_rule: Language instruction for the AI

        Returns:
            Dictionary with inference results or None on failure
        """
        safe_residue = self.sanitize_input(residue)
        if not safe_residue:
            return None

        system_prompt = self.SYSTEM_PROMPT.format(language_rule=language_rule)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analiza este residuo: ### {safe_residue} ###"}
        ]

        try:
            # Initial inference
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=TEMPERATURE
            )

            message = response.choices[0].message

            # Handle tool calls if needed
            if message.tool_calls:
                messages.append(message)
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "buscar_herramienta_uniprot":
                        arguments = json.loads(tool_call.function.arguments)
                        real_results = UniProtClient.search_enzymes(
                            arguments.get("compuesto", "cellulose")
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": real_results
                        })

                # Final response after tool calls
                final_response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE
                )
                result_text = final_response.choices[0].message.content
            else:
                result_text = message.content

            return json.loads(
                result_text.strip()
                .replace("```json", "")
                .replace("```", "")
            )
        except Exception as error:
            logger.error("AI inference error: %s", error)
            return None


class UIManager:
    """Manages the Streamlit user interface."""

    @staticmethod
    def setup_page():
        """Configure Streamlit page settings and styles."""
        st.set_page_config(
            page_title="BioLoop Engine | ZFG",
            page_icon="DNA",
            layout="centered"
        )

        st.markdown("""
            <style>
            .stApp { 
                background-color: #0a0a0f; 
                color: #d1d5db; 
            }
            h1, h2, h3 { 
                color: #00ff41; 
                font-family: 'Courier New', Courier, monospace; 
            }
            .stButton>button { 
                color: #00ff41; 
                border: 1px solid #00ff41; 
                background-color: transparent; 
                width: 100%; 
            }
            .stButton>button:hover { 
                background-color: #00ff41; 
                color: #0a0a0f; 
            }
            div[data-testid="InputInstructions"] { 
                display: none !important; 
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header(texts: Dict):
        """Render the application header."""
        st.markdown(
            f"<h1 style='color: #deff9a; font-size: 48px; margin-bottom: 0;'>"
            f"{texts['title_main']}</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='font-size: 24px; margin-top: 0;'>"
            f"{texts['title_sub']}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(texts["sub"])
        st.write(texts["desc"])
        st.write("---")

    @staticmethod
    def get_residue_input(
        residues: List[str],
        texts: Dict,
        language: str
    ) -> str:
        """
        Get residue input from user with autocomplete support.

        Args:
            residues: List of known residues
            texts: Localized text dictionary
            language: Selected language code

        Returns:
            Selected or entered residue name
        """
        visual_list = [
            RESIDUE_TRANSLATIONS.get(res, res)
            if language == "EN" else res
            for res in residues
        ]

        st.caption(texts["tip"])

        chosen_option = st.selectbox(
            texts["dropdown"],
            visual_list + [texts["other"]],
            label_visibility="collapsed"
        )

        if chosen_option == texts["other"]:
            return st.text_input(
                texts["manual"],
                placeholder="Ej: Pina / Pineapple..."
            )

        return (
            REVERSE_TRANSLATIONS.get(chosen_option, chosen_option)
            if language == "EN"
            else chosen_option
        )

    @staticmethod
    def display_local_result(dataframe: pd.DataFrame, language: str):
        """Display results from local database with translations."""
        df_display = dataframe.copy()

        if language == "EN":
            df_display['Residue'] = (
                df_display['Residue']
                .map(RESIDUE_TRANSLATIONS)
                .fillna(df_display['Residue'])
            )
            df_display = df_display.rename(columns={
                "Key_Component": "Key Component",
                "Suggested_Enzyme": "Suggested Enzyme",
                "Target_Chassis": "Target Chassis",
                "Final_Product": "Final Product",
                "Justification": "Justification"
            })
            df_display = df_display.replace(CELL_TRANSLATIONS)
            df_display['Justification'] = (
                df_display['Residue']
                .map(JUSTIFICATIONS_EN)
                .fillna(df_display['Justification'])
            )

        st.dataframe(df_display, hide_index=True)

    @staticmethod
    def display_ai_result(
        result: Dict,
        texts: Dict,
        uniprot_client: UniProtClient,
        codon_optimizer: CodonOptimizer
    ):
        """Display AI inference results and handle sequence generation."""
        st.json(result)
        st.info(f"**{texts['justification']}** {result.get('Justification', '')}")

        uniprot_id = result.get("UniProt_ID")
        chassis = result.get("Chassis", DEFAULT_CHASSIS)

        if not uniprot_id or uniprot_id in ["ID_Real", "Desconocido"]:
            return

        st.write("---")
        st.write(texts["synth"])

        protein_seq, description = uniprot_client.fetch_sequence(uniprot_id)

        if not protein_seq:
            st.error(texts["err_uniprot"])
            return

        dna_sequence = codon_optimizer.optimize(protein_seq, chassis)

        if not dna_sequence:
            st.error(texts["err_codon"])
            return

        fasta_content = (
            f">{uniprot_id} | Optimized for {chassis} | BioLoop Engine\n"
            f"{dna_sequence}"
        )
        st.success(texts["success_dna"])
        st.download_button(
            label=texts["download"],
            data=fasta_content,
            file_name=f"BioLoop_{uniprot_id}.fasta",
            mime="text/plain"
        )


class BioLoopApp:
    """Main application class for BioLoop Engine."""

    def __init__(self):
        """Initialize the application components."""
        self.validate_configuration()
        self.ai_client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.getenv("GITHUB_TOKEN")
        )
        self.inference_engine = AIInferenceEngine(self.ai_client)
        self.uniprot_client = UniProtClient()
        self.codon_optimizer = CodonOptimizer()
        self.ui_manager = UIManager()

    @staticmethod
    def validate_configuration():
        """Validate required configuration and credentials."""
        api_token = os.getenv("GITHUB_TOKEN")
        if not api_token:
            st.error("ALERTA DE SEGURIDAD: Falta el token de API. Revisa el archivo .env.")
            st.stop()

    @st.cache_data
    @staticmethod
    def load_data() -> pd.DataFrame:
        """Load and cache the biological routes dataset."""
        return pd.read_csv("ZFG_bio_routes.csv")

    def run(self):
        """Execute the main application flow."""
        self.ui_manager.setup_page()

        # Language selection
        language = st.radio(
            "Language / Idioma",
            ["EN", "ES"],
            horizontal=True
        )
        texts = LocalizationManager.get_text
        t = lambda key: texts(key, language)

        # Render header
        self.ui_manager.render_header({k: t(k) for k in [
            "title_main", "title_sub", "sub", "desc"
        ]})

        # Load data
        df = self.load_data()
        residues = df['Residue'].tolist()

        # Get user input
        residue_input = self.ui_manager.get_residue_input(residues, {
            k: t(k) for k in ["dropdown", "other", "manual", "tip"]
        }, language)

        # Process button
        st.write("---")
        if st.button(t("btn")):
            self.process_residue(residue_input, df, t, language)

    def process_residue(
        self,
        residue_input: str,
        df: pd.DataFrame,
        texts_func,
        language: str
    ):
        """
        Process the entered residue and display results.

        Args:
            residue_input: Residue name entered by user
            df: DataFrame with known residues
            texts_func: Function to get localized text
            language: Selected language
        """
        if not residue_input.strip():
            st.warning(texts_func("warn"))
            return

        local_result = df[
            df['Residue'].str.lower() == residue_input.lower()
        ]

        if not local_result.empty:
            st.success(texts_func("found"))
            self.ui_manager.display_local_result(local_result, language)
        else:
            self.process_unknown_residue(residue_input, texts_func)

    def process_unknown_residue(self, residue_input: str, texts_func):
        """Process a residue not found in the local database."""
        with st.spinner(texts_func("inferring")):
            ai_result = self.inference_engine.infer(
                residue_input,
                texts_func("ai_prompt")
            )

            if ai_result:
                self.ui_manager.display_ai_result(
                    ai_result,
                    {k: texts_func(k) for k in [
                        "justification", "synth", "success_dna",
                        "err_codon", "err_uniprot", "download"
                    ]},
                    self.uniprot_client,
                    self.codon_optimizer
                )
            else:
                st.error(texts_func("err_agent"))


def main():
    """Application entry point."""
    app = BioLoopApp()
    app.run()


if __name__ == "__main__":
    main()
