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

# AQUÍ ESTÁ EL ÚNICO TIP BILINGÜE
st.caption(t["tip"])

opcion_elegida = st.selectbox(t["dropdown"], lista_visual + [t["other"]], label_visibility="collapsed")

residuo_input = ""
if opcion_elegida == t["other"]:
    residuo_input = st.text_input(t["manual"], placeholder="Ej: Piña / Pineapple...")
else:
    residuo_input = TRADUCCION_INVERSA.get(opcion_elegida, opcion_elegida) if idioma_seleccionado == "EN" else opcion_elegida

# --- LÓGICA DE EJECUCIÓN ---
st.write("---")

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
