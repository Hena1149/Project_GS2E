import streamlit as st
import pandas as pd

st.set_page_config(page_title="Générateur de Cas de Test", layout="wide")

st.title("🧪 Générateur de Cas de Test à partir du CDC")

st.markdown("""
Ce générateur permet de créer des cas de test à partir du **cahier de charges (CDC)** et des **règles de gestion**.  
Les **points de contrôle (PDC)** sont **optionnels**.
""")

# Chargement des fichiers CSV
cdc_file = st.file_uploader("📂 Charger le CDC (Fichier CSV)", type="csv")
rg_file = st.file_uploader("📂 Charger les Règles de Gestion (Fichier CSV)", type="csv")
pdc_file = st.file_uploader("📂 Charger les PDC (Fichier CSV - facultatif)", type="csv")

if cdc_file is not None and rg_file is not None:
    df_cdc = pd.read_csv(cdc_file)
    df_rg = pd.read_csv(rg_file)

    st.subheader("📘 Cahier de Charges (CDC)")
    st.dataframe(df_cdc)

    st.subheader("📏 Règles de Gestion")
    st.dataframe(df_rg)

    # PDC facultatif
    if pdc_file is not None:
        df_pdc = pd.read_csv(pdc_file)
        st.subheader("🧩 Points de Contrôle (PDC)")
        st.dataframe(df_pdc)
    else:
        df_pdc = None
        st.info("Aucun fichier de PDC chargé. Le générateur fonctionnera uniquement avec les règles de gestion.")

    st.markdown("---")
    st.subheader("🛠 Génération des Cas de Test")

    # Logique simple de génération (à améliorer selon besoin réel)
    cas_de_test = []

    for index, regle in df_rg.iterrows():
        cdc_associe = df_cdc[df_cdc["id_cdc"] == regle["id_cdc"]]
        description_cdc = cdc_associe["description"].values[0] if not cdc_associe.empty else "Non défini"

        pdc_associe = "Non défini"
        if df_pdc is not None:
            pdc_lien = df_pdc[df_pdc["id_rg"] == regle["id_rg"]]
            if not pdc_lien.empty:
                pdc_associe = pdc_lien["controle"].values[0]

        cas_de_test.append({
            "ID RG": regle["id_rg"],
            "Libellé RG": regle["libelle"],
            "Description CDC": description_cdc,
            "Contrôle (PDC)": pdc_associe
        })

    df_test = pd.DataFrame(cas_de_test)

    st.success("✅ Cas de test générés avec succès !")
    st.dataframe(df_test)

    # Export CSV
    csv = df_test.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les cas de test en CSV", data=csv, file_name="cas_de_test.csv", mime="text/csv")

else:
    st.warning("Veuillez charger au minimum le CDC et les Règles de Gestion pour générer les cas de test.")
