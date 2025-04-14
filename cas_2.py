#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pdfminer.high_level
import docx
import re
import spacy
import pandas as pd
from docx import Document
import random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Configuration de la page Streamlit
st.set_page_config(page_title="Génération Automatique de Cas de test", layout="wide")
st.title("📑 Génération Automatique de Cas de test")

# Téléchargements NLTK (un seul fois)
nltk.download('punkt')
nltk.download('stopwords')

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Paramètres")
    file_type = st.radio("Type de document", [".pdf", ".docx"])
    show_raw_text = st.checkbox("Afficher le texte brut")
    show_bow = st.checkbox("Afficher le Bag-of-Words")

# Section 1: Upload du document
uploaded_file = st.file_uploader("Déposez votre document (PDF ou DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    # Section 2: Extraction du texte
    st.header("1. Extraction du Texte")
    
    def extract_text_from_pdf(pdf_path):
        try:
            text = pdfminer.high_level.extract_text(pdf_path)
            if not text.strip():
                raise ValueError("Aucun texte trouvé dans le PDF")
            return fix_incomplete_lines(text)
        except Exception as e:
            st.error(f"Erreur PDF: {str(e)}")
            return None

    def extract_text_from_docx(docx_path):
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return fix_incomplete_lines(text)
        except Exception as e:
            st.error(f"Erreur DOCX: {str(e)}")
            return None

    def fix_incomplete_lines(text):
        lines = text.split("\n")
        fixed_lines = []
        for i in range(len(lines)):
            if lines[i].strip().lower().startswith(("si", "lorsqu'", "quand", "dès que", "en cas de")):
                if i + 1 < len(lines) and not lines[i].strip().endswith("."):
                    lines[i] += " " + lines[i + 1].strip()
                    lines[i + 1] = ""
            fixed_lines.append(lines[i])
        return "\n".join(fixed_lines)

    # Extraction en fonction du type de fichier
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_docx(uploaded_file)

    if text and show_raw_text:
        with st.expander("Voir le texte extrait"):
            st.text_area("Texte brut", text, height=300)

    # Section 3: Nettoyage et analyse
    if text:
        st.header("2. Analyse des Règles de Gestion")
        
        # Chargement du modèle spaCy
        try:
            nlp = spacy.load("fr_core_news_md")
        except:
            st.error("Modèle spaCy français non trouvé. Veuillez installer avec: python -m spacy download fr_core_news_md")
            st.stop()

        # Nettoyage du texte
        def nettoyer_texte(text):
            text = text.lower()
            text = re.sub(r"[\n\t\xa0«»\"']", " ", text)
            text = re.sub(r"\b[lLdDjJcCmM]'(\w+)", r"\1", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            
            doc = nlp(text)
            mots_nets = [
                token.lemma_ for token in doc
                if token.text not in stopwords.words('french')
                and len(token.lemma_) > 2
                and not token.is_digit
            ]
            return " ".join(mots_nets)

        text_clean = nettoyer_texte(text)

        if show_bow:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([text_clean])
            df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
            st.dataframe(df_bow.T.sort_values(by=0, ascending=False).head(10))

        # Section 4: Extraction des règles
        st.header("3. Règles de Gestion Identifiées")
        
        def extract_rules(text):
            patterns = [
                r"(Si|Lorsqu'|Quand|Dès que|En cas de).*?(alors|doit|devra|est tenu de|nécessite|implique|entraîne|peut).*?\.",
                r"(Tout utilisateur|L'[a-zA-Z]+|Un client|Le système|Une demande).*?(doit|est tenu de|devra|ne peut pas|ne doit pas|est interdit de).*?\.",
                r"(Le non-respect|Toute infraction|Une violation).*?(entraîne|provoque|peut entraîner|résulte en|sera soumis à).*?\."
            ]
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, text, re.IGNORECASE))
            return [" ".join(match).capitalize() for match in matches]

        rules = extract_rules(text)
        rules = list(set(rules))  # Supprimer les doublons
        
        if rules:
            st.success(f"{len(rules)} règles identifiées")
            for i, rule in enumerate(rules, 1):
                st.markdown(f"{i}. {rule}")
            
            # Section 5: Génération des PDC et Cas de Test
            st.header("4. Génération Automatique")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Générer les Points de Contrôle"):
                    pdc_list = [f"Vérifier que {rule}" for rule in rules]
                    st.session_state.pdc = pdc_list
                    
            with col2:
                if st.button("Générer les Cas de Test"):
                    test_cases = []
                    for i, rule in enumerate(rules, 1):
                        test_case = {
                            "ID": f"CT-{i:03d}",
                            "Description": f"Vérification: {rule}",
                            "Résultat Attendu": f"Le système doit respecter: {rule}"
                        }
                        test_cases.append(test_case)
                    st.session_state.test_cases = test_cases
            
            # Affichage des résultats
            if 'pdc' in st.session_state:
                with st.expander("Points de Contrôle Générés"):
                    for i, pdc in enumerate(st.session_state.pdc, 1):
                        st.markdown(f"{i}. {pdc}")
                    
                    st.download_button(
                        label="Télécharger les PDC (DOCX)",
                        data=generate_docx(st.session_state.pdc, "Points de Contrôle"),
                        file_name="points_de_controle.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            
            if 'test_cases' in st.session_state:
                with st.expander("Cas de Test Générés"):
                    df = pd.DataFrame(st.session_state.test_cases)
                    st.dataframe(df)
                    
                    st.download_button(
                        label="Télécharger les Cas de Test (DOCX)",
                        data=generate_docx([f"{tc['ID']}: {tc['Description']}" for tc in st.session_state.test_cases], "Cas de Test"),
                        file_name="cas_de_test.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

        else:
            st.warning("Aucune règle de gestion identifiée dans le document")

def generate_docx(content, title):
    """Génère un document Word à partir du contenu"""
    doc = Document()
    doc.add_heading(title, level=1)
    for item in content:
        doc.add_paragraph(item)
    return doc
