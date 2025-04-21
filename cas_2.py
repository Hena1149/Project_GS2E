from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
import streamlit as st
import re
import string
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pdfminer.high_level
import docx
from functools import lru_cache

# Configuration de l'application
st.set_page_config(page_title="Générateur de Cas de test à partir du CDC", layout="wide", page_icon="📑")

# ----------------------------
# FONCTIONS UTILITAIRES AMÉLIORÉES
# ----------------------------

@st.cache_resource
def load_nlp_model():
    """Charge le modèle spaCy pour le traitement NLP"""
    try:
        nlp = spacy.load("fr_core_news_md")
        st.success("Modèle NLP chargé avec succès !")
        return nlp
    except OSError:
        try:
            st.error("Modèle français non trouvé. Installation en cours...")
            import os
            os.system("python -m spacy download fr_core_news_md")
            nlp = spacy.load("fr_core_news_md")
            return nlp
        except Exception as e:
            st.error(f"Échec du chargement : {str(e)}")
            return None

def extract_text(uploaded_file):
    """Extrait le texte depuis PDF ou DOCX"""
    try:
        file_bytes = uploaded_file.getvalue()
        
        if uploaded_file.type == "application/pdf":
            with BytesIO(file_bytes) as f:
                text = pdfminer.high_level.extract_text(f)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with BytesIO(file_bytes) as f:
                doc = docx.Document(f)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            raise ValueError("Format non supporté")
            
        return text if text and text.strip() else None
        
    except Exception as e:
        st.error(f"Erreur d'extraction : {str(e)}")
        return None

def clean_rule_text(rule):
    """Nettoyage intelligent des règles"""
    # Suppression des numéros et puces
    rule = re.sub(r'^[\d\s•\-]*', '', rule)
    # Normalisation des espaces
    rule = re.sub(r'\s+', ' ', rule).strip()
    # Correction de la ponctuation
    if rule and not rule.endswith(('.', '!', '?')):
        rule += '.'
    # Capitalisation
    return rule[0].upper() + rule[1:] if rule else rule

@lru_cache(maxsize=1000)
def is_similar_rule(rule1, rule2, threshold=0.75):
    """Détection de similarité sémantique entre règles"""
    if 'nlp' not in st.session_state:
        st.session_state.nlp = load_nlp_model()
    doc1 = st.session_state.nlp(rule1)
    doc2 = st.session_state.nlp(rule2)
    return doc1.similarity(doc2) >= threshold

def is_potential_rule(sentence, nlp_model):
    """Détection avancée des règles potentielles"""
    if len(sentence.split()) < 6:
        return False
    
    doc = nlp_model(sentence)
    
    # Marqueurs positifs
    has_obligation = any(token.lemma_ in {'devoir', 'falloir', 'nécessiter'} for token in doc)
    has_validation = any(token.lemma_ in {'vérifier', 'contrôler', 'valider'} for token in doc)
    has_condition = any(token.text.lower() in {'si', 'lorsque', 'quand'} for token in doc)
    
    # Marqueurs négatifs
    is_question = any(token.tag_ == 'INTJ' for token in doc)
    is_example = any(token.text.lower() in {'exemple', 'comme'} for token in doc)
    
    return (has_obligation or has_validation or has_condition) and not (is_question or is_example)

def extract_business_rules(text, nlp_model, sensitivity=3):
    """Extraction complète des règles de gestion"""
    # Normalisation du texte
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\n\d+[.)])', r'\1 ', text)
    
    # Segmentation avancée
    sentences = []
    for paragraph in text.split('\n'):
        if len(paragraph.strip()) > 10:
            if nlp_model:
                doc = nlp_model(paragraph)
                sentences.extend([sent.text for sent in doc.sents])
            else:
                sentences.extend(re.split(r'(?<=[.!?])\s+', paragraph))
    
    # Extraction avec motifs étendus
    patterns = [
        r'(?i)((?:doit|devra|obligatoire|requis|vérifier|contrôler|si\b|alors\b).{8,}?[.!?])',
        r'(?i)\b(?:le système|l\'application)\b.{8,}?[.!?]',
        r'(?i)((?:lorsque|quand|dès que|en cas de).{8,}?(?:alors|donc|par conséquent).{8,}?[.!?])',
        r'(?i)(?:l\'utilisateur doit|il est nécessaire que).{8,}?[.!?]'
    ]
    
    rules = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        rules.update([clean_rule_text(m) for m in matches])
    
    # Analyse syntaxique complémentaire
    if nlp_model:
        for sent in sentences:
            if is_potential_rule(sent, nlp_model):
                rules.add(clean_rule_text(sent))
    
    # Filtrage adaptatif selon la sensibilité
    min_length = [12, 10, 8, 6, 4][min(sensitivity-1, 4)]
    rules = [r for r in rules if len(r.split()) >= min_length]
    
    # Dédoublonnage sémantique
    unique_rules = []
    for rule in sorted(rules, key=len, reverse=True):
        if not any(is_similar_rule(rule, existing) for existing in unique_rules):
            unique_rules.append(rule)
    
    return unique_rules[:200]  # Limite raisonnable

def generate_pdc_from_rule(rule):
    """Génération intelligente de PDC"""
    transformations = [
        (r'(?i)doit (.+?)\.', r'Vérifier que \1'),
        (r'(?i)il est obligatoire de (.+?)\.', r'Contrôler que \1'),
        (r'(?i)le système doit (.+?)\.', r'Tester que le système \1'),
        (r'(?i)si (.+?), alors (.+?)\.', r'Vérifier que lorsque \1, alors \2'),
        (r'(?i)l\'utilisateur peut (.+?)\.', r'Valider que l\'utilisateur peut \1')
    ]
    
    for pattern, replacement in transformations:
        if re.search(pattern, rule):
            pdc = re.sub(pattern, replacement, rule)
            return format_pdc(pdc)
    
    return f"Vérifier que {rule.lower().rstrip('.')}."

def format_pdc(text):
    """Formattage professionnel des PDC"""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text.endswith('.'):
        text += '.'
    return text[0].upper() + text[1:]

def create_test_case(pdc, index, is_manual=False):
    """Création de cas de test bien formulés"""
    test_types = {
        'vérifier': 'Validation',
        'contrôler': 'Contrôle',
        'tester': 'Test',
        'valider': 'Vérification'
    }
    
    # Détection du type de test
    first_word = pdc.split()[0].lower()
    test_type = test_types.get(first_word, 'Test')
    
    return {
        "ID": f"CT-{index:03d}",
        "Type": test_type,
        "PDC": pdc,
        "Description": generate_test_description(pdc),
        "Préconditions": "1. Environnement de test configuré\n2. Données de test disponibles",
        "Étapes": generate_test_steps(pdc),
        "Résultat attendu": generate_expected_result(pdc)
    }

def generate_test_description(pdc):
    """Génération automatique de descriptions cohérentes"""
    action_map = {
        'vérifier': "Vérification du bon fonctionnement de",
        'contrôler': "Contrôle de la conformité de",
        'tester': "Test d'implémentation de",
        'valider': "Validation du comportement de"
    }
    
    first_word = pdc.split()[0].lower()
    action = action_map.get(first_word, "Test de")
    return f"{action} {pdc[len(first_word)+1:].rstrip('.')}."

# ----------------------------
# INTERFACE STREAMLIT AMÉLIORÉE
# ----------------------------

st.title("📑 Générateur Automatique de Cas de Test")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Extraction", "🔍 Analyse", "☁️ WordCloud", "📜 Règles", "✅ PDC & Tests"])

with tab1:
    st.header("Extraction de Texte")
    uploaded_file = st.file_uploader("Téléversez un document (PDF ou DOCX)", type=["pdf", "docx"])
    
    if uploaded_file and st.button("Extraire le texte"):
        with st.spinner("Extraction en cours..."):
            extracted_text = extract_text(uploaded_file)
            
            if extracted_text:
                st.session_state.text = extracted_text
                st.success(f"Texte extrait ({len(extracted_text.split())} mots)")
                
                with st.expander("Aperçu du texte"):
                    st.text(extracted_text[:1500] + ("..." if len(extracted_text) > 1500 else ""))

with tab2:
    st.header("Analyse Textuelle Avancée")
    
    if 'text' not in st.session_state:
        st.warning("Veuillez d'abord extraire un texte")
    else:
        nlp_model = load_nlp_model()
        if not nlp_model:
            st.error("Modèle NLP non disponible")
        else:
            with st.spinner("Analyse linguistique en cours..."):
                st.session_state.text_clean = clean_text(st.session_state.text, nlp_model)
                st.session_state.freq = calculate_frequencies(st.session_state.text_clean)
            
            st.subheader("Mots-clés principaux")
            top_n = st.slider("Nombre de mots à afficher", 5, 50, 15)
            st.dataframe(st.session_state.freq.head(top_n))

with tab3:
    st.header("Visualisation des Concepts")
    
    if 'text_clean' not in st.session_state:
        st.warning("Veuillez d'abord analyser un texte")
    else:
        with st.expander("Paramètres avancés"):
            col1, col2 = st.columns(2)
            with col1:
                width = st.slider("Largeur", 400, 1200, 800)
                height = st.slider("Hauteur", 200, 800, 400)
            with col2:
                bg_color = st.color_picker("Couleur de fond", "#FFFFFF")
                colormap = st.selectbox("Palette", ["viridis", "plasma", "inferno", "magma", "cividis"])
        
        if st.button("Générer le WordCloud"):
            freq_dict = st.session_state.freq.to_dict()
            fig = generate_wordcloud(freq_dict, width, height, bg_color, colormap)
            st.pyplot(fig)

with tab4:
    st.header("Extraction des Règles Métier")
    
    if 'text' not in st.session_state:
        st.warning("Veuillez d'abord extraire un texte")
    else:
        nlp_model = load_nlp_model()
        sensitivity = st.slider("Niveau de détection", 1, 5, 3,
                              help="Augmentez pour détecter plus de règles (peut inclure des faux positifs)")
        
        if st.button("Extraire les règles", type="primary"):
            with st.spinner("Analyse approfondie en cours..."):
                rules = extract_business_rules(st.session_state.text, nlp_model, sensitivity)
                
                if rules:
                    st.session_state.rules = rules
                    st.success(f"{len(rules)} règles identifiées")
                    
                    # Affichage paginé
                    items_per_page = 5
                    total_pages = (len(rules) + items_per_page - 1) // items_per_page
                    page = st.number_input("Page", 1, total_pages, 1)
                    
                    start_idx = (page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, len(rules))
                    
                    for i in range(start_idx, end_idx):
                        st.markdown(f"**Règle {i+1}**")
                        st.info(rules[i])
                    
                    # Export
                    docx_file = create_rules_document(rules)
                    st.download_button(
                        "📄 Télécharger les règles",
                        data=docx_file,
                        file_name="regles_metier.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

with tab5:
    st.header("Génération des Tests")
    
    if 'rules' not in st.session_state:
        st.warning("Veuillez d'abord extraire les règles")
    else:
        # Section PDC
        st.subheader("Points de Contrôle")
        if st.button("Générer les PDC"):
            with st.spinner("Création des PDC..."):
                st.session_state.pdc_list = [generate_pdc_from_rule(r) for r in st.session_state.rules]
                st.success(f"{len(st.session_state.pdc_list)} PDC générés")
        
        if 'pdc_list' in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state.pdc_list, columns=["Points de Contrôle"]).head(10))
            
            # Section Cas de Test
            st.subheader("Cas de Test")
            if st.button("Générer les Tests"):
                with st.spinner("Construction des cas de test..."):
                    st.session_state.test_cases = [
                        create_test_case(pdc, i) 
                        for i, pdc in enumerate(st.session_state.pdc_list, 1)
                    ]
                    st.success(f"{len(st.session_state.test_cases)} cas de test créés")
            
            if 'test_cases' in st.session_state:
                st.dataframe(pd.DataFrame(st.session_state.test_cases))
                
                # Export complet
                st.download_button(
                    "📥 Exporter tous les tests (Excel)",
                    data=pd.DataFrame(st.session_state.test_cases).to_excel(),
                    file_name="cas_de_tests.xlsx",
                    mime="application/vnd.ms-excel"
                )

# ----------------------------
# FONCTIONS COMPLÉMENTAIRES
# ----------------------------

def clean_text(text, nlp_model, min_word_length=3):
    """Nettoyage approfondi du texte"""
    if not text or not nlp_model:
        return ""
    
    text = text.lower()
    text = re.sub(r"[^\w\sàâäéèêëîïôöùûüç]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    doc = nlp_model(text)
    cleaned_tokens = []
    
    for token in doc:
        if (token.is_stop or token.is_punct or 
            len(token.text) < min_word_length or
            token.pos_ in ["DET", "ADP", "CCONJ", "PRON"]):
            continue
        lemma = token.lemma_.strip()
        if lemma:
            cleaned_tokens.append(lemma)
    
    return " ".join(cleaned_tokens)

def calculate_frequencies(text):
    """Calcul des fréquences des mots"""
    words = [word for word in text.split() if len(word) > 2]
    return pd.Series(words).value_counts()

def generate_wordcloud(freq_dict, width=800, height=400, background_color="white", colormap="viridis"):
    """Génération du nuage de mots"""
    fig, ax = plt.subplots(figsize=(10, 5))
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=100
    ).generate_from_frequencies(freq_dict)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def create_rules_document(rules):
    """Création d'un document Word des règles"""
    doc = Document()
    doc.add_heading('Règles de Gestion Identifiées', level=1)
    for i, rule in enumerate(rules, 1):
        doc.add_paragraph(f"{i}. {rule}", style='ListBullet')
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def generate_test_steps(pdc):
    """Génération automatique des étapes de test"""
    action = pdc.split()[0].lower()
    target = ' '.join(pdc.split()[1:]).rstrip('.')
    
    steps = [
        f"1. Préparer l'environnement de test",
        f"2. Exécuter l'action: {action} {target}",
        f"3. Enregistrer les résultats observés"
    ]
    return '\n'.join(steps)

def generate_expected_result(pdc):
    """Génération du résultat attendu"""
    return f"La condition '{pdc.rstrip('.')}' est correctement respectée."

# ----------------------------
# PIED DE PAGE
# ----------------------------
st.markdown("---")
st.caption("© Outil de génération de tests - Version 2025")
