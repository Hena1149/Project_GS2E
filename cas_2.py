#!/usr/bin/env python
# coding: utf-8

# # **Étape 1 : Extraction du texte du CDC**

# In[1]:


get_ipython().system(' pip install pdfminer.six python-docx')


# In[3]:


import pdfminer.high_level
import docx

def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un fichier PDF"""
    try:
        text = pdfminer.high_level.extract_text(pdf_path)
        if text is None:
            raise ValueError("Le texte n'a pas pu être extrait du PDF")
        return text
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction du texte du PDF: {str(e)}")

def extract_text_from_docx(docx_path):
    """Extrait le texte d'un fichier Word"""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction du texte du DOCX: {str(e)}")

def extract_text(file_path):
    """Détecte le type de fichier et extrait son texte"""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Format de fichier non supporté. Utilise un fichier .pdf ou .docx")

file_path = r"C:\Users\yao.abo\Desktop\Cas_Test_01\Livrable1 - Résiliation HT  v1 (2).pdf"
try:
    text = extract_text(file_path)
    print(text)
except ValueError as e:
    print(f"Erreur: {e}")


# In[4]:


get_ipython().system(' pip install nltk')


# In[5]:


get_ipython().system(' pip install transformers')


# In[6]:


get_ipython().system('python -m spacy download fr_core_news_md')


# In[7]:


import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline


# In[8]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')


# # **Tokenisation et Lemmatisation avec spaCy & Bag of Word**

# In[9]:


import spacy
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd

nltk.download('stopwords')

nlp = spacy.load("fr_core_news_md")

stopwords_fr = set(stopwords.words('french'))

mots_inutiles = {"les", "des", "aux", "une", "dans", "sur", "par", "avec", "pour", "ce", "ces", "ses", "leur", "leurs",
                 "sous", "comme", "plus", "tous", "tout", "sans", "non", "peu", "donc", "ainsi", "même", "alors", "or"}

def nettoyer_texte(text):
    """Nettoyage et prétraitement du texte"""
    text = text.lower()
    text = re.sub(r"[\n\t\xa0«»\"']", " ", text)
    text = re.sub(r"\b[lLdDjJcCmM]'(\w+)", r"\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    doc = nlp(text)

    mots_nets = [
        token.lemma_ for token in doc
        if token.text not in stopwords_fr
        and token.lemma_ not in mots_inutiles
        and len(token.lemma_) > 2
        and not token.is_digit
    ]

    return " ".join(mots_nets)


text_clean = nettoyer_texte(text)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform([text_clean])

df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=["Texte"])

freq = df_bow.loc["Texte"].sort_values(ascending=False)

print("\n Mots les plus fréquents dans le texte :\n", freq.head(10))


# In[10]:


import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


# In[11]:


nltk.download('punkt_tab')


# # **Extraction des règles de gestion**

# In[13]:


import pdfminer.high_level
import docx
import re
import spacy
import pandas as pd

# Charger le modèle NLP en français
nlp = spacy.load("fr_core_news_md")

def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un fichier PDF"""
    try:
        text = pdfminer.high_level.extract_text(pdf_path)
        if not text.strip():
            raise ValueError("Le texte n'a pas pu être extrait du PDF")
        return fix_incomplete_lines(text)
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction du texte du PDF: {str(e)}")

def extract_text_from_docx(docx_path):
    """Extrait le texte d'un fichier Word"""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return fix_incomplete_lines(text)
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction du texte du DOCX: {str(e)}")

def extract_text(file_path):
    """Détecte le type de fichier et extrait son texte"""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Format de fichier non supporté. Utilise un fichier .pdf ou .docx")

def fix_incomplete_lines(text):
    """Corrige les phrases de conditions coupées en fusionnant les lignes incomplètes."""
    lines = text.split("\n")
    fixed_lines = []

    for i in range(len(lines)):
        if lines[i].strip().lower().startswith(("si", "lorsqu’", "quand", "dès que", "en cas de")):
            if i + 1 < len(lines) and not lines[i].strip().endswith("."):
                lines[i] += " " + lines[i + 1].strip()
                lines[i + 1] = ""
        fixed_lines.append(lines[i])

    return "\n".join(fixed_lines)

def extract_rules_with_regex(text):
    """Extraction des règles de gestion avec Regex"""

    patterns = [
        r"(Si|Lorsqu’|Quand|Dès que|En cas de).*?(alors|doit|devra|est tenu de|nécessite|implique|entraîne|peut).*?\.",
        r"(Tout utilisateur|L’[a-zA-Z]+|Un client|Le système|Une demande).*?(doit|est tenu de|devra|ne peut pas|ne doit pas|est interdit de).*?\.",
        r"(Le non-respect|Toute infraction|Une violation).*?(entraîne|provoque|peut entraîner|résulte en|sera soumis à).*?\.",
        r"(L’utilisateur|Le client|Le prestataire|L’agent|Le système).*?(est autorisé à|peut|a le droit de).*?\."
    ]

    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend([" ".join(match) for match in found])

    return [clean_text(rule) for rule in matches]

def extract_rules_with_nlp(text):
    """Extraction des règles de gestion avec NLP (spaCy)"""
    doc = nlp(text)
    rules = []

    for sent in doc.sents:
        sent_text = sent.text.strip()

        if any(keyword in sent_text.lower() for keyword in [
            "si ", "alors", "doit", "est tenu de", "ne peut pas", "entraîne", "provoque",
            "peut entraîner", "doit être", "est obligatoire", "a le droit de", "est autorisé à"
        ]):
            rules.append(clean_text(sent_text))

    return rules

def clean_text(text):
    """Nettoie le texte en supprimant les caractères spéciaux et les espaces inutiles"""
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.,'’ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("."):
        text += "."
    return text

def clean_short_rules(rules):
    """Supprime les fausses règles trop courtes"""
    return [rule for rule in rules if len(rule.split()) > 3]

def number_rules(rules):
    """Numérote les règles de gestion"""
    return [f"{rule}" for i, rule in enumerate(rules)]

#def save_to_excel(rules, file_name="regles_gestion.xlsx"):
#    """Enregistre les règles de gestion dans un fichier Excel"""
#    df = pd.DataFrame({"Règle de Gestion": rules})
#    df.to_excel(file_name, index=False)
#    print(f"Fichier Excel '{file_name}' créé avec succès !")

def save_to_docx(rules, file_name="regles_gestion.docx"):
    """Enregistre les règles de gestion dans un fichier Word"""
    doc = docx.Document()
    doc.add_heading("Règles de Gestion", level=1)

    for rule in rules:
        doc.add_paragraph(rule)

    doc.save(file_name)
    print(f"Fichier Word '{file_name}' créé avec succès !")

file_path = r"C:\Users\yao.abo\Desktop\Cas_Test_01\Livrable1 - Résiliation HT  v1 (2).pdf"

try:
    text = extract_text(file_path)
    #print("Texte extrait :\n", text[:500])

    rules_regex = extract_rules_with_regex(text)
    rules_nlp = extract_rules_with_nlp(text)

    all_rules = list(set(rules_regex + rules_nlp))
    all_rules = clean_short_rules(all_rules)
    all_rules = number_rules(all_rules)

#    save_to_excel(all_rules)  # Sauvegarde dans Excel
    save_to_docx(all_rules)   # Sauvegarde dans Word

except ValueError as e:
    print(f"Erreur: {e}")


# # **Générer des PDC pour les règles de gestion**

# In[14]:


from docx import Document
import spacy

# Chargement du modèle NLP français
nlp = spacy.load("fr_core_news_md")

def generate_pdc_for_rule(rule):
    """
    Génère un PDC basé uniquement sur une règle de gestion.
    """
    doc = nlp(rule)

    if "doit" in rule.lower():
        return f"Vérifier que {rule} est respecté(e)."

    if rule.lower().startswith("si"):
        return f"Vérifier la condition suivante : {rule}."

    return f"Vérifier que {rule}."

def save_pdc_to_docx(generated_pdc, filename="PDC_final_Test.docx"):
    """
    Sauvegarde les PDC générés dans un fichier Word avec une numérotation continue.
    """
    doc = Document()
    doc.add_heading('Points de Contrôle (PDC) Générés', level=1)

    for i, pdc in enumerate(generated_pdc, start=1):
        doc.add_paragraph(f"{i}. {pdc}")

    doc.save(filename)
    print(f"Le fichier {filename} a été sauvegardé avec succès.")

def generate_pdc_from_rules(rules_cdc):
    """
    Génère des PDC uniquement à partir des règles du CDC.
    """
    return [generate_pdc_for_rule(rule) for rule in rules_cdc]

rules_cdc = all_rules

generated_pdc = generate_pdc_from_rules(rules_cdc)

save_pdc_to_docx(generated_pdc)


# # **Génération des Cas de Test à partir des règles de gestion**

# In[16]:


from docx import Document
import spacy
import random

nlp = spacy.load("fr_core_news_md")

EXPECTED_RESULT_TEMPLATES = [
    "Le système doit assurer que {}.",
    "L'application de cette règle doit entraîner {}.",
    "Une vérification doit permettre de constater que {}.",
    "Le comportement attendu est que {}.",
    "Le processus doit respecter la règle suivante : {}."
]

def reformulate_expected_result(rule):
    """
    Reformule le résultat attendu en utilisant une phrase plus naturelle et différente de la description.
    """
    template = random.choice(EXPECTED_RESULT_TEMPLATES)
    return template.format(rule)

def generate_pdc_for_rule(rule):
    """
    Génère un PDC pour une règle donnée.
    """
    return f"Vérifier que {rule}."

def generate_test_case(rule, index):
    """
    Génère un cas de test basé sur une règle de gestion ou un PDC généré.
    """
    description = f"Vérifier que {rule}"
    preconditions = "Aucune précondition spécifique."
    steps = f"1. Vérifier que {rule}\n2. Observer le comportement du système."
    expected_result = reformulate_expected_result(rule)

    return [f"CT-{index:03d}", description, preconditions, steps, expected_result, "✅ / ❌"]

def save_test_cases_to_docx(test_cases, filename="Cas_Test_Gestion_PDC.docx"):
    """
    Sauvegarde les cas de test dans un document Word.
    """
    doc = Document()
    doc.add_heading('Cas de Test Automatiquement Générés', level=1)

    table = doc.add_table(rows=1, cols=6)
    table.style = 'Table Grid'

    headers = ["ID", "Description", "Préconditions", "Étapes", "Résultat Attendu", "Statut"]
    for i, header in enumerate(headers):
        table.cell(0, i).text = header

    for test_case in test_cases:
        row = table.add_row().cells
        for i, value in enumerate(test_case):
            row[i].text = value

    doc.save(filename)
    print(f"Fichier {filename} généré avec succès !")

def generate_test_cases(rules_cdc):
    """
    Génère des cas de test uniquement pour les règles de gestion et les PDC générés.
    """
    test_cases = []

    # Générer des cas de test pour les règles de gestion
    for i, rule in enumerate(rules_cdc, start=1):
        test_cases.append(generate_test_case(rule, i))

    # Générer les PDC à partir des règles de gestion et les utiliser pour des cas de test
    pdc_generated = [generate_pdc_for_rule(rule) for rule in rules_cdc]
    for j, pdc in enumerate(pdc_generated, start=len(rules_cdc) + 1):
        test_cases.append(generate_test_case(pdc, j))

    return test_cases

test_cases = generate_test_cases(rules_cdc)
save_test_cases_to_docx(test_cases)


# In[ ]:


get_ipython().system(' jupyter nbconvert --to script cas_2.ipynb')

