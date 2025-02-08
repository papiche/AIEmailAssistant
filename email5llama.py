import imaplib
import email
from email.header import decode_header
import requests
import logging
from logging.handlers import RotatingFileHandler
import sys
import traceback
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import time
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from queue import Queue
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('log.txt', maxBytes=10000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Vérification des variables d'environnement
required_env_vars = ['IMAP_SERVER', 'SMTP_SERVER', 'SMTP_PORT', 'EMAIL', 'PASSWORD', 'OLLAMA_API_URL', 'MODEL', 'CONTEXT']
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"La variable d'environnement {var} n'est pas définie")
        sys.exit(1)

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
MODEL = os.getenv("MODEL")

# Cache pour les embeddings
EMBEDDING_CACHE = {}
CACHE_FILE = "embedding_cache.json"

class EmailAssistantUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Email Assistant")
        self.geometry("1000x800")

        # Panneau de contrôle
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = ttk.Button(self.control_frame, text="Démarrer", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(self.control_frame, text="Arrêter", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar(value=MODEL)
        self.model_selector = ttk.Combobox(self.control_frame, textvariable=self.model_var, state="readonly")
        self.model_selector['values'] = ["llama2", "mistral", "custom_model"]  # Exemples de modèles
        self.model_selector.pack(side=tk.LEFT, padx=5)

        # Zone de texte pour afficher les logs
        self.log_area = scrolledtext.ScrolledText(self, state='disabled', height=10)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Zone de texte pour afficher le contexte envoyé au LLM
        self.context_label = ttk.Label(self, text="Contexte envoyé au LLM")
        self.context_label.pack(padx=10, pady=(10, 0), anchor=tk.W)

        self.context_area = scrolledtext.ScrolledText(self, state='normal', height=10)
        self.context_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Zone de texte pour afficher les résultats du traitement RAG
        self.rag_label = ttk.Label(self, text="Résultats du traitement RAG")
        self.rag_label.pack(padx=10, pady=(10, 0), anchor=tk.W)

        self.rag_area = scrolledtext.ScrolledText(self, state='normal', height=10)
        self.rag_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # File d'attente pour le traitement des emails
        self.email_queue = Queue()
        self.processing_thread = None
        self.running = False

    def start_processing(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.processing_thread = Thread(target=self.process_emails)
        self.processing_thread.start()

    def stop_processing(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def process_emails(self):
       while self.running:
            try:
                for sujet, contenu, email_message in lire_emails(os.getenv("IMAP_SERVER"), os.getenv("EMAIL"), os.getenv("PASSWORD")):
                    expediteur = email.utils.parseaddr(email_message['From'])[1]
                    original_message_id = email_message['Message-ID']
                    self.log(f"Traitement de l'email {original_message_id} de {expediteur} avec le sujet: {sujet}")

                    # Générer la réponse avec affichage du contexte et des résultats RAG
                    reponse_generee, contexte_llm, resultats_rag = generer_reponse_avec_details(expediteur, sujet, contenu, self.model_var.get())
                    self.log(f"Réponse générée : {reponse_generee}")

                    # Afficher le contexte et les résultats RAG dans l'interface
                    self.afficher_contexte_llm(contexte_llm)
                    self.afficher_resultats_rag(resultats_rag)

                    sauvegarder_brouillon(os.getenv("IMAP_SERVER"), os.getenv("EMAIL"), os.getenv("PASSWORD"), expediteur, sujet, reponse_generee, original_message_id)
                    self.log(f"Brouillon sauvegardé pour {expediteur}")
            except Exception as e:
                self.log(f"Erreur lors du traitement des emails : {str(e)}", level="error")

    def afficher_contexte_llm(self, contexte):
        self.context_area.config(state='normal')
        self.context_area.delete(1.0, tk.END)
        self.context_area.insert(tk.END, contexte)
        self.context_area.config(state='disabled')

    def afficher_resultats_rag(self, resultats):
        self.rag_area.config(state='normal')
        self.rag_area.delete(1.0, tk.END)
        self.rag_area.insert(tk.END, resultats)
        self.rag_area.config(state='disabled')

    def log(self, message, level="info"):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, f"{message}\n")
        self.log_area.config(state='disabled')
        self.log_area.yview(tk.END)
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)

# Fonctions principales
def lire_emails(imap_server, email_address, password):
    logger.info(f"Tentative de connexion au serveur IMAP: {imap_server}")
    try:
        imap = imaplib.IMAP4_SSL(imap_server)
        logger.info("Connexion SSL établie, tentative de login")
        imap.login(email_address, password)
        logger.info("Login réussi")

        # Sélectionner la boîte de réception
        status, messages = imap.select("INBOX")
        if status != "OK":
            logger.error(f"Impossible de sélectionner la boîte de réception: {messages}")
            return

        logger.info("Boîte de réception sélectionnée avec succès")

        _, message_numbers = imap.search(None, "UNSEEN")

        for num in message_numbers[0].split():
            _, msg_data = imap.fetch(num, "(RFC822)")
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)

            sujet = decode_header(email_message["Subject"])[0][0]
            if isinstance(sujet, bytes):
                sujet = sujet.decode()

            contenu = ""
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        contenu = part.get_payload(decode=True).decode()
            else:
                contenu = email_message.get_payload(decode=True).decode()

            yield sujet, contenu, email_message

        imap.close()
        imap.logout()
    except imaplib.IMAP4.error as e:
        logger.error(f"Erreur IMAP lors de la lecture des emails: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la lecture des emails: {str(e)}")
        logger.error(traceback.format_exc())

def charger_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def sauvegarder_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def generer_embedding(texte):
    if texte in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[texte]

    embedding_data = {
        "model": MODEL,
        "prompt": texte
    }
    response = requests.post(f"{OLLAMA_API_URL}/api/embeddings", json=embedding_data)
    embedding = response.json().get("embedding")
    if not embedding:
        raise ValueError(f"Embedding vide généré pour le texte: {texte[:50]}...")

    EMBEDDING_CACHE[texte] = embedding
    sauvegarder_cache(EMBEDDING_CACHE)
    return embedding

def generer_reponse(expediteur, sujet, contenu, model_name):
     try:
        contenu_nettoye = "\n".join([ligne for ligne in contenu.split("\n") if not ligne.strip().startswith(">")])
        email_embedding = generer_embedding(contenu_nettoye)

        dataset_embeddings = charger_dataset_embeddings(expediteur)
        logger.info(f"Dataset chargé avec {len(dataset_embeddings)} entrées pour l'expéditeur: {expediteur}")

        if not dataset_embeddings:
            logger.info(f"Pas d'exemples d'emails trouvés pour l'expéditeur {expediteur}. Utilisation d'un prompt par défaut.")
            contexte_global = charger_contexte_global()
            prompt = f"[CONTEXTE] :\n{contexte_global}\n\n[EMAIL ACTUEL]:\nExpéditeur: {expediteur}\nSujet: {sujet}\nContenu: {contenu_nettoye}\n\nRéponse:"
        else:
            similarites = [cosine_similarity([email_embedding], [item["input_embedding"]])[0][0] for item in dataset_embeddings]
            indices_tries = np.argsort(similarites)[::-1][:2]
            exemples_similaires = [dataset_embeddings[i]["output"] for i in indices_tries]

            contexte_global = charger_contexte_global()
            prompt = f"[CONTEXTE] :\n{contexte_global}\n\n[SUGGESTIONS]:\n"
            for exemple in exemples_similaires:
                prompt += f"{exemple}\n\n"

            prompt += f"[EMAIL ACTUEL]:\nExpéditeur: {expediteur}\nSujet: {sujet}\nContenu: {contenu_nettoye}\n\nRéponse:"

        generate_data = {
            "model": model_name,
            "prompt": prompt,
            "system": f"Tu es ASTRO, un assistant intelligent qui lit et répond aux messages de la boite email de {expediteur}.",
            "stream": False
        }

        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=generate_data)
        response_json = response.json()
        return response_json['response']
     except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse: {str(e)}")
        logger.error(traceback.format_exc())
        return "Désolé, une erreur s'est produite lors de la génération de la réponse."

def generer_reponse_avec_details(expediteur, sujet, contenu, model_name):
    try:
        contenu_nettoye = "\n".join([ligne for ligne in contenu.split("\n") if not ligne.strip().startswith(">")])
        email_embedding = generer_embedding(contenu_nettoye)

        dataset_embeddings = charger_dataset_embeddings(expediteur)
        logger.info(f"Dataset chargé avec {len(dataset_embeddings)} entrées pour l'expéditeur: {expediteur}")

        # Préparer le contexte pour le LLM et le RAG
        contexte_global = charger_contexte_global()
        contexte_llm = f"[CONTEXTE] :\n{contexte_global}\n\n"
        resultats_rag = "Résultats du traitement RAG :\n"


        if not dataset_embeddings:
             logger.info(f"Pas d'exemples d'emails trouvés pour l'expéditeur {expediteur}. Utilisation d'un prompt par défaut.")
             contexte_llm += f"[EMAIL ACTUEL]:\nExpéditeur: {expediteur}\nSujet: {sujet}\nContenu: {contenu_nettoye}\n\nRéponse:"
             resultats_rag += "Aucun exemple RAG trouvé pour cet expéditeur.\n"
        else:
            # Calcul des similarités
            similarites = [cosine_similarity([email_embedding], [item["input_embedding"]])[0][0] for item in dataset_embeddings]
            indices_tries = np.argsort(similarites)[::-1][:2]  # Prendre les 2 plus similaires
            exemples_similaires = [dataset_embeddings[i]["output"] for i in indices_tries]

            contexte_llm += "[SUGGESTIONS]:\n"
            for exemple in exemples_similaires:
               contexte_llm += f"{exemple}\n\n"
            contexte_llm += f"[EMAIL ACTUEL]:\nExpéditeur: {expediteur}\nSujet: {sujet}\nContenu: {contenu_nettoye}\n\nRéponse:"

            # Préparer les résultats du RAG pour l'affichage
            for i, idx in enumerate(indices_tries):
                resultats_rag += f"Exemple {i+1} (similarité: {similarites[idx]:.2f}):\n{dataset_embeddings[idx]['input']}\n\n"


        # Générer la réponse
        generate_data = {
            "model": model_name,
            "prompt": contexte_llm,
            "system": f"Tu es ASTRO, un assistant intelligent qui lit et répond aux messages de la boite email de {expediteur}.",
            "stream": False
        }

        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=generate_data)
        response_json = response.json()

        return response_json['response'], contexte_llm, resultats_rag
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse: {str(e)}")
        logger.error(traceback.format_exc())
        return "Désolé, une erreur s'est produite lors de la génération de la réponse.", "", ""

def sauvegarder_brouillon(imap_server, email_address, password, recipient, subject, body, original_message_id):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            imap = imaplib.IMAP4_SSL(imap_server)
            imap.login(email_address, password)

            msg = MIMEMultipart()
            msg['From'] = email_address
            msg['To'] = recipient
            msg['Subject'] = f"Re: {subject}"
            msg['In-Reply-To'] = original_message_id
            msg['References'] = original_message_id
            msg['X-Original-Message-ID'] = original_message_id
            msg.attach(MIMEText(body, 'plain'))

            draft_folder = os.getenv('DRAFTDIR', 'INBOX.Drafts')

            # Sélectionner le dossier brouillons
            status, _ = imap.select(f'"{draft_folder}"')
            if status != 'OK':
                logger.error(f"Impossible de sélectionner le dossier INBOX.Drafts: {status}")
                imap.logout()
                continue

            # Ajouter le brouillon
            status, _ = imap.append(f'"{draft_folder}"', '\\Draft', imaplib.Time2Internaldate(time.time()), msg.as_bytes())
            if status == 'OK':
                logger.info(f"Brouillon sauvegardé pour {recipient}")
                imap.logout()
                return
            else:
                logger.error(f"Échec de la sauvegarde du brouillon: {status}")

            imap.logout()
        except Exception as e:
            logger.error(f"Tentative {attempt+1}/{max_retries} - Erreur lors de la sauvegarde du brouillon: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Échec de la sauvegarde du brouillon après plusieurs tentatives")
            time.sleep(2)  # Attendre 2 secondes avant de réessayer

def charger_contexte_global():
    CONTEXT = os.getenv("CONTEXT")
    if not os.path.exists(CONTEXT):
        with open(CONTEXT, 'w') as f:
            f.write("Contexte global initial")

    with open(CONTEXT, 'r') as f:
        contexte_global = f.read()

    return contexte_global

def charger_dataset_embeddings(email):
    if not os.path.exists("email_dataset.json"):
        return []

    with open("email_dataset.json", "r") as f:
        dataset = json.load(f)

    dataset_embeddings = []
    for item in dataset:
        if email == "" or item.get("email") == email:
            input_embedding = generer_embedding(item["input"])
            output_embedding = generer_embedding(item["output"])
            dataset_embeddings.append({
                "input": item["input"],
                "output": item["output"],
                "input_embedding": input_embedding,
                "output_embedding": output_embedding
            })

    return dataset_embeddings

def extraire_dataset():
    cache = charger_cache()
    dataset = []
    imap = imaplib.IMAP4_SSL(os.getenv("IMAP_SERVER"))
    imap.login(os.getenv("EMAIL"), os.getenv("PASSWORD"))

    status, messages = imap.select("INBOX")
    if status != "OK":
        logger.error(f"Impossible de sélectionner le dossier INBOX: {messages}")
        imap.logout()
        return

    status, message_numbers = imap.search(None, "ALL")
    if status != "OK":
        logger.error(f"Erreur lors de la recherche des messages: {message_numbers}")
        imap.logout()
        return
    try:
        for num in message_numbers[0].split():
            logger.info(f"Processing email number: {num}")
            max_retries = 3
            msg_data = None
            for attempt in range(max_retries):
                try:
                    status, data = imap.fetch(num, "UID")
                    if status != "OK":
                        logger.error(f"Failed to fetch UID for email {num}: {data}")
                        break

                    uid_bytes = data[0].split()[-1]
                    uid = uid_bytes.decode()

                    status, msg_data = imap.uid('fetch', uid, 'RFC822')  # Pass as an argument
                    if status == "OK":
                        break
                    else:
                        logger.error(f"Failed to fetch email {uid} on attempt {attempt + 1}: {msg_data}")
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to fetch email {uid} after several attempts.")
                            break
                        time.sleep(2)
                except Exception as e:
                   logger.error(f"Error fetching email {num}: {e}")
                   logger.error(traceback.format_exc())
                   if attempt == max_retries-1:
                       logger.error(f"Skipping email {num} after several failed attempts")
                       break
                   time.sleep(2)

            if status != 'OK' or msg_data is None:
                 continue

            try:
              email_recu = email.message_from_bytes(msg_data[0][1])
              id_message = email_recu["Message-ID"]

              if id_message is None:
                  logger.warning(f"Message-ID manquant pour le message {num}, impossible de chercher la réponse")
                  continue

              if id_message in cache:
                  dataset.append(cache[id_message])
                  logger.info(f"Utilisation du cache pour le message {id_message}")
                  continue

              contenu_recu = get_email_content(email_recu)
              contenu_reponse = chercher_reponse(imap, id_message)

              if contenu_reponse:
                  item = {
                      "email": email.utils.parseaddr(email_recu["From"])[1],
                      "input": contenu_recu,
                      "output": contenu_reponse
                  }
                  dataset.append(item)
                  cache[id_message] = item
                  logger.info(f"Nouvelle paire d'emails ajoutée au dataset et au cache")
            except Exception as e:
                logger.error(f"Error processing email {num}: {e}")
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error in extraction loop: {e}")
        logger.error(traceback.format_exc())

    imap.logout()
    sauvegarder_cache(cache)

    with open("email_dataset.json", "w") as f:
        json.dump(dataset, f)

    logger.info(f"Dataset créé avec {len(dataset)} paires d'emails")

def get_email_content(email_message):
    """
    Extrait le contenu texte d'un email.
    """
    if email_message.is_multipart():
        for part in email_message.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode()
    else:
        return email_message.get_payload(decode=True).decode()


def chercher_reponse(imap, id_message):
    if id_message is None:
        logger.warning("Message-ID est None, impossible de chercher la réponse")
        return None

    sent_folder = os.getenv('SENTDIR', 'INBOX.Sent')
    status, _ = imap.select(f'"{sent_folder}"')
    if status != "OK":
        logger.error("Impossible de sélectionner le dossier Sent Mail")
        return None

    cleaned_id = id_message.strip().replace('"', '\\"')
    search_criteria = f'(OR (HEADER "In-Reply-To" "{cleaned_id}") (HEADER "X-Original-Message-ID" "{cleaned_id}"))'
    status, response_numbers = imap.search(None, search_criteria)

    if status != "OK" or not response_numbers[0]:
        return None

    status, response_data = imap.fetch(response_numbers[0].split()[-1], "(RFC822)")
    if status != "OK":
        return None

    email_reponse = email.message_from_bytes(response_data[0][1])
    return get_email_content(email_reponse)

if __name__ == "__main__":
    try:
        logger.info("Démarrage du processus de traitement des emails")
        extraire_dataset()
        app = EmailAssistantUI()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Processus interrompu par l'utilisateur")
    except Exception as e:
        logger.critical(f"Erreur critique: {str(e)}")
        logger.critical(traceback.format_exc())
