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
    if os.path.exists('./email_cache.json'):
        with open('./email_cache.json', 'r') as f:
            return json.load(f)
    return {}


def sauvegarder_cache(cache):
    with open('./email_cache.json', 'w') as f:
        json.dump(cache, f)


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

    for num in message_numbers[0].split():
        status, msg_data = imap.fetch(num, "(RFC822)")
        if status != "OK":
            continue

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


def creer_fichier_contextuel(email_address, reponse_generee, contenu):
    try:
        chemin_dossier = os.path.join("./emails/", email_address)
        if not os.path.exists(chemin_dossier):
            os.makedirs(chemin_dossier)

        with open(os.path.join(chemin_dossier, "context.txt"), 'a') as file:
            file.write("Contenu de l'email : \n")
            file.write(contenu)
            file.write("\n\nRéponse générée : \n")
            file.write(reponse_generee)
        logger.info(f"Fichier contextuel créé pour {email_address}")
    except Exception as e:
        logger.error(f"Erreur lors de la création du fichier contextuel : {str(e)}")


def charger_dataset_embeddings(email):
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


def generer_embedding(texte):
    embedding_data = {
        "model": MODEL,
        "prompt": texte
    }
    response = requests.post(f"{OLLAMA_API_URL}/api/embeddings", json=embedding_data)
    embedding = response.json().get("embedding")
    if not embedding:
        raise ValueError(f"Embedding vide généré pour le texte: {texte[:50]}...")
    return embedding


def stocker_embedding(expediteur, contenu, embedding):
    chemin_dossier = os.path.join("./emails/", expediteur)
    if not os.path.exists(chemin_dossier):
        os.makedirs(chemin_dossier)

    fichier_embeddings = os.path.join(chemin_dossier, "embeddings.json")

    try:
        with open(fichier_embeddings, 'r') as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        embeddings = []

    embeddings.append({"contenu": contenu, "embedding": embedding})

    with open(fichier_embeddings, 'w') as f:
        json.dump(embeddings, f)


def charger_contexte_global():
    CONTEXT = os.getenv("CONTEXT")
    if not os.path.exists(CONTEXT):
        with open(CONTEXT, 'w') as f:
            f.write("Contexte global initial")

    with open(CONTEXT, 'r') as f:
        contexte_global = f.read()

    return contexte_global


def ajouter_au_contexte_global(nouveau_contenu):
    CONTEXT = os.getenv("CONTEXT")
    with open(CONTEXT, 'a') as f:
        f.write(f"\n\n{nouveau_contenu}")

def charger_contextes_conversation():
    if os.path.exists('emails/conversation_context.json'):
        with open('emails/conversation_context.json', 'r') as f:
            return json.load(f)
    return {}

def sauvegarder_contextes_conversation(contextes):
    with open('emails/conversation_context.json', 'w') as f:
        json.dump(contextes, f)


def obtenir_contexte_conversation(expediteur):
    contextes = charger_contextes_conversation()
    return contextes.get(expediteur, [])


def ajouter_au_contexte_conversation(expediteur, email_contenu, reponse_contenu):
    contextes = charger_contextes_conversation()
    if expediteur not in contextes:
        contextes[expediteur] = []
    contexte_expediteur = contextes[expediteur]
    contexte_expediteur.append({"email": email_contenu, "reponse": reponse_contenu})

    # Garder seulement les 3 derniers échanges
    contextes[expediteur] = contexte_expediteur[-3:]
    sauvegarder_contextes_conversation(contextes)

def generer_reponse(expediteur, sujet, contenu, model_name):
    try:
        # Prétraitement du contenu pour enlever les lignes commençant par "> "
        contenu_nettoye = "\n".join([ligne for ligne in contenu.split("\n") if not ligne.strip().startswith(">")])

        # Générer l'embedding pour le contenu nettoyé de l'email actuel
        email_embedding = generer_embedding(contenu_nettoye)

        # chargement du dataset global pour recherche de similarité de input
        dataset_embeddings = charger_dataset_embeddings("")
        logger.info(f"Dataset GLOBAL chargé pour avec {len(dataset_embeddings)} entrées")

        # Trouver les entrées les plus similaires dans le dataset global
        similarites = [cosine_similarity([email_embedding], [item["input_embedding"]])[0][0] for item in dataset_embeddings]
        indices_tries = np.argsort(similarites)[::-1][:2]  # Prendre les 2 plus similaires
        exemples_similaires = [dataset_embeddings[i]["output"] for i in indices_tries]

        # Charger le dataset_embeddings spécifique à l'expéditeur
        dataset_embeddings = charger_dataset_embeddings(expediteur)
        logger.info(f"Dataset chargé pour {expediteur} avec {len(dataset_embeddings)} entrées")

        contexte_conversation = obtenir_contexte_conversation(expediteur)
        prompt = f"[CONVERSATION] :\n"

        for i, echange in enumerate(contexte_conversation):
            prompt += f"Echange {i+1}:\n"
            prompt += f"Email: {echange['email']}\n"
            prompt += f"Réponse: {echange['reponse']}\n\n"

        prompt += "[SUGGESTIONS]:\n"
        for exemple in exemples_similaires:
            prompt += f"{exemple}\n\n"

        prompt += f"[EMAIL ACTUEL]:\nExpéditeur: {expediteur}\nSujet: {sujet}\nContenu: {contenu_nettoye}\n\nRéponse:"
        # Générer la réponse
        generate_data = {
            "model": model_name,
            "prompt": prompt,
            "system": f"Tu es ASTRO, un assistant intelligent qui lit et répond aux messages de la boite email de {expediteur}. En te fiant à [CONVERSATION], avec l'aide des [SUGGESTIONS], formule une réponse pertinente à [EMAIL ACTUEL] de la part de {expediteur}",
            "stream": False
        }

        logger.info(f"PROMPT {prompt}")

        response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=generate_data)
        response_json = response.json()

        # Stocker le nouvel embedding
        stocker_embedding(expediteur, contenu_nettoye, email_embedding)
        return response_json['response']

    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse: {str(e)}")
        logger.error(traceback.format_exc())
        return "Désolé, une erreur s'est produite lors de la génération de la réponse."


def traiter_emails_et_appliquer_rag(imap_server, email_address, password, smtp_server, smtp_port, model_name):
    try:
        for sujet, contenu, email_message in lire_emails(imap_server, email_address, password):
            expediteur = email.utils.parseaddr(email_message['From'])[1]
            original_message_id = email_message['Message-ID']
            logger.info(f"Traitement de l'email {original_message_id} de {expediteur} avec le sujet: {sujet}")

            # Générer la réponse initial
            reponse_generee = generer_reponse(expediteur, sujet, contenu, model_name)
            sauvegarder_brouillon(imap_server, email_address, password, expediteur, sujet, reponse_generee, original_message_id)

            # Chercher la réponse effective dans le dossier "Sent"
            imap = imaplib.IMAP4_SSL(imap_server)
            imap.login(email_address, password)

            reponse_effective = chercher_reponse(imap, original_message_id)

            # Si une réponse a été envoyée, mettre à jour le contexte avec la réponse effective.
            if reponse_effective:
                ajouter_au_contexte_conversation(expediteur, contenu, reponse_effective)
            else:
                # Sinon, on met à jour le contexte avec la réponse generée
                 ajouter_au_contexte_conversation(expediteur, contenu, reponse_generee)
            imap.logout()


    except Exception as e:
        logger.error(f"Erreur générale dans le processus de traitement des emails: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        logger.info("Démarrage du processus de traitement des emails")
        IMAP_SERVER = os.getenv("IMAP_SERVER")
        SMTP_SERVER = os.getenv("SMTP_SERVER")
        SMTP_PORT = int(os.getenv("SMTP_PORT"))
        EMAIL = os.getenv("EMAIL")
        PASSWORD = os.getenv("PASSWORD")
        MODEL = os.getenv("MODEL")

        # ~ global dataset_embeddings

        # ~ dataset_embeddings = charger_dataset_embeddings()
        # ~ logger.info(f"Dataset chargé avec {len(dataset_embeddings)} entrées")
        logger.info(f"Mise à jour DATASET pour la BAL {EMAIL}")
        extraire_dataset()

        traiter_emails_et_appliquer_rag(IMAP_SERVER, EMAIL, PASSWORD, SMTP_SERVER, SMTP_PORT, MODEL)
        logger.info("=============== Fin du processus de traitement des emails... ====================")

    except KeyboardInterrupt:
        logger.info("Processus interrompu par l'utilisateur")
    except Exception as e:
        logger.critical(f"Erreur critique: {str(e)}")
        logger.critical(traceback.format_exc())
