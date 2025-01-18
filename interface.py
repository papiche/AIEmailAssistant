from flask import Flask, render_template, request, redirect, url_for, flash
import os
import json
from dotenv import load_dotenv, set_key
import subprocess
import threading

app = Flask(__name__)
app.secret_key = "une_cle_secrete"  # Clé secrète pour les flash messages

# Chemin du fichier .env
DOTENV_PATH = ".env"

# Variables d'environnement modifiables
MODIFIABLE_ENV_VARS = [
    'IMAP_SERVER',
    'SMTP_SERVER',
    'SMTP_PORT',
    'EMAIL',
    'PASSWORD',
    'SENTDIR',
    'DRAFTDIR',
    'OLLAMA_API_URL',
    'MODEL',
    'CONTEXT'
]


# Fonction pour charger les variables d'environnement
def load_env():
    load_dotenv(DOTENV_PATH)
    return os.environ.copy()

# Route pour la page d'accueil
@app.route('/')
def index():
    emails = os.listdir('./emails') if os.path.exists('./emails') else []
    return render_template('index.html', emails=emails)

# Route pour afficher le contexte et les embeddings d'un email
@app.route('/context/<email>', methods=['GET', 'POST'])
def show_context(email):
    context_file = os.path.join('./emails', email, 'context.txt')
    embeddings_file = os.path.join('./emails', email, 'embeddings.json')
    conversation_file = os.path.join('./emails', 'conversation_context.json')

    context = ""
    if os.path.exists(context_file):
        with open(context_file, 'r') as f:
            context = f.read()

    embeddings = []
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'r') as f:
            embeddings = json.load(f)

    conversations = {}
    if os.path.exists(conversation_file):
         with open(conversation_file, 'r') as f:
            conversations = json.load(f).get(email, [])

    if request.method == 'POST':
        edited_embeddings = []
        for index, item in enumerate(embeddings):
            edited_content = request.form.get(f'content_{index}', item["contenu"])
            edited_embeddings.append({"contenu": edited_content, "embedding": item["embedding"]})

        with open(embeddings_file, 'w') as f:
             json.dump(edited_embeddings, f)
        flash('Embeddings mis à jour !', 'success')
        return redirect(url_for('show_context', email=email))

    return render_template('context.html', email=email, context=context, embeddings=embeddings, conversations =conversations)


# Route pour afficher et éditer le fichier .env
@app.route('/env', methods=['GET', 'POST'])
def manage_env():
    env_vars = load_env()
    modifiable_vars = {key: env_vars.get(key, '') for key in MODIFIABLE_ENV_VARS}
    if request.method == 'POST':
        for key, value in request.form.items():
            if key in MODIFIABLE_ENV_VARS:
                set_key(DOTENV_PATH, key, value)
        flash('Variables d\'environnement mises à jour !', 'success')
        return redirect(url_for('manage_env'))
    return render_template('env.html', env_vars=modifiable_vars)


# Route pour lancer le traitement des emails
@app.route('/run', methods=['GET', 'POST'])
def run_process():
    if request.method == 'POST':
        # Lancer le processus en arrière plan
        threading.Thread(target=run_email_script).start()
        flash('Le traitement des emails a démarré en arrière-plan.', 'info')
        return redirect(url_for('index'))
    return render_template('run.html')

def run_email_script():
    """Lance le script email3llama.py dans un sous-processus."""
    try:
        subprocess.run(["python", "email3llama.py"], check=True)
        flash('Le traitement des emails est terminé !', 'success')
    except subprocess.CalledProcessError as e:
         flash(f'Erreur lors du traitement des emails : {e}', 'error')
    except FileNotFoundError:
         flash(f'Erreur, script `email3llama.py` introuvable', 'error')



# Gestion des erreurs FileNotFoundError
@app.errorhandler(FileNotFoundError)
def handle_file_not_found(e):
    return render_template('error.html', error=str(e)), 404

if __name__ == '__main__':
    app.run(debug=True)
