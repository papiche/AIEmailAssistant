# Assistant Email Intelligent avec Ollama

Ce projet implémente un assistant email intelligent utilisant Ollama pour générer des réponses automatiques aux emails entrants. Le système apprend continuellement des interactions passées pour améliorer ses réponses futures.

## Fonctionnalités Principales

1. **Lecture des Emails** : Connexion à un serveur IMAP pour lire les emails non lus.
2. **Génération de Réponses** : Utilisation d'Ollama pour créer des réponses pertinentes basées sur le contenu de l'email et le contexte historique.
3. **Brouillons Intelligents** : Sauvegarde des réponses générées comme brouillons pour révision humaine.
4. **Apprentissage Continu** : Extraction et utilisation des paires email/réponse pour améliorer les futures générations.
5. **Gestion de Contexte** : Maintien d'un contexte global pour des réponses plus cohérentes.

Le code "email3llama.py" est un système de traitement automatisé des emails utilisant l'intelligence artificielle.
Voici une explication de ses principales fonctionnalités :

* Connexion à la boîte email via IMAP
* Lecture des emails non lus
* Extraction d'un dataset à partir des emails précédents
* Génération de réponses aux nouveaux emails
*Sauvegarde des réponses en brouillon

Workflow de traitement des emails

text
graph TD
    A[Connexion IMAP] --> B[Lecture emails non lus]
    B --> C[Extraction dataset]
    C --> D[Génération embedding]
    D --> E[Recherche similarités]
    E --> F[Génération réponse]
    F --> G[Sauvegarde brouillon]

Étapes clés du processus

* Extraction du dataset : La fonction extraire_dataset() parcourt les emails reçus et leurs réponses pour créer un jeu de données d'entraînement.
* Génération d'embeddings : generer_embedding() utilise le modèle Ollama pour créer des représentations vectorielles des emails.
* Recherche de similarités : Le système compare l'embedding de l'email entrant avec ceux du dataset pour trouver des exemples pertinents.
* Génération de réponse : generer_reponse() utilise le contexte, les exemples similaires et le contenu de l'email pour produire une réponse appropriée.
* Sauvegarde en brouillon : La réponse générée est sauvegardée comme brouillon dans la boîte email.

## Configuration

1. Copiez `.env.template` en `.env` et remplissez les variables :

```
IMAP_SERVER=mail.example.com
SMTP_SERVER=mail.example.com
SMTP_PORT=587
EMAIL=votre@email.com
PASSWORD=votremotdepasse
SENTDIR=INBOX.Sent
DRAFTDIR=INBOX.Draft
OLLAMA_API_URL=http://localhost:11434
MODEL=llama2
CONTEXT=./context.txt
```

2. Chemin de stockage des embeddings
```
# 'ln -s'
mkdir emails
```

2. Assurez-vous qu'Ollama est installé et en cours d'exécution sur votre système.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Exécutez le script principal :

```bash
python email3llama.py
```

## Contribution

Les contributions sont les bienvenues. Veuillez ouvrir une issue ou un pull request pour toute suggestion ou amélioration.

## Licence

AGPL
