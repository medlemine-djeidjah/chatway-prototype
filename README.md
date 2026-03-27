<div align="center">
  <img src="https://img.shields.io/badge/AI-Gemini%202.5%20Flash-blue?style=for-the-badge&logo=google" alt="Gemini">
  <img src="https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Vector%20DB-FAISS-red?style=for-the-badge&logo=meta" alt="FAISS">
  <img src="https://img.shields.io/badge/Frontend-TailwindCSS-38B2AC?style=for-the-badge&logo=tailwind-css" alt="Tailwind">
  <img src="https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker" alt="Docker">

  <h1>💬 ChatWay - Assistant Q&R Documentaire</h1>

  <p>Un système de <b>RAG Multimodal</b> (Retrieval-Augmented Generation) pour interroger intelligemment des documents non-sélectionnables (PDFs scannés, graphiques, tableaux).</p>

  <p>
    <a href="https://votre-lien-demo.com"><b>🚀 Tenter la Démo en Ligne</b></a> • 
    <a href="https://medlemine-djeidjah.github.io/"><b>💼 Visiter mon Portfolio</b></a>
  </p>
</div>

---

## 🚀 Fonctionnalités Principales
- **Analyse Visuelle Multimodale** : Utilisation de modèles de Vision LLM (Gemini) pour extraire intelligemment du texte depuis des documents scannés, y compris la description et la donnée précise des infographies et tableaux.
- **Chunking Sémantique Haute Fidélité** : Extraction structurée préservant nativement le contexte hiérarchique (Chapitre, Sous-chapitre, Sujet) pour une précision de recherche inégalée de la part de l'IA.
- **Interface Utilisateur Moderne** : Design conversationnel fluide, propulsé par des animations CSS et un layout Tailwind/DaisyUI.
- **Micro-Services & Déploiement** : Architecture parfaitement scindée (Routing vs Machine Learning) conteneurisée via Docker.

## 🛠️ Stack Technique
- **Backend Core** : Python 3.10, Flask
- **Intelligence Artificielle** : LangChain, Google Gemini API (Modèles Vision & Embedding), vectorstore FAISS (Facebook AI Similarity Search)
- **Extraction PDF** : `pdf2image`, `poppler-utils`
- **Frontend** : HTML5, Vanilla JavaScript, TailwindCSS, DaisyUI
- **Infrastructure** : Docker, Docker Compose, Scripts Bash

---

## 📥 Installation & Lancement Rapide (Via Docker)

### Prérequis
- [Docker](https://www.docker.com/) et Docker Compose installés.

### 1. Cloner le Projet
Assurez-vous d'être dans le dossier racine du projet `chatway-prototype`.

### 2. Obtenir et Configurer la Clé API Gemini
Vous devez générer une clé API pour que le modèle fonctionne :
- Rendez-vous sur [Google AI Studio](https://aistudio.google.com/).
- Allez dans la section **API keys**.
- Cliquez sur **Create API key** pour en générer une nouvelle.

Copiez ensuite le fichier d'exemple pour créer votre configuration locale :
```bash
cp .env.example .env
```
Ouvrez le fichier `.env` et collez votre clé à côté de la variable `GOOGLE_API_KEY` :
```env
GOOGLE_API_KEY=votre_nouvelle_clé_api_ici
```

### 3. Générer la Base de Connaissances (Inférence Initiale)
Déposez votre document PDF source sous le nom exact `data.pdf` à la racine de votre dossier. Exécutez le script d'extraction local pour que l'API analyse visuellement le PDF et construise l'espace vectoriel :
```bash
python ingest.py
```
*(Cela générera le dossier `faiss_index/` indispensable au RAG).*

### 4. Déployer l'Application
Lancez le script de déploiement via Docker Compose :
```bash
./deploy.sh
```
L'application est déployée ! 🌍 Visitez [http://localhost:5000](http://localhost:5000).

---

## 🖥️ Utilisation locale (Environnement de Développement)
Si vous ne souhaitez pas utiliser Docker:
1. Installez les librairies : `pip install -r requirements.txt`
2. Installez `poppler-utils` au niveau du système d'exploitation.
3. Construisez l'index : `python ingest.py`
4. Lancez le serveur Flask de dev : `python app.py`

---
<div align="center">
  <p>Développé par Med Lemine.</p>
  <a href="https://medlemine-djeidjah.github.io/"><b>👉 Voir mes autres expérimentations sur mon portfolio</b></a>
</div>
