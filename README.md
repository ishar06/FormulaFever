# 🏎️ FormulaFever

A comprehensive, all-in-one web platform for Formula 1 fans, offering everything from live data and predictive analysis to an integrated AI chatbot and a vibrant community forum.

-----

## 📋 Table of Contents

  * [About The Project](#about-the-project)
  * [✨ Core Features](#-core-features)
  * [💡 Novelty & Uniqueness](#-novelty--uniqueness)
  * [🚀 Tech Stack](#-tech-stack)
  * [🏁 Getting Started](#-getting-started)
      * [Prerequisites](#prerequisites)
      * [Installation & Setup (macOS)](#installation--setup-macos)
      * [Installation & Setup (Windows)](#installation--setup-windows)
  * [📂 Project Structure](#-project-structure)
  * [👥 Our Team](#-our-team)

-----

## About The Project

FormulaFever is a Flask-based web application built for passionate Formula 1 enthusiasts. We aim to be the ultimate pit stop for fans by consolidating historical analysis, live race data, predictive modeling, and community interaction into a single, user-friendly platform.

Whether you want to compare your favorite drivers head-to-head, get live stats during a race, predict the outcome of the next Grand Prix, or simply ask our AI chatbot a complex F1 question, FormulaFever has you covered.

## ✨ Core Features

  * **🤖 AI-Powered Chatbot:** Ask complex questions about drivers, teams, and race history. Powered by the **Groq API** for lightning-fast, conversational responses.
  * **📊 In-Depth Driver Analysis:** Visualize and explore detailed performance metrics for individual drivers across various seasons.
  * **🆚 Driver Comparator:** Pit any two drivers against each other with a dynamic, head-to-head comparison of their career stats and performance.
  * **🔮 Future Race Prediction:** Utilizes a **scikit-learn** model trained on historical data to predict the outcomes of
    upcoming races.
  * **🔴 Live Race Stats:** Get real-time updates, lap times, and driver positions during live Grand Prix weekends.
  * **🏁 Previous Race Analysis:** Dive deep into past race results, strategies, and key moments with detailed visualizations.
  * **💬 Community Hub:** A dedicated space for fans to chat, react, and discuss all things F1 in real-time.

## 💡 Novelty & Uniqueness

The uniqueness of FormulaFever lies in its **integration**. While other platforms might offer one or two of these features, FormulaFever is one of the first to combine:

1.  **Rich Historical Data Analysis** (using `fastf1` & `jolpica`)
2.  **Predictive Machine Learning** (using `scikit-learn`)
3.  **Next-Gen AI Interaction** (using the ultra-fast `Groq` LPU)
4.  **Real-Time Community** features

It's not just a data dashboard; it's an interactive and predictive ecosystem. The use of the Groq API, in particular, provides near-instantaneous, conversational insights, a feature not commonly found in F1 fan projects.

## 🚀 Tech Stack

  * **Backend:** Python, Flask
  * **F1 Data APIs:** `fastf1`, `jolpica`
  * **Machine Learning:** `scikit-learn`, `pandas`, `numpy`
  * **AI Chatbot:** `groq`
  * **Frontend:** HTML, CSS, JavaScript
  * **Database:** SQLite / SQLAlchemy (for community chat & user data)

-----

## 🏁 Getting Started

Follow these instructions to get a local copy up and running on your machine.

### Prerequisites

  * Python 3.9+
  * Git

### Installation & Setup (macOS)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-team/FormulaFever.git
    cd FormulaFever
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory (`/FormulaFever`) and add your API keys:
    ```.env
    GROQ_API_KEY='your_groq_api_key_here'
    FLASK_SECRET_KEY='your_strong_random_secret_key_here'
    ```
5.  **Initialize the Database** (for the community feature):
    ```bash
    # (Assuming you are using Flask-Migrate)
    flask db init
    flask db migrate -m "Initial migration."
    flask db upgrade
    ```
6.  **Run the application:**
    ```bash
    flask run
    ```
    Open `http://127.0.0.1:5000` in your browser.

> **Note on `ff1_cache`:** The `fastf1` library requires a cache to store F1 data, which speeds up load times. This project is pre-configured to use the `ff1_cache` directory. The app will automatically write to this folder.

-----

### Installation & Setup (Windows)

1.  **Clone the repository (using Git Bash or Command Prompt):**
    ```bash
    git clone https://github.com/your-team/FormulaFever.git
    cd FormulaFever
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    .\env\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory (`/FormulaFever`) and add your API keys:
    ```.env
    GROQ_API_KEY='your_groq_api_key_here'
    FLASK_SECRET_KEY='your_strong_random_secret_key_here'
    ```
5.  **Initialize the Database** (for the community feature):
    ```bash
    # (Assuming you are using Flask-Migrate)
    flask db init
    flask db migrate -m "Initial migration."
    flask db upgrade
    ```
6.  **Run the application:**
    ```bash
    # Set the Flask app environment variable
    set FLASK_APP=app.py
    flask run
    ```
    Open `http://127.0.0.1:5000` in your browser.

-----

## 📂 Project Structure

```
FORMULAFEVER/
├── app.py                # Main Flask application (routes, config)
├── analysis.py           # Data analysis & prediction logic
├── requirements.txt      # Project dependencies
├── README.md             # This file
├── .env                  # (To be created) API keys & secrets
├── env/                  # Virtual environment
├── ff1_cache/            # Cache for fastf1 data
├── instance/             # Instance folder (e.g., for SQLite DB)
├── static/               # CSS, JS, images
└── templates/            # HTML files
    ├── layout.html
    ├── home.html
    ├── driver.html
    ├── comparator.html
    ├── predictor.html
    ├── community.html
    ├── ...
```

## 👥 Our Team

This project was brought to life by the following team members:

  * [**Ishardeep**](https://github.com/ishar06e)
  * [**Damanjeet Singh**](https://github.com/daman-max)
  * [**Bhumika**](https://github.com/BhumikaNagpal)
  * [**Karandeep Kaur**](https://github.com/karandeepkaur18)


-----
