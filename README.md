# 🔗 Six Degrees of AI Separation

## Overview
Six Degrees of AI Separation is a locally hosted, interactive AI agent that finds the hidden historical, scientific, or cultural connections between any two completely unrelated topics. 

Instead of just guessing, the app actively searches Wikipedia for real-time facts, reads the articles, and uses a local AI to weave a factual, step-by-step narrative connecting the two subjects. It then takes that data and automatically generates an interactive "evidence board" so users can visually explore the connections.

## Key Features
* **Live Web Research:** Uses a Python web-scraper to pull factual data directly from Wikipedia before answering, preventing the AI from "hallucinating" or making up fake history.
* **Dynamic Chaos Control:** Includes a UI slider that allows users to adjust the AI's "temperature" on the fly, controlling whether the generated story reads like a strict encyclopedia or a dramatic documentary.
* **Interactive Data Visualization:** Automatically translates the AI's text output into a physics-based, interactive network graph where users can click, drag, and explore the steps connecting the topics.
* **100% Local Processing:** The core AI brain (Llama 3) runs entirely on the user's local hardware, ensuring complete data privacy.

## How to Run
1. Clone this repository.
2. Install the required libraries found in `requirements.txt`.
3. Download a compatible local `.gguf` language model (like Llama 3) and update the absolute path in `app.py`.
4. Run `python3 app.py` to launch the local web interface.
