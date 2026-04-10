# Gemini Python Chatbot

Jednoduchý chatbot v Pythonu používající Google Gemini API.

## Co umí

* Uživatel zadává text do terminálu
* Model odpovídá
* Pamatuje si kontext (historii konverzace)
* Ukončení příkazem `exit`

## Instalace

```bash
pip install google-genai python-dotenv
```

## Nastavení API klíče

Vytvoř `.env` soubor:

```bash
GEMINI_API_KEY=tvuj_api_klic
```

## Spuštění

```bash
python chatbot_gem.py
```

## Testování

* Zeptej se na cokoliv (např. "Co je Python?")
* Zeptej se navazující otázku (ověření paměti)
* Ukonči pomocí `exit`

## Možná vylepšení

* GUI (web nebo desktop)
* Ukládání historie do souboru
* Streaming odpovědí
* Role asistenta (např. učitel, programátor)

## Technologie

* Python
* Google Gemini API
* python-dotenv