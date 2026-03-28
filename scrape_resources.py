"""
Scraper for official institutional web pages related to Spanish electricity regulation.
Downloads content and saves as text files in the documents folder for ChromaDB ingestion.

Usage: py scrape_resources.py
"""

import os
import re
import time
import requests
from datetime import datetime

DOCUMENTS_DIR = "documents"
SCRAPED_DIR = os.path.join(DOCUMENTS_DIR, "scraped")

# Pages to scrape - organized by institution
PAGES_TO_SCRAPE = {
    # CNMC
    "CNMC - Mercado electrico": "https://www.cnmc.es/ambitos-de-actuacion/energia/mercado-electrico",
    "CNMC - Consumidores de energia": "https://www.cnmc.es/ambitos-de-actuacion/energia/consumidores-energia",
    "CNMC - Herramientas utiles": "https://www.cnmc.es/facil-para-ti/herramientas-utiles",

    # MITECO - Bono Social
    "MITECO - Bono Social portal": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social.html",
    "MITECO - Bono Social normativa": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social/normativa-bono-social.html",
    "MITECO - Bono Social requisitos": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social/requisitos.html",
    "MITECO - Bono Social FAQ": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social/preguntas-frecuentes-bono-social.html",
    "MITECO - Cortes de suministro y SMV": "https://www.miteco.gob.es/es/energia/energia-electrica/electricidad/contratacion-suministro/cortes-suministro.html",
    "MITECO - Comercializadores": "https://www.miteco.gob.es/es/energia/energia-electrica/electricidad/distribuidores/comercializadores.html",

    # REE
    "REE - Acerca de ESIOS": "https://www.esios.ree.es/es/acerca-de-esios",
    "REE - Participacion en SIMEL": "https://www.ree.es/es/clientes/representante/gestion-medidas-electricas/participar-en-sistema-de-medidas",
}

# Simple headers to avoid being blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


def clean_html(html_text):
    """Extract readable text from HTML, removing tags, scripts, styles."""
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove cookie banners and similar
    text = re.sub(r'<div[^>]*class="[^"]*cookie[^"]*"[^>]*>.*?</div>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Replace common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&aacute;', 'a')
    text = text.replace('&eacute;', 'e')
    text = text.replace('&iacute;', 'i')
    text = text.replace('&oacute;', 'o')
    text = text.replace('&uacute;', 'u')
    text = text.replace('&ntilde;', 'n')

    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove very short lines (likely menu items, buttons)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 30:  # Keep only meaningful lines
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def safe_filename(name):
    """Convert a name to a safe filename."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:80]


def scrape_page(name, url):
    """Scrape a single page and return cleaned text."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'

        raw_html = response.text
        clean_text = clean_html(raw_html)

        if len(clean_text) < 100:
            print(f"  WARNING: Very little content extracted ({len(clean_text)} chars)")
            return None

        # Add metadata header
        header = f"""FUENTE: {name}
URL: {url}
FECHA DE DESCARGA: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'=' * 60}

"""
        return header + clean_text

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("=" * 60)
    print("SCRAPER: Descargando contenido de webs oficiales")
    print("=" * 60)

    # Create scraped directory
    os.makedirs(SCRAPED_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0
    total_chars = 0

    for name, url in PAGES_TO_SCRAPE.items():
        print(f"\n[{success_count + fail_count + 1}/{len(PAGES_TO_SCRAPE)}] {name}")
        print(f"  URL: {url}")

        content = scrape_page(name, url)

        if content:
            filename = safe_filename(name) + ".txt"
            filepath = os.path.join(SCRAPED_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            chars = len(content)
            total_chars += chars
            success_count += 1
            print(f"  OK: {chars:,} caracteres guardados en {filename}")
        else:
            fail_count += 1
            print(f"  FALLO: No se pudo extraer contenido")

        # Be polite - wait between requests
        time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"COMPLETADO:")
    print(f"  Paginas descargadas: {success_count}/{len(PAGES_TO_SCRAPE)}")
    print(f"  Fallos: {fail_count}")
    print(f"  Total caracteres: {total_chars:,}")
    print(f"  Archivos guardados en: {SCRAPED_DIR}/")
    print(f"\nAhora ejecuta 'py ingest.py' para incorporar el contenido al chatbot.")
    print("=" * 60)

    # Save a manifest of what was scraped
    manifest_path = os.path.join(SCRAPED_DIR, "_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"Scraping realizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Paginas: {success_count}/{len(PAGES_TO_SCRAPE)}\n\n")
        for name, url in PAGES_TO_SCRAPE.items():
            filename = safe_filename(name) + ".txt"
            filepath = os.path.join(SCRAPED_DIR, filename)
            exists = "OK" if os.path.exists(filepath) else "FALLO"
            f.write(f"[{exists}] {name}\n  {url}\n\n")


if __name__ == "__main__":
    main()
