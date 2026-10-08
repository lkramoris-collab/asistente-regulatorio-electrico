import os
import json
import chromadb
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)

# Configuration
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "reglamento"
NUM_RESULTS = 6
TOKEN_BUDGET_FILE = "token_usage.json"
MONTHLY_TOKEN_BUDGET = 3_000_000

# Initialize components
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print("Initializing Claude API client...")
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

conversation_history = []


def load_token_usage():
    if os.path.exists(TOKEN_BUDGET_FILE):
        with open(TOKEN_BUDGET_FILE, "r") as f:
            data = json.load(f)
            from datetime import datetime
            current_month = datetime.now().strftime("%Y-%m")
            if data.get("month") != current_month:
                return {"month": current_month, "tokens_used": 0, "queries": 0}
            return data
    from datetime import datetime
    return {"month": datetime.now().strftime("%Y-%m"), "tokens_used": 0, "queries": 0}


def save_token_usage(usage):
    with open(TOKEN_BUDGET_FILE, "w") as f:
        json.dump(usage, f)


# Answer length configurations
LENGTH_CONFIG = {
    "short": {
        "max_tokens": 500,
        "instruction": "Responde de forma MUY BREVE: maximo 3-4 frases. Ve directo al punto clave, cita el articulo y la fecha de entrada en vigor."
    },
    "medium": {
        "max_tokens": 1200,
        "instruction": "Responde de forma MODERADA: contexto, puntos clave, referencias normativas y plazos. No mas de 10-12 frases."
    },
    "detailed": {
        "max_tokens": 2500,
        "instruction": "Responde de forma DETALLADA: incluye contexto completo, todos los matices, plazos transitorios, normativa pendiente de desarrollo, impacto operativo, y URLs de referencia."
    }
}

# Precision level configurations
PRECISION_CONFIG = {
    "orientativa": {
        "instruction": """## Nivel de precision: ORIENTATIVA
Usa lenguaje conversacional y de negocio. Explica los conceptos en terminos sencillos sin necesidad de citar articulos especificos en cada frase. Da una vision general que permita al equipo entender rapidamente el tema. Puedes parafrasear libremente. Si no encuentras informacion exacta en los fragmentos, puedes dar una respuesta general basada en el contexto normativo disponible."""
    },
    "precisa": {
        "instruction": """## Nivel de precision: PRECISA
Incluye articulos especificos, fechas exactas, porcentajes, plazos y cifras concretas de los fragmentos proporcionados. Cada afirmacion importante debe ir acompanada de su referencia normativa (ej: Art. 28.5 RD 88/2026). Si un dato concreto no aparece en los fragmentos, NO lo inventes: indica claramente que falta esa informacion."""
    },
    "literal": {
        "instruction": """## Nivel de precision: LITERAL (para uso en comunicaciones oficiales)
Cita o parafrasea muy de cerca el texto exacto de la normativa. Incluye referencias completas con articulo, apartado y parrafo (ej: Art. 22.1.ah) parrafo tercero, RD 88/2026). Cada afirmacion DEBE estar respaldada por un fragmento concreto. Si algo NO esta en los fragmentos proporcionados, di explicitamente: "Esta informacion no consta en los fragmentos disponibles - verificar en [fuente]". NO completes ni interpretes mas alla del texto literal. Marca con [VERIFICAR] cualquier dato que no puedas confirmar al 100% con los fragmentos."""
    }
}

# Mode configurations
MODE_CONFIG = {
    "consulta": {
        "instruction": """Modo CONSULTA REGULATORIA: Responde preguntas sobre la normativa de forma precisa y profesional. 
Cita siempre el articulo y la normativa de referencia. Indica plazos de entrada en vigor y normativa pendiente de desarrollo."""
    },
    "comunicacion_cliente": {
        "instruction": """Modo BORRADOR COMUNICACION A CLIENTES: Genera un borrador de comunicacion dirigida a consumidores finales (personas fisicas o pymes).

Requisitos del borrador:
- Lenguaje CLARO, SENCILLO y NO TECNICO. Evita jerga juridica innecesaria.
- Tono informativo y cercano, propio de una comercializadora que se dirige a sus clientes.
- Estructura: saludo breve, que cambia y cuando, como les afecta, que deben hacer (si aplica), cierre.
- Incluye las fechas clave de entrada en vigor.
- NO incluyas informacion interna de la empresa ni datos confidenciales.
- El borrador debe cumplir con las obligaciones de transparencia del RD 88/2026 (Art. 6.1.m y 6.1.n).
- Marca claramente [COMPLETAR] donde haya datos especificos de la empresa que deban anadirse.

Genera el borrador listo para revision, no una explicacion de como redactarlo."""
    },
    "briefing_interno": {
        "instruction": """Modo BRIEFING INTERNO: Genera un resumen ejecutivo para equipos internos (ICT, B2C, Contratos, Cobros/Pagos).

Estructura del briefing:
1. RESUMEN EJECUTIVO (2-3 frases del cambio regulatorio)
2. QUE CAMBIA (puntos concretos con referencia al articulo)
3. PLAZOS (fechas de entrada en vigor y deadlines de adaptacion)
4. IMPACTO POR AREA:
   - ICT/Sistemas: cambios tecnicos necesarios (formatos CNMC, SIPS, facturacion, etc.)
   - B2C/Comunicacion: nuevas obligaciones de informacion al cliente
   - Contratos: cambios en clausulas, contenido minimo, documentacion
   - Cobros/Pagos: cambios en facturacion, plazos de pago, garantias
5. ACCIONES REQUERIDAS (lista priorizada)
6. NORMATIVA PENDIENTE (que falta por desarrollar)

Usa tono profesional y directo. Prioriza la informacion accionable."""
    },
    "deadlines": {
        "instruction": """Modo CALENDARIO DE PLAZOS: Genera una lista cronologica de todos los plazos y fechas limite relevantes.

Formato:
- Ordena por fecha (mas proxima primero)
- Para cada plazo indica: FECHA | QUE ENTRA EN VIGOR/VENCE | ARTICULO | QUIEN DEBE ACTUAR | ESTADO (en vigor / pendiente)
- Destaca los plazos que afectan directamente a comercializadoras
- Senala los plazos que dependen de normativa pendiente de desarrollo (Orden Ministerial, Resolucion CNMC, etc.)
- Si ya ha vencido un plazo, indicalo claramente."""
    }
}


WEB_REFERENCES = {
    # BOE - Normativa
    "BOE RD 88/2026": "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2026-3212",
    "BOE RD 88/2026 (texto consolidado)": "https://boe.es/buscar/act.php?id=BOE-A-2026-3212",
    "Ley 24/2013 del Sector Electrico": "https://www.boe.es/buscar/act.php?id=BOE-A-2013-13645",
    "RD 216/2014 PVPC": "https://www.boe.es/buscar/act.php?id=BOE-A-2014-3376",
    "RD 897/2017 Consumidor Vulnerable": "https://www.boe.es/buscar/act.php?id=BOE-A-2017-11505",
    "Circular CNMC 3/2020 Peajes": "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2020-1066",
    "RD 244/2019 Autoconsumo": "https://www.boe.es/buscar/act.php?id=BOE-A-2019-5089",
    "RD 1955/2000 Transporte y Distribucion": "https://www.boe.es/buscar/act.php?id=BOE-A-2000-24019",
    "RD 1048/2013 Retribucion Distribucion": "https://www.boe.es/buscar/act.php?id=BOE-A-2013-13768",
    "RD 1183/2020 Acceso y Conexion": "https://www.boe.es/buscar/act.php?id=BOE-A-2020-17278",
    # CNMC
    "CNMC Portal principal": "https://www.cnmc.es/",
    "CNMC Mercado electrico": "https://www.cnmc.es/ambitos-de-actuacion/energia/mercado-electrico",
    "CNMC Consumidores de energia": "https://www.cnmc.es/ambitos-de-actuacion/energia/consumidores-energia",
    "CNMC Comparador de ofertas": "https://comparador.cnmc.gob.es/",
    "CNMC Herramientas utiles": "https://www.cnmc.es/facil-para-ti/herramientas-utiles",
    "CNMC Listado comercializadoras": "https://sede.cnmc.gob.es/listado/censo/2",
    "CNMC Listado comercializadoras de referencia": "https://sede.cnmc.gob.es/listado/censo/16",
    "CNMC Listado distribuidoras": "https://sede.cnmc.gob.es/listado/censo/1",
    "CNMC Expediente Circular 3/2020": "https://www.cnmc.es/expedientes/cirde00219",
    # MITECO
    "MITECO Bono Social portal": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social.html",
    "MITECO Bono Social normativa": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social/normativa-bono-social.html",
    "MITECO Bono Social requisitos": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social/requisitos.html",
    "MITECO Bono Social FAQ": "https://www.miteco.gob.es/es/energia/energia-electrica/bono-social/preguntas-frecuentes-bono-social.html",
    "MITECO Cortes de suministro y SMV": "https://www.miteco.gob.es/es/energia/energia-electrica/electricidad/contratacion-suministro/cortes-suministro.html",
    "MITECO Comercializadores": "https://www.miteco.gob.es/es/energia/energia-electrica/electricidad/distribuidores/comercializadores.html",
    # REE / Redeia
    "REE Portal principal": "https://www.ree.es/es",
    "REE e-sios (datos del sistema)": "https://www.esios.ree.es/es",
    "REE Participacion en SIMEL": "https://www.ree.es/es/clientes/representante/gestion-medidas-electricas/participar-en-sistema-de-medidas",
    # OMIE
    "OMIE Operador del Mercado": "https://www.omie.es/",
}

SYSTEM_PROMPT = """Eres un asistente experto en regulacion del sector electrico espanol, utilizado por un equipo de consultores que asesora a una comercializadora de electricidad. Tu funcion es ayudar al equipo a entender la normativa, preparar comunicaciones y gestionar los plazos de adaptacion regulatoria.

## Extension de la respuesta:
{length_instruction}

{precision_instruction}

## Modo de trabajo activo:
{mode_instruction}

## Comportamiento general:

1. **Desafia las preguntas vagas**: Si la pregunta es ambigua, pide clarificacion antes de responder. Explica por que necesitas mas detalle y que opciones hay.

2. **Cita siempre las fuentes**: Articulo especifico, normativa, y fecha de entrada en vigor.

3. **Plazos transitorios**: El RD 88/2026 tiene multiples fechas de entrada en aplicacion (12/02/2026, 12/04/2026, 12/05/2026, 12/06/2026, 12/08/2026+). Siempre indica cual aplica.

4. **Normativa pendiente**: Senala cuando algo depende de Ordenes Ministeriales, Resoluciones de la CNMC o Procedimientos de Operacion aun no publicados.

5. **Seguridad**: NUNCA incluyas informacion interna, estrategias comerciales ni datos confidenciales en las respuestas. Toda la informacion debe basarse exclusivamente en normativa publica.

6. **Idioma**: Responde siempre en espanol.

7. **URLs de referencia disponibles**:
{urls}

8. **IMPORTANTE sobre URLs**: Utiliza UNICAMENTE las URLs listadas arriba. NUNCA inventes, supongas o generes URLs que no esten en esta lista. Si necesitas referenciar una normativa para la que no tienes URL, indica el nombre completo de la norma y sugiere buscarla en www.boe.es, pero NO inventes el enlace.

9. **NIVEL DE CONFIANZA - OBLIGATORIO en cada respuesta**: Al final de CADA respuesta, anade una seccion con este formato exacto:

---
**Nivel de confianza:** [ALTO / MEDIO / BAJO]
**Verificar en:** [URL del BOE o fuente oficial de la lista anterior]

Criterios para el nivel de confianza:
- ALTO: La respuesta se basa directamente en texto literal de los fragmentos proporcionados, con articulo especifico identificado.
- MEDIO: La respuesta combina informacion de varios fragmentos o requiere interpretacion del texto normativo. Puede haber matices no recogidos.
- BAJO: Los fragmentos proporcionados no contienen informacion suficiente, o la respuesta se basa parcialmente en conocimiento general del sector. REQUIERE VERIFICACION OBLIGATORIA.

Si el nivel es MEDIO o BAJO, indica especificamente QUE parte de la respuesta necesita verificacion.

## Base documental:
- RD 88/2026 Reglamento general de suministro, comercializacion y agregacion (BOE)
- Ley 24/2013 del Sector Electrico (texto consolidado)
- RD 897/2017 Consumidor Vulnerable y Bono Social (texto consolidado)
- RD 216/2014 PVPC (texto consolidado)
- Paginas web oficiales de CNMC, MITECO y REE (contenido scrapeado)

Basa tus respuestas exclusivamente en estos documentos publicos. Si la pregunta no puede responderse con la informacion disponible, indicalo claramente."""


def search_documents(query, n_results=NUM_RESULTS):
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results


def search_multi_query(queries, n_per_query=3, max_total=12):
    """Search with multiple queries, deduplicate, and cap total results."""
    all_docs = []
    all_metadatas = []
    seen_texts = set()

    for q in queries:
        embedding = embedding_model.encode([q])
        results = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=n_per_query
        )
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            key = doc[:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_docs.append(doc)
                all_metadatas.append(meta)

    # Cap total chunks to control token usage
    all_docs = all_docs[:max_total]
    all_metadatas = all_metadatas[:max_total]

    return {
        "documents": [all_docs],
        "metadatas": [all_metadatas]
    }


# Supplementary search queries by mode
MODE_EXTRA_QUERIES = {
    "deadlines": [
        "disposicion transitoria plazos entrada en vigor",
        "disposicion final novena entrada en vigor meses",
        "plazo maximo comercializadora adaptacion obligacion",
        "CNMC tres meses formatos ficheros intercambio",
        "cuatro meses devolucion garantias distribuidoras",
    ],
    "briefing_interno": [
        "obligaciones comercializadora sistemas informacion",
        "facturacion lectura peajes cargos condiciones",
        "contrato suministro contenido minimo cambio",
        "SIPS datos puntos suministro actualizacion",
    ],
    "comunicacion_cliente": [
        "derechos consumidor informacion transparente",
        "comunicacion precios revision contrato",
        "rescision penalizacion contrato suministro",
    ],
}


def build_context(search_results):
    context_parts = []
    for i, (doc, metadata) in enumerate(zip(
        search_results["documents"][0],
        search_results["metadatas"][0]
    )):
        source = metadata.get("source", "Unknown")
        context_parts.append(f"[Fragmento {i+1} - Fuente: {source}]\n{doc}")
    return "\n\n---\n\n".join(context_parts)


def format_urls():
    return "\n".join(f"   - {name}: {url}" for name, url in WEB_REFERENCES.items())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history

    user_message = request.json.get("message", "").strip()
    answer_length = request.json.get("length", "medium")
    mode = request.json.get("mode", "consulta")
    precision = request.json.get("precision", "precisa")

    if not user_message:
        return jsonify({"error": "Mensaje vacio"}), 400

    usage = load_token_usage()
    budget_percent = (usage["tokens_used"] / MONTHLY_TOKEN_BUDGET) * 100
    budget_warning = None

    if budget_percent >= 90:
        budget_warning = "Has consumido mas del 90% del presupuesto mensual. Usa respuestas cortas."
    elif budget_percent >= 75:
        budget_warning = "Has consumido mas del 75% del presupuesto mensual."
    elif budget_percent >= 50:
        budget_warning = "Has consumido mas del 50% del presupuesto mensual."

    length_config = LENGTH_CONFIG.get(answer_length, LENGTH_CONFIG["medium"])
    mode_config = MODE_CONFIG.get(mode, MODE_CONFIG["consulta"])
    precision_config = PRECISION_CONFIG.get(precision, PRECISION_CONFIG["precisa"])

    # Use multi-query search for modes that need broader coverage
    extra_queries = MODE_EXTRA_QUERIES.get(mode, [])
    if extra_queries:
        all_queries = [user_message] + extra_queries
        search_results = search_multi_query(all_queries, n_per_query=3, max_total=12)
    else:
        search_results = search_documents(user_message)
    context = build_context(search_results)

    augmented_message = f"""Pregunta del usuario: {user_message}

---
Fragmentos relevantes de la documentacion:

{context}"""

    # Store only the clean question in history (not the bulky chunks)
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Keep only last 6 messages to manage token usage
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]

    # Build messages to send: past history + current query with chunks
    messages_to_send = conversation_history[:-1] + [{
        "role": "user",
        "content": augmented_message
    }]

    try:
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=length_config["max_tokens"],
            system=SYSTEM_PROMPT.format(
                urls=format_urls(),
                length_instruction=length_config["instruction"],
                precision_instruction=precision_config["instruction"],
                mode_instruction=mode_config["instruction"]
            ),
            messages=messages_to_send
        )

        assistant_message = response.content[0].text

        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        tokens_this_query = response.usage.input_tokens + response.usage.output_tokens
        usage["tokens_used"] += tokens_this_query
        usage["queries"] += 1
        save_token_usage(usage)

        sources = list(set(
            m.get("source", "Unknown")
            for m in search_results["metadatas"][0]
        ))

        return jsonify({
            "response": assistant_message,
            "sources": sources,
            "tokens_used": tokens_this_query,
            "total_tokens": usage["tokens_used"],
            "budget_percent": round((usage["tokens_used"] / MONTHLY_TOKEN_BUDGET) * 100, 1),
            "budget_warning": budget_warning,
            "queries_this_month": usage["queries"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/usage", methods=["GET"])
def get_usage():
    usage = load_token_usage()
    budget_percent = (usage["tokens_used"] / MONTHLY_TOKEN_BUDGET) * 100
    return jsonify({
        "month": usage["month"],
        "tokens_used": usage["tokens_used"],
        "budget_total": MONTHLY_TOKEN_BUDGET,
        "budget_percent": round(budget_percent, 1),
        "queries": usage["queries"]
    })


@app.route("/resources", methods=["GET"])
def get_resources():
    """Return organized list of all reference resources."""
    resources = {
        "normativa": {
            "title": "Normativa (BOE)",
            "items": []
        },
        "cnmc": {
            "title": "CNMC - Comision Nacional de los Mercados y la Competencia",
            "items": []
        },
        "miteco": {
            "title": "MITECO - Ministerio para la Transicion Ecologica",
            "items": []
        },
        "operadores": {
            "title": "Operadores del Sistema y del Mercado",
            "items": []
        }
    }

    for name, url in WEB_REFERENCES.items():
        item = {"name": name, "url": url}
        if name.startswith("BOE") or name.startswith("Ley") or name.startswith("RD") or name.startswith("Circular"):
            resources["normativa"]["items"].append(item)
        elif name.startswith("CNMC"):
            resources["cnmc"]["items"].append(item)
        elif name.startswith("MITECO"):
            resources["miteco"]["items"].append(item)
        else:
            resources["operadores"]["items"].append(item)

    # Add info about ingested documents
    doc_sources = set()
    all_metadatas = collection.get()["metadatas"]
    for meta in all_metadatas:
        doc_sources.add(meta.get("source", "Unknown"))

    resources["documentos_ingested"] = {
        "title": "Documentos en base de conocimiento",
        "items": [{"name": s} for s in sorted(doc_sources)]
    }

    return jsonify(resources)


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "Conversacion reiniciada"})


if __name__ == "__main__":
    usage = load_token_usage()
    print("\n" + "=" * 60)
    print("REGLAMENTO CHATBOT - Equipo Regulatorio")
    print(f"Documentos en base de datos: {collection.count()} chunks")
    print(f"Tokens usados este mes: {usage['tokens_used']:,} / {MONTHLY_TOKEN_BUDGET:,}")
    print(f"Consultas este mes: {usage['queries']}")
    print("Abrir http://localhost:5000 en el navegador")
    print("=" * 60 + "\n")
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=5000)
