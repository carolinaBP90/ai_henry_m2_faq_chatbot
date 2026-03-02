# AI Henry M2 - FAQ Chatbot RAG

Sistema de Preguntas Frecuentes (FAQ) con Generación Aumentada por Recuperación (RAG) usando embeddings locales y búsqueda vectorial. Diseñado para proporcionar respuestas precisas a preguntas sobre políticas de empresa.

## Características

- ✅ **Embeddings Locales**: Usa `sentence-transformers` (sin dependencia de APIs externas)
- ✅ **Búsqueda Vectorial Rápida**: FAISS para k-NN search eficiente
- ✅ **Respuestas Generadas**: LLM opcional (OpenAI GPT-3.5) o rule-based
- ✅ **Evaluación Automática**: Evaluador de calidad de respuestas
- ✅ **Procesamiento Multilenguaje**: Soporta español, inglés y otros idiomas
- ✅ **Modular y Extensible**: Arquitectura limpia para fácil mantenimiento

## Requisitos Previos

- **Python**: 3.10 o superior
- **pip**: Gestor de paquetes de Python  
- **Git**: Para clonar el repositorio (opcional)

## Instalación Rápida

### 1. Clonar el Repositorio

```bash
git clone <tu-repositorio-url>
cd ai_henry_m2_faq_chatbot
```

### 2. Crear Entorno Virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias (CRÍTICO)

```bash
pip install -r requirements.txt
```

**Dependencias requeridas**:
- `numpy` - Operaciones numéricas
- `scikit-learn` - Machine learning utilities
- `python-dotenv` - Variables de entorno
- `sentence-transformers` - **CRÍTICO**: Embeddings locales sin API
- `faiss-cpu` - **RECOMENDADO**: Búsqueda vectorial rápida

### 4. Configurar Variables de Entorno (Opcional)

```bash
cp env.example .env
```

Edita `.env` si deseas usar OpenAI APIs:

```plaintext
OPENAI_API_KEY=sk-...              # Solo si usas OpenAI
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
USE_OPENAI=false                    # Por defecto: generación local
```

## Uso - 3 Formas

### Forma 1: Pipeline Completo (Recomendado)

```bash
python main.py
```

**Resultado**:
- ✅ Construye índice vectorial
- ✅ Ejecuta 3 preguntas de ejemplo
- ✅ Guarda respuestas en `outputs/sample_queries.json`

**Salida esperada**:
```
============================================================
PIPELINE EXECUTION SUMMARY
============================================================
Document processed: data/faq_document.txt
Vector store created: vectorstore
Sample queries executed: 3
Results saved to: outputs/sample_queries.json
============================================================
```

### Forma 2: Construir Índice Solamente

```bash
python src/build_index.py \
  --document-path data/faq_document.txt \
  --vectorstore-path vectorstore
```

**Parámetros**:
- `--document-path`: Ruta al FAQ (default: `data/faq_document.txt`)
- `--vectorstore-path`: Donde guardar índice (default: `vectorstore`)
- `--use-openai`: Usa OpenAI embeddings (requiere API key)

### Forma 3: Consultar el Sistema

```bash
python src/query.py \
  --question "¿Cómo agrego un nuevo integrante del equipo?" \
  --vectorstore-path vectorstore
```

**Parámetros**:
- `--question`: Tu pregunta (requerido)
- `--vectorstore-path`: Ruta al índice (default: `vectorstore`)
- `--use-openai`: Usa OpenAI LLM (opcional)
- `--no-evaluate`: Desactiva evaluador de calidad
- `--output-json`: Guarda resultado en JSON

## Estructura del Proyecto

```
ai_henry_m2_faq_chatbot/
├── main.py                        # ← PUNTO DE ENTRADA
├── requirements.txt               # Dependencias (CRÍTICAS)
├── pyproject.toml                # Metadatos del proyecto
├── env.example                   # Plantilla .env
├── .gitignore                    # Git ignore rules
├── README.md                     # Este archivo
│
├── src/
│   ├── __init__.py
│   ├── build_index.py            # Módulo 1: Construir índice
│   │   ├── DocumentChunker       # Divide FAQ en chunks
│   │   ├── EmbeddingGenerator    # Genera embeddings
│   │   └── VectorStore           # Almacena vectores
│   │
│   └── query.py                  # Módulo 2: Consultar
│       ├── RAGQueryEngine        # Búsqueda + respuesta
│       ├── ResponseEvaluator     # Evaluador de calidad
│       └── load_and_query()      # API pública
│
├── data/
│   └── faq_document.txt          # FAQ fuente
│
├── vectorstore/                  # (GENERADO)
│   ├── faiss.index
│   ├── embeddings.npy
│   ├── chunks.pkl
│   └── metadata.json
│
└── outputs/
    └── sample_queries.json       # (GENERADO)
```

## Decisiones Técnicas

### Chunking (Dividir FAQ)

**Estrategia**: Jerárquica por secciones y topics

1. Divide por secciones (marcadas con `=====`)
2. Divide por `TOPIC:`
3. Divide por párrafos
4. Fusiona en chunks de 80-350 tokens

**¿Por qué?**: Mantiene coherencia semántica y mejora recuperación en FAQ estructurados.

### Embeddings (Convertir texto a vectores)

**Backend principal**: `sentence-transformers` (local, 384-dim)

- ✅ Rápido (~50ms)
- ✅ Sin API external
- ✅ Buena calidad

**Fallback**: `HashingVectorizer` (si sentence-transformers falta)

**OpenAI** (opcional): `text-embedding-3-small` (1536-dim, requiere API)

### Búsqueda Vectorial

**Método**: k-NN (k-Nearest Neighbors)

- 🚀 FAISS si disponible (muy rápido)
- 📊 Fallback a cosine similarity (NumPy)
- 📌 Recupera top-5 chunks por defecto

## Configuración Avanzada

### Objetivo
Construir un chatbot que:
- procese un documento FAQ en texto plano,
- lo divida en chunks (20+),
- genere embeddings,
- realice búsqueda vectorial para recuperar contexto relevante,
- genere respuestas y devuelva JSON estructurado con:
  - `user_question`
  - `system_answer`
  - `chunks_related`
- (bonus) evalúe automáticamente la calidad de la respuesta (0–10).

## Estructura del proyecto

```bash
M2_chatbot_IA/
├── data/
│   └── faq_document.txt
├── outputs/
│   └── sample_queries.json
├── src/
│   ├── build_index.py
│   └── query.py
├── vectorstore/
│   ├── embeddings.npy
│   ├── chunks.pkl
│   └── metadata.json
├── .env.example
├── main.py
├── requirements.txt
└── README.md
```

## Requisitos
- Python **3.10+** (probado en 3.13)
- pip

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración de variables de entorno
Crea un archivo `.env` a partir de `.env.example`.

Ejemplo:

```bash
OPENAI_API_KEY=your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
USE_OPENAI=false
```

> Si no configuras `OPENAI_API_KEY`, el sistema funciona en modo offline con embeddings locales (HashingVectorizer).

## Uso

### 1) Construir índice (pipeline de datos)

```bash
python src/build_index.py --document-path data/faq_document.txt --vectorstore-path vectorstore
```

Este paso:
1. Carga el documento FAQ
2. Divide en chunks inteligentes (20+)
3. Genera embeddings
4. Guarda embeddings + chunks para recuperación

### 2) Consultar el chatbot (pipeline de queries)

```bash
python src/query.py --question "How do I add a new team member?" --vectorstore-path vectorstore
```

Salida: JSON con `user_question`, `system_answer`, `chunks_related` y metadatos.

### 3) Ejecutar demo end-to-end

```bash
python main.py
```

Esto:
- construye índice,
- ejecuta 3 consultas de ejemplo,
- guarda resultados en `outputs/sample_queries.json`.

## Método de búsqueda vectorial
- **k-NN** sobre embeddings (FAISS si está disponible; fallback a similitud coseno con NumPy).
- Recupera top-k chunks relevantes (`top_k=5` por defecto).
- Método reportado en salida como:
  - `k-NN (FAISS) if available, else cosine similarity`

## Estrategia de chunking (decisiones técnicas)
Se usa chunking jerárquico:
1. División por secciones macro del documento.
2. División por `TOPIC:`.
3. Segmentación por párrafos.
4. Fusión en chunks con límites de tamaño (`min_tokens=80`, `max_tokens=350`).

¿Por qué esta estrategia?
- Mantiene coherencia semántica por bloque temático.
- Evita chunks demasiado largos o demasiado pobres en contexto.
- Mejora la recuperación en FAQs, donde el contexto está altamente estructurado.

## Embeddings (decisiones técnicas)
Backends soportados:
1. **OpenAI embeddings** (si hay API key y se usa `--use-openai`)
2. **Local offline embeddings** con `HashingVectorizer` (fallback robusto)

Razón de diseño:
- El modo local garantiza ejecución sin depender de API externa.
- El modo OpenAI permite mejor calidad semántica en producción.

## Agente evaluador (bonus)
`src/query.py` incluye `ResponseEvaluator` que puntúa respuesta de 0–10 según:
- relevancia,
- completitud,
- precisión,
- claridad.

Puede operar:
- con heurísticas locales,
- o con LLM (si se habilita OpenAI).

## Formato de salida esperado
Ejemplo simplificado:

```json
{
  "user_question": "How do I add a new team member?",
  "system_answer": "...",
  "chunks_related": [
    {
      "rank": 1,
      "section": "...",
      "topic": "...",
      "relevance_score": 64.61,
      "content_preview": "..."
    }
  ]
}
```

## Manejo de errores implementado
- Validación de existencia del documento fuente.
- Manejo de fallback cuando faltan dependencias opcionales (FAISS, sentence-transformers).
- Respuestas de error para preguntas vacías.
- Logging detallado para trazabilidad de pipeline.

## Archivo de evidencia
- `outputs/sample_queries.json` contiene al menos 3 consultas de ejemplo con salida JSON completa.
