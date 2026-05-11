# AI Learning Journey

My learning path toward AI/LLM Application Developer.
Built with Python, OpenAI API, Google Gemini, and LangChain.

---

## Projects

### 1. Chatbots

#### [chatbot_gem.py](chatbot_gem.py) — Gemini Chatbot
Multi-turn chatbot using Google Gemini instead of OpenAI.
**Stack:** google-genai, python-dotenv

#### [chatbot_v2.py](chatbot_v2.py) — Multi-turn Chatbot
Basic conversational chatbot with system prompt and conversation history (cooking assistant).
**Stack:** OpenAI API, python-dotenv

#### [chatbot_v3.py](chatbot_v3.py) — Chatbot with Utility Commands
Multi-turn chatbot extended with slash commands for summarization, translation, and keyword extraction.
**Commands:** `/shrnout <text>` · `/prelozit <text>` · `/klicovaslova <text>`
**Stack:** OpenAI API, python-dotenv

---

### 2. LangChain

#### [langchain_intro.py](langchain_intro.py) — First LangChain Chain
Minimal LangChain pipeline using LCEL (`|` operator).
**Stack:** LangChain, ChatPromptTemplate, ChatOpenAI

#### [langchain_conversation.py](langchain_conversation.py) — Conversational Chain with Memory
Multi-turn chatbot with session-based memory using `RunnableWithMessageHistory`.
**Stack:** LangChain, ChatOpenAI

#### [langchain_pipeline.py](langchain_pipeline.py) — Sequential Pipeline
Two-step pipeline: topic extraction → explanation generation.
**Stack:** LangChain LCEL, StrOutputParser

---

### 3. RAG — Retrieval-Augmented Generation ⭐

#### [rag_indexer.py](rag_indexer.py) — Document Indexer
Loads `dokument.txt`, splits it into chunks, generates embeddings, and saves a FAISS index to disk.

#### [rag_query.py](rag_query.py) — Basic RAG
Loads an existing FAISS index and answers questions. No conversation memory.

#### [rag_query2.py](rag_query2.py) — Improved RAG
Accepts a document path as argument (indexes on the fly) or loads an existing index.

#### [rag_queryQA.py](rag_queryQA.py) — RAG with Conversation Memory ⭐
Full RAG chatbot that remembers conversation context across questions. Best version.

**How RAG works:**
```
document → chunks → embeddings → FAISS index → question → retrieved context → answer
```

**Stack:** LangChain, OpenAI Embeddings, FAISS, RunnableWithMessageHistory

---

## Sample Documents

- [dokument.txt](dokument.txt) — text about RAG and LLMs (used by `rag_indexer.py` by default)
- [dokument2.txt](dokument2.txt) — text about LangChain (pass as argument to `rag_query2.py` / `rag_queryQA.py`)

---

## Usage

```bash
# Chatbot with utility commands (OpenAI)
python chatbot_v3.py

# Chatbot using Google Gemini
python chatbot_gem.py

# LangChain intro (one-shot)
python langchain_intro.py

# LangChain sequential pipeline (one-shot)
python langchain_pipeline.py

# LangChain conversational chatbot
python langchain_conversation.py

# RAG — index the default document (creates faiss_index/)
python rag_indexer.py

# RAG — ask questions (load existing index)
python rag_queryQA.py

# RAG — index a new document and ask questions
python rag_queryQA.py dokument2.txt
```

---

## Example

```
$ python rag_queryQA.py dokument.txt

Načteno 8 chunků z dokument.txt

Otázka: Co je RAG?

Odpověď: RAG (Retrieval-Augmented Generation) kombinuje vyhledávání
v dokumentech s generováním textu. Místo aby model odpovídal pouze
ze své tréninkové paměti, nejprve vyhledá relevantní informace
z externích dokumentů.
```