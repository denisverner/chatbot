# AI Learning Journey

My learning path toward AI/LLM Application Developer.
Built with Python, OpenAI API, and LangChain.

---

## Projects

### chatbot.py — Multi-turn Chatbot
Conversational chatbot with system prompt, conversation history and utility commands.
**Commands:** `/shrnout <text>` · `/prelozit <text>` · `/klicovaslova <text>`
**Stack:** OpenAI API, python-dotenv

### langchain_intro.py — First LangChain Chain
Minimal LangChain pipeline using LCEL (`|` operator).
**Stack:** LangChain, ChatPromptTemplate, ChatOpenAI

### langchain_conversation.py — Conversational Chain with Memory
Multi-turn chatbot with session-based memory.
**Stack:** LangChain, RunnableWithMessageHistory

### langchain_pipeline.py — Sequential Pipeline
Two-step pipeline: topic extraction → answer generation.
**Stack:** LangChain LCEL, StrOutputParser

### rag_indexer.py + rag_query.py — RAG over Documents ⭐
Question answering over custom documents using Retrieval-Augmented Generation.
Supports any `.txt` file. Remembers conversation context across questions.

**How it works:**
document → chunks → embeddings → FAISS → question → retrieved context → answer

**Stack:** LangChain, OpenAI Embeddings, FAISS, RunnableWithMessageHistory

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-learning-journey
cd ai-learning-journey
pip install openai langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu python-dotenv
```

Create `.env` file:
OPENAI_API_KEY=your_key_here

---

## Usage

```bash
# Chatbot
python chatbot.py

# RAG — index a document
python rag_indexer.py

# RAG — ask questions (load existing index)
python rag_query.py

# RAG — ask questions over a new document
python rag_query.py dokument2.txt
```

---

## Example output
$ python rag_query.py dokument.txt

Načteno 8 chunků z dokument.txt

Otázka: Co je RAG?

Odpověď: RAG (Retrieval-Augmented Generation) kombinuje vyhledávání
v dokumentech s generováním textu. Místo aby model odpovídal pouze
ze své tréninkové paměti, nejprve vyhledá relevantní informace
z externích dokumentů.
