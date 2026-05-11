from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import sys

load_dotenv()

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

def build_vectorstore(filepath):
    """Načte dokument, rozdělí na chunky, uloží do FAISS."""
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    )
    chunks = splitter.split_documents(documents)
    print(f"📄 Načteno {len(chunks)} chunků z {filepath}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def load_vectorstore():
    """Načte existující FAISS index z disku."""
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Index načten z disku — {vectorstore.index.ntotal} vektorů")
    return vectorstore

def get_rag_chain(vectorstore):
    """Sestaví RAG pipeline."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Odpovídej pouze na základě poskytnutého kontextu.
Pokud odpověď v kontextu není, řekni 'Tuto informaci v dokumentu nemám.'

Kontext:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | parser
    )

# --- Hlavní logika ---
# Spuštění s dokumentem:  python rag_query.py dokument.txt
# Spuštění bez argumentu: načte existující index

if len(sys.argv) > 1:
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"❌ Soubor {filepath} nenalezen.")
        sys.exit(1)
    vectorstore = build_vectorstore(filepath)
else:
    vectorstore = load_vectorstore()

rag_chain = get_rag_chain(vectorstore)

print("\nRAG chatbot — ptej se na obsah dokumentu. Napiš 'konec' pro ukončení.\n")

while True:
    question = input("Otázka: ").strip()
    if not question:
        continue
    if question.lower() == "konec":
        break

    answer = rag_chain.invoke(question)
    print(f"\nOdpověď: {answer}\n")
    print("-" * 60)