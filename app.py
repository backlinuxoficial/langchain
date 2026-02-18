from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate  # ✅ CORRIGIDO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import time

# Carrega o Livro
loader = PyPDFLoader("A-ARTE-DA-GUERRA.pdf")
documents = loader.load()
print(f"Total de páginas: {len(documents)}")

# Divide em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"Total de chunks: {len(chunks)}")

# Embeddings locais (gratuito)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Teste de dimensões
v = embeddings.embed_query("machine learning")
print(f"Dimensões: {len(v)}")

# Cria/carrega o banco vetorial
vectorstore = Chroma(
    collection_name="arte_guerra",
    embedding_function=embeddings,
    persist_directory="./chroma.db"
)

n = vectorstore._collection.count()
print(f"Vetores já indexados: {n}")

# ✅ LÓGICA CORRIGIDA: só popula se estiver vazio
if n == 0:  # ✅ SE for zero, então precisa popular
    print("Populando banco vetorial...")
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            vectorstore.add_documents(batch)
            print(f"Lote {i//batch_size+1} indexado com {len(batch)} documentos")
            time.sleep(1)  # pausa preventiva
        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(45)
    print("População concluída!")
else:
    print("Banco já populado. Pulando indexação.")

# Pergunta do usuário
query = "Qual a estratégia definida no livro que pode ser aplicada no cotidiano?"

# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Recupera os 4 chunks mais relevantes
docs = retriever.invoke(query)
print(f"\n🔍 {len(docs)} chunks recuperados:\n")
for i, doc in enumerate(docs):
    pag = doc.metadata.get('page', '?')
    print(f"Chunk {i+1} - pág. {pag}:")
    print(doc.page_content[:150])
    print()

# ✅ PROMPT CORRIGIDO
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente especialista.
Use APENAS o contexto abaixo para responder.
Se não estiver no contexto, diga isso claramente.

Contexto: {context}"""),  # ✅ placeholder "context" (não "contexto")
    ("human", "{question}")
])

# Gemini Pro
os.environ["GOOGLE_API_KEY"] = "Sua API"
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.3
)

# Chain LCEL
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # ✅ "context" consistente
    | prompt_template
    | llm
    | StrOutputParser()
)

# Invoca a chain
print("\n🤖 Gerando resposta...\n")
print("="*60)
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)
print("\n" + "="*60)