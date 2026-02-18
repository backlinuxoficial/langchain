# LangChain - Exemplo RAG com PDF e Gemini/Google Generative AI

Descrição
- Projeto de exemplo que demonstra um fluxo RAG (Retrieval-Augmented Generation):
  - Carrega um PDF (`A-ARTE-DA-GUERRA.pdf`), divide em chunks e gera embeddings locais com `sentence-transformers/all-MiniLM-L6-v2`.
  - Persiste vetores em um banco Chroma (`./chroma.db`) e usa um retriever para recuperar trechos relevantes.
  - Envia contexto ao modelo de LLM (ex.: Gemini via `google-genai`) para gerar respostas baseadas no contexto recuperado.

Conteúdo importante
- Arquivo principal: `app.py` — pipeline de carregamento, chunking, indexação e geração de respostas.
- Arquivo de exemplo: `models.py` — exibe como listar modelos com `google-genai`.
- Dependências: `requirements.txt`.

Clonar o repositório
```bash
git clone https://github.com/backlinuxoficial/langchain.git
cd langchain
```

Instalação e ambiente
- Recomendo criar um virtualenv e instalar dependências:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

Configuração
- Configure a chave de API do Google (Gemini / Generative AI) via variável de ambiente:
```bash
# Windows (PowerShell)
$env:GOOGLE_API_KEY = "SEU_GOOGLE_API_KEY"
# macOS / Linux
export GOOGLE_API_KEY="SEU_GOOGLE_API_KEY"
```
- Você também pode usar um arquivo `.env` e carregar as variáveis no código (não incluído aqui).

Uso
1. Coloque o PDF `A-ARTE-DA-GUERRA.pdf` na raiz do projeto (ou ajuste o caminho em `app.py`).
2. Rode o script principal:
```bash
python app.py
```
- O script criará/carregará o banco vetorial em `./chroma.db`.
- Se o banco já estiver populado, o script pulará a re-indexação.

Observações
- `app.py` usa embeddings locais via `HuggingFaceEmbeddings` (modelo `sentence-transformers/all-MiniLM-L6-v2`).
- A integração com LLM é feita via `ChatGoogleGenerativeAI` (Gemini) — ver `GOOGLE_API_KEY`.
- `models.py` contém um pequeno exemplo de uso da biblioteca `google-genai` para listar modelos.

Problemas comuns
- Falha ao conectar com a API do Google: verifique `GOOGLE_API_KEY` e permissões do projeto Google Cloud.
- Erros de dependência: confirme que `pip install -r requirements.txt` terminou sem erros.

Contribuição
- Pull requests e issues são bem-vindos.

Licença
- Não especificada no repositório. Use conforme necessidade.
