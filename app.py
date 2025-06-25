import os
import sys
import tempfile
import typing
import logging
import json
import streamlit as st
import tika
from tika import parser
from io import StringIO
import faiss
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain import PromptTemplate, LLMChain
import requests
from pydantic import PrivateAttr
import fitz  # PyMuPDF

# Suppress Tika logs
logging.getLogger("tika").setLevel(logging.WARNING)

# Configuration
EMBEDDING_MODEL = "hkunlp/instructor-large"
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
TOP_K = 5

# extract_text, chunk_text, prepare_docs, build_faiss_index, build_rag_chain, ask_question,
# build_metadata_chain remain as before (see your existing code), with your prompt adjustments.

# Example for RAG chain builder:
class MistralHTTPLLM(LLM):
    _endpoint: str = PrivateAttr()
    _headers: dict = PrivateAttr()
    def __init__(self, token: str, model: str = "mistralai/Mistral-7B-Instruct-v0.3", **kwargs):
        super().__init__(**kwargs)
        self._endpoint = f"https://api-inference.huggingface.co/models/{model}"
        self._headers = {"Authorization": f"Bearer {token}"}
    def _call(self, prompt: str, stop=None) -> str:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.5}}
        resp = requests.post(self._endpoint, headers=self._headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise ValueError(f"Inference error: {data['error']}")
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    @property
    def _llm_type(self) -> str:
        return "mistral-http-inference"

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    else:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        buf = StringIO()
        sys.stdout = buf; sys.stderr = buf
        try:
            content = parser.from_file(file_path).get("content", "") or ''
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        return content

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)

def prepare_docs(files: typing.List[str]):
    docs, meta = [], []
    for fp in files:
        content = extract_text(fp)
        if not content.strip():
            continue
        for i, c in enumerate(chunk_text(content)):
            docs.append(c)
            meta.append({"source": os.path.basename(fp), "chunk": i})
    return docs, meta

def build_faiss_index(docs: typing.List[str], metas: typing.List[dict]):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  model_kwargs={"device": DEVICE})
    db = FAISS.from_texts(docs, embed, metadatas=metas)
    db.save_local("faiss_index")
    return db

def build_rag_chain():
    token = "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"
    if not token:
        raise ValueError("Set HUGGINGFACEHUB_API_TOKEN in your environment")
    llm = MistralHTTPLLM(token=token)
    prompt = PromptTemplate(
        template="Based on the context below, answer the question. Return only the answer text.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )
    return LLMChain(llm=llm, prompt=prompt, verbose=False)

def ask_question(chain, db, question: str):
    docs = db.similarity_search(question, k=TOP_K)
    if not docs:
        return "No relevant context found."
    context = "\n---\n".join([d.page_content for d in docs])
    return chain.run(context=context, question=question)

def build_metadata_chain(llm) -> LLMChain:
    metadata_prompt = PromptTemplate(
        template="""
Document content:
{content}

Extract metadata in JSON with these fields:
Title, Summary, Author, Creation date, Last modified date,
Publication date, Identifier, Keywords/Tags, Subjects/Topics, Language, Version,
Publisher/Source, Format, Rights/License, Contributor, Coverage, Provenance,
Relation, Audience/Intended use, Access controls, Technical details, Metadata versioning/timestamps.

Return only the JSON object.
""".strip(),
        input_variables=["content"]
    )
    return LLMChain(llm=llm, prompt=metadata_prompt, verbose=False)

def run_pipeline(file_paths: typing.List[str], question: str):
    docs, metas = prepare_docs(file_paths)
    if not docs:
        return "No text extracted from uploaded files."
    db = build_faiss_index(docs, metas)
    chain = build_rag_chain()
    return ask_question(chain, db, question)

def generate_metadata_for_files(file_paths: typing.List[str]):
    token = "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"
    if not token:
        raise ValueError("Set HUGGINGFACEHUB_API_TOKEN in your environment")
    
    llm = MistralHTTPLLM(token=token)
    metadata_chain = build_metadata_chain(llm)

    results = {}
    for fp in file_paths:
        text = extract_text(fp)
        if not text.strip():
            continue

        # If too long, summarize into sentences
        content_for_meta = text
        if len(text) > 20000:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            parts = splitter.split_text(text)
            # Summarization chain
            sum_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    template="Summarize the following text in one sentence:\n\n{text}",
                    input_variables=["text"]
                ),
                verbose=False
            )
            summaries = [sum_chain.run(text=part) for part in parts]
            content_for_meta = "\n".join(summaries)

        # Run metadata extraction
        try:
            meta_json_str = metadata_chain.run(content=content_for_meta)
        except Exception as e:
            # If the chain itself errors, store the error message
            results[os.path.basename(fp)] = f"Metadata generation error: {e}"
            continue

        # Isolate the JSON object between the first '{' and last '}'
        start = meta_json_str.find('{')
        end = meta_json_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_part = meta_json_str[start:end+1]
        else:
            json_part = meta_json_str

        # Try parsing
        try:
            parsed = json.loads(json_part)
            results[os.path.basename(fp)] = parsed
        except Exception:
            # Fallback: return the raw json_part string
            results[os.path.basename(fp)] = json_part

    return results

def main():
    st.title("📚 Automated Metadata Generator with a question-answer pipeline using RAGs")
    uploaded = st.file_uploader(
        "Upload one or more documents",
        type=["pdf","docx","txt"],
        accept_multiple_files=True
    )
    question = st.text_input("Ask your question:")
    # Prepare temp files once per run
    temp_files = []
    if uploaded:
        for up in uploaded:
            suffix = os.path.splitext(up.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.getbuffer())
            tmp_path = tmp.name
            tmp.close()
            temp_files.append(tmp_path)

    col1, col2 = st.columns(2)
    if col1.button("Get Answer"):
        if not temp_files or not question.strip():
            st.error("Upload files and enter a question first.")
        else:
            with st.spinner("Running RAG..."):
                try:
                    ans = run_pipeline(temp_files, question)
                    marker = f"Question: {question}"
                    idx = ans.find(marker)
                    if idx != -1:
                        trimmed = ans[idx:]
                    else:
                        trimmed = ans
                    st.markdown(f"Answer:")
                    st.write(trimmed)
                except Exception as e:
                    st.error(f"Error in answer pipeline: {e}")

    if col2.button("Generate Metadata"):
        if not temp_files:
            st.error("Upload files first.")
        else:
            with st.spinner("Extracting metadata..."):
                try:
                    metas = generate_metadata_for_files(temp_files)
                    for name, meta in metas.items():
                        st.markdown(f"### Metadata for {name}")
                        if isinstance(meta, dict):
                            st.json(meta)
                        else:
                            st.code(meta, language="json")
                except Exception as e:
                    st.error(f"Error in metadata pipeline: {e}")

    # Cleanup temp files after operations
    # Note: Streamlit reruns script on each interaction; we can clean up at end of run
    for fp in temp_files:
        try:
            os.unlink(fp)
        except Exception:
            pass

if __name__ == "__main__":
    main()
