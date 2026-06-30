import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Ethics of AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Ethics of AI RAG Chatbot")
st.write("Ask questions about the Ethics of AI document.")


@st.cache_resource
def load_rag_pipeline():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Get the absolute path to the ChromaDB folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "chroma_db_phase4")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    groq_api_key = st.secrets.get("GROQ_KEY")

    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_KEY not found. Please add it in Streamlit Secrets.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0
    )

    return retriever, llm


def ask_question(question, retriever, llm):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:
"I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": docs
    }


try:

    retriever, llm = load_rag_pipeline()

    question = st.text_input("Enter your question:")

    if question:

        with st.spinner("Searching documents and generating answer..."):

            result = ask_question(
                question,
                retriever,
                llm
            )

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(result["sources"], 1):

                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)
                st.divider()

except Exception as e:
    st.error(f"App Error: {e}")
