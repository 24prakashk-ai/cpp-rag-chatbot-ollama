
from pathlib import Path

import streamlit as st

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="C++ RAG Chatbot",
    page_icon="💻",
    layout="centered"
)

st.title("💻 C++ RAG Chatbot")
st.write("Ask questions based on your C++ study material.")


# --------------------------------------------------
# 2. LOAD CPP2.TXT AND CREATE FAISS DATABASE
# --------------------------------------------------

@st.cache_resource
def load_db():

    # Get CPP2.txt from the same folder as this Python file
    data_path = Path(__file__).resolve().parent / "CPP2.txt"

    # Check if the file exists
    if not data_path.is_file():
        raise FileNotFoundError(
            f"CPP2.txt was not found at: {data_path}"
        )

    # Read the C++ text file
    text = data_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    if not text.strip():
        raise ValueError("CPP2.txt is empty.")

    # Convert text into a LangChain document
    text_documents = [
        Document(
            page_content=text,
            metadata={"source": str(data_path)}
        )
    ]

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_documents = text_splitter.split_documents(
        text_documents
    )

    if not final_documents:
        raise ValueError("No text chunks were created.")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    db = FAISS.from_documents(
        final_documents,
        embeddings
    )

    return db, len(final_documents), len(text)


# --------------------------------------------------
# 3. INITIALIZE THE DATABASE
# --------------------------------------------------

try:
    db, total_chunks, total_characters = load_db()

    st.success("C++ data loaded successfully!")

    st.caption(
        f"Characters: {total_characters} | "
        f"Text chunks: {total_chunks}"
    )

except Exception as e:
    st.error("Could not load the C++ database.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 4. CONNECT TO OLLAMA GEMMA 2B
# --------------------------------------------------

llm = ChatOllama(
    model="gemma2:2b",
    temperature=0
)


# --------------------------------------------------
# 5. CREATE THE PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful C++ programming assistant.

        Answer the user's question using the provided context.
        Explain concepts clearly and give examples when
        supported by the context.

        If the answer is not available in the provided
        context, clearly say that it was not found in
        the supplied C++ data."""
    ),
    (
        "user",
        """Context:
{context}

Question:
{question}

Answer:"""
    )
])


# --------------------------------------------------
# 6. CREATE THE RAG CHAIN
# --------------------------------------------------

output_parser = StrOutputParser()

chain = prompt | llm | output_parser


# --------------------------------------------------
# 7. USER INPUT
# --------------------------------------------------

question = st.text_input(
    "Enter your C++ question:",
    placeholder="Example: What is a constructor in C++?"
)


# --------------------------------------------------
# 8. RETRIEVE DATA AND GENERATE ANSWER
# --------------------------------------------------

if st.button("Submit", type="primary"):

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            with st.spinner(
                "Searching C++ data and generating answer..."
            ):

                # Retrieve the 3 most relevant chunks
                results = db.similarity_search(
                    question,
                    k=3
                )

                # Combine retrieved chunks into context
                context = "\n\n".join(
                    doc.page_content for doc in results
                )

                # Send context and question to Gemma
                answer = chain.invoke({
                    "context": context,
                    "question": question
                })

            # Display answer
            st.subheader("Answer")
            st.write(answer)

            # Display retrieved C++ text
            with st.expander("View retrieved C++ data"):

                for i, doc in enumerate(results, start=1):
                    st.markdown(f"**Retrieved chunk {i}**")
                    st.write(doc.page_content)
                    st.divider()

        except Exception as e:
            st.error("An error occurred while answering.")
            st.exception(e)
