import streamlit as st
from rag import RagPipeline

##########################################################
st.set_page_config(page_title=" Local RAG ", layout="wide")

# navigation state
if "page" not in st.session_state:
    st.session_state.page = "setup"

# RAG pipeline state
if "rag" not in st.session_state:
    st.session_state.rag = RagPipeline()
    st.session_state.file_uploaded = False
    st.session_state.config = {}

##########################################################
def show_setup():
    st.title("📄🔍 Local RAG Q&A")

    ##########################################################
    uploaded_file = st.file_uploader("Upload a PDF or text file here ", type=["txt","pdf"])
    ##########################################################
    splitter_options = {
        "RecursiveCharacterTextSplitter": " Recursive — smart paragraph/sentence-based splitting",
        "TokenTextSplitter": " Token — splits by model token count (good for exact control)",
        "NLTKTextSplitter": " NLTK — splits using natural language sentence detection",
        "MarkdownHeaderTextSplitter": " Markdown — splits by headers like #, ##, ### (useful for docs)",
    }
    splitter_display_names = list(splitter_options.values())
    selected_display = st.selectbox("Choose a splitting strategy:", splitter_display_names)
    split_strategy = [key for key, value in splitter_options.items() if value == selected_display][0]
    st.session_state.config["split_strategy"] = split_strategy
    ##########################################################
    embedding_options = {
        "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 — Fast, lightweight, good accuracy",
        "BAAI/bge-base-en-v1.5": "bge-base-en-v1.5 — Great for semantic search and question answering",
        "intfloat/e5-large": " e5-large — High-quality embeddings, slower but precise",
    }

    embedding_names = list(embedding_options.values())
    selected_embedding = st.selectbox("Choose a Embedding model: ", embedding_names)
    embedding_model = [model for model, dis in embedding_options.items() if dis == selected_embedding][0]
    st.session_state.config["embedding_model"] = embedding_model
    ##########################################################
    vectorstore_options = {
        "Chroma": "Chroma — Simple local vector DB, stores text + metadata",
        "FAISS": "FAISS — Fast local storage, ideal for prototyping",
    }

    vectorstore_names = list(vectorstore_options.values())
    selected_vectorstore = st.selectbox("Choose a Vector store: ", vectorstore_names)
    vectorstore = [vector for vector, dis in vectorstore_options.items() if dis == selected_vectorstore][0]
    st.session_state.config["vectorstore"] = vectorstore
    ##########################################################
    retriever_options = {
        "basic": "Basic Vector Search — standard cosine similarity",
        "multi_query": "Multi-Query — expands query into multiple versions for deeper search",
        "compression": "Compression — summarize context before passing to the LLM (compact input)",
    }
    retriever_names = list(retriever_options.values())
    selected_retriever = st.selectbox("Chooes a Retriever option: ", retriever_names)
    retriever = [ret for ret, dis in retriever_options.items() if dis == selected_retriever][0]
    st.session_state.config["retriever"] = retriever
    ##########################################################
    number_documents = st.number_input("Number of documents to retrieve", step=1)
    st.session_state.config["k"] = number_documents
    ##########################################################
    chain_type_options = {
        "stuff": "stuff- Concatenates all retrieved documents into a single prompt (simple & fast)",
        "map_reduce": "map_reduce- Runs LLM on each chunk independently (map), then summarizes (reduce). Better for long docs",
        "refine": "refine- Iteratively builds an answer. Starts with one doc, refines answer with each new chunk",
        "map_rerank": "map_rerank- Ranks chunks based on relevance and picks best one. Great for precision",
    }
    chain_type_names = list(chain_type_options.values())
    selected_chain_type = st.selectbox("Chooes a Chain type: ", chain_type_names)
    chain_type = [chain for chain, dis in chain_type_options.items() if dis == selected_chain_type][0]
    st.session_state.config["chain_type"] = chain_type
    ##########################################################

    if uploaded_file is not None:
        file_type = uploaded_file.type
        text_file=st.session_state.rag.extract_text(uploaded_file.name)
        st.write(text_file)
        # if file_type == "application/pdf":
        #     try:
        #         text_file = st.session_state.rag.add_file(
        #             uploaded_file,
        #             split_strategy,
        #             embedding_model,
        #             vectorstore
        #         )
        #         st.session_state.file_uploaded = True
        #         st.text_area("Extracted PDF text", text_file, height=200)
        #         st.success(f"✅ {uploaded_file.name} processed and ready!")
        #     except Exception as e:
        #         st.error(f"Failed to extract text from PDF: {e}")
        # else:
        #     st.write("Faild to upload a valid file type")

    # proceed button
    if st.session_state.file_uploaded:
        if st.button("Continue to Chat"):
            st.session_state.page = "chat"


def show_chat():
    st.title("📄🔍 Local RAG Chat")

    if not st.session_state.file_uploaded:
        st.error("No document loaded. Go back to setup.")
        return

    question = st.chat_input("Ask a question about your document")

    if question is not None:
        with st.chat_message("user"):
            st.write(question)

        answer, retrieved_docs = st.session_state.rag.query(
            question,
            st.session_state.config.get("retriever"),
            st.session_state.config.get("k"),
            st.session_state.config.get("chain_type")
        )

        st.subheader("📄 Retrieved chunks")
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}**\n> {doc.page_content.strip()}")

        with st.chat_message("assistant"):
            st.write(answer)

    if st.button("← Back to Setup"):
        st.session_state.page = "setup"


# page routing
if st.session_state.page == "setup":
    show_setup()
else:
    show_chat()
