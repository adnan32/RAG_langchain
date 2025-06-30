import fitz
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    NLTKTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from difflib import SequenceMatcher

splitter_classes = {
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    "TokenTextSplitter": TokenTextSplitter,
    "NLTKTextSplitter": NLTKTextSplitter,
    "MarkdownHeaderTextSplitter": MarkdownHeaderTextSplitter,
}

embedding_classes = {
    "all-MiniLM-L6-v2": lambda: HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    ),
    "BAAI/bge-base-en-v1.5": lambda: HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"}
    ),
    "intfloat/e5-large": lambda: HuggingFaceEmbeddings(
        model_name="intfloat/e5-large",
        model_kwargs={"device": "cpu"}
    ),
}


class RagPipeline():
    def __init__(self):
        self.embeddings = None 
        self.vectorstore = None
        self.qa_chain = None
        self.rewriter = None           
        self.doc_summary = None         
        self.retriever_type = "basic"
        self.number_documents = 5
        self.chain_type = "stuff"
        self.split_strategy = "RecursiveCharacterTextSplitter"
        self.embedding_model = "all-MiniLM-L6-v2"
        self.vectorstore_type = "Chroma"

    def extract_text(self, uploaded_file):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        raw_text = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                raw_text += page_text + "\n\n"
        cleaned_text = (
            raw_text
            .replace("\r\n", "\n")
            .replace("\n", " ")
            .replace("  ", " ")
            .strip()
        )
        return cleaned_text

    def add_file(self, file, split_strategy=None, embedding_model=None, vectorstore=None):
        split_strategy  = split_strategy  or self.split_strategy
        embedding_model = embedding_model or self.embedding_model
        vectorstore     = vectorstore     or self.vectorstore_type

        # 1) extract & store text
        text = self.extract_text(file)
        # keep a short summary for rewriting
        self.doc_summary = text[:1000]

        # 2) split into chunks
        SplitterClass = splitter_classes[split_strategy]
        if split_strategy == "MarkdownHeaderTextSplitter":
            splitter = SplitterClass(headers_to_split_on=["#", "##", "###"])
        else:
            splitter = SplitterClass(chunk_size=1000, chunk_overlap=200)
        def is_similar(a: str, b: str, thresh: float = 0.8) -> bool:
            return SequenceMatcher(None, a, b).ratio() > thresh

        raw_chunks = splitter.split_text(text)

        unique_chunks = []
        for c in raw_chunks:
            # compare to all previously accepted chunks
            if any(is_similar(c, uc) for uc in unique_chunks):
                continue
            unique_chunks.append(c)

        chunks = unique_chunks

        # 3) embeddings + vectorstore
        if self.embeddings is None:
            self.embeddings = embedding_classes[embedding_model]()
        if vectorstore == "Chroma":
            name = f"rag_{embedding_model.replace('/', '_')}"
            self.vectorstore = Chroma.from_texts(
                chunks, self.embeddings,
                collection_name=name,
                persist_directory=f"./chroma_db_{embedding_model.replace('/', '_')}"
            )
        else:  # FAISS
            self.vectorstore = FAISS.from_texts(chunks, self.embeddings)

        return text

    def query(self, question, retriever_type=None, number_documents=None, chain_type=None):
        if self.vectorstore is None:
            return "⚠️ No document loaded."

        llm = OllamaLLM(model="deepseek-r1:32b")

        # 1) build base retriever
        if retriever_type == "basic":
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": number_documents})
        elif retriever_type == "multi_query":
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": number_documents}),
                llm=llm,
                verbose=True
            )
        else:  # compression
            compressor = LLMChainExtractor.from_llm(llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": number_documents})
            )

        # 2) initialize the rewriter chain once
        if self.rewriter is None:
            refine_template = PromptTemplate(
                input_variables=["doc_summary", "user_question"],
                template=(
                    "You are performing a semantic search over a document whose summary is:\n"
                    "{doc_summary}\n\n"
                    "Rewrite the user’s question into a short, focused search query.\n"
                    "Question: {user_question}\n"
                    "Refined query:"
                )
            )
            self.rewriter = LLMChain(llm=llm, prompt=refine_template)

        # 3) rewrite the user’s question
        refined_q = self.rewriter.run({
            "doc_summary": self.doc_summary,
            "user_question": question
        }).strip()

        # 4) retrieve using the refined query
        docs = retriever.get_relevant_documents(refined_q)

        # 5) run the QA chain (you can pass refined_q or original question here)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type=chain_type
        )
        answer = self.qa_chain.run(refined_q)

        return answer, docs
