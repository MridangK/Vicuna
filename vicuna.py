from PyPDF2 import PdfReader

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

langchain.verbose = False


def main():


    # stable-vicuna through LlamaCpp
    # Download model manually at https://huggingface.co/TheBloke/stable-vicuna-13B-GGML/tree/main
    llm = LlamaCpp(
        model_path="stable-vicuna-13B.ggmlv3.q8_0.bin",
        stop=["### Human:"],
        verbose=True,
        n_ctx=2048,
        n_batch=512,
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    if "Helpful Answer:" in chain.llm_chain.prompt.template:
        chain.llm_chain.prompt.template = (
            f"### Human:{chain.llm_chain.prompt.template}".replace(
                "Helpful Answer:", "\n### Assistant:"
            )
        )

   
    pdf = "PE.pdf"

    if pdf:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Using https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 as embedding
        # (downloaded automatically)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        knowledge_base = Qdrant.from_texts(
            chunks,
            embeddings,
            location=":memory:",
            collection_name="doc_chunks",
        )

        user_question = "What is this document about"

        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=4)

            prompt_len = chain.prompt_length(docs=docs, question=user_question)
            if prompt_len > llm.n_ctx:
                print("Output too big")

            response = chain.run(input_documents=docs, question=user_question)
            print(response)



if __name__ == "__main__":
    main()