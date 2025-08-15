from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are MediBot, an AI assistant. Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Use only the text in the context.
- Do NOT use prior knowledge or make assumptions.
- Do NOT provide medical advice or diagnoses.
- If the answer is not in the context, respond exactly:
"I'm sorry, but I couldn't find relevant information in the provided documents."
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
