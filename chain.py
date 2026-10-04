from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag import retriever

from dotenv import load_dotenv
load_dotenv()

llm = ChatAnthropic(model="claude-opus-5", max_tokens=16000)

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant. Answer based ONLY on the context below.
If the answer isn't in the context, say so.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the test center address?")
print(answer)