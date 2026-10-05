import argparse

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag import COLLECTIONS, get_retriever

from dotenv import load_dotenv
load_dotenv()

llm = ChatAnthropic(model="claude-opus-5", max_tokens=16000)

prompt = ChatPromptTemplate.from_template("""
You are an expert support assistant. Answer based ONLY on the context below.
The context contains Jira tickets (past issues and how they were handled) and/or
SOP documents (the official procedures). Prefer the SOP when they disagree.
Cite the ticket keys and SOP titles you used. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}
""")


def format_doc(d):
    m = d.metadata
    if m.get("dataset") == "jira":
        label = f"[JIRA {m['ticket']} | status: {m.get('status', '?')}]"
    else:
        page = f" p.{m['page'] + 1}" if "page" in m else ""
        label = f"[SOP {m.get('title', m.get('source'))}{page}]"
    return f"{label}\n{d.page_content}"


def format_docs(docs):
    return "\n\n".join(format_doc(d) for d in docs)


def build_chain(dataset="all"):
    return (
        {"context": get_retriever(dataset) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question over the indexed datasets")
    parser.add_argument("question")
    parser.add_argument("--dataset", choices=[*COLLECTIONS, "all"], default="all")
    args = parser.parse_args()
    print(build_chain(args.dataset).invoke(args.question))
