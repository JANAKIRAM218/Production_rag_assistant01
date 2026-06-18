# test.py

from pathlib import Path

from rag_backend import (
    FAISS_PATH,
    DOCS_PATH,
    load_vectorstore,
    build_vectorstore,
    load_documents,
    build_retrievers,
    _build_chunks,
    _build_reranker,
    _build_llm,
    answer_question,
)


TEST_QUESTIONS = [

    # Leave Policy
    "What is the maternity leave policy?",
    "How many sick leaves are allowed?",
    "How many casual leaves are allowed?",
    "How many earned leaves are allowed?",
    "What is bereavement leave?",

    # WFH
    "How does work from home work?",
    "Can employees permanently work remotely?",

    # Compensation
    "How are bonuses calculated?",
    "What benefits do employees receive?",

    # Performance
    "How are performance reviews conducted?",
    "What is a PIP?",

    # IT Security
    "What are password requirements?",
    "Can I use a personal laptop for work?",

    # Separation
    "What is the notice period?",
    "What is full and final settlement?",

    # POSH
    "How can I report workplace harassment?",

    # Out Of Scope
    "Who won IPL 2025?",
    "What is Bitcoin?",
    "Write Python code for quicksort",
    "What is today's weather?"
]


def load_pipeline():

    print("=" * 80)
    print("LOADING RAG PIPELINE")
    print("=" * 80)

    if Path(FAISS_PATH).exists():

        print("[INFO] Loading FAISS index...")

        vectorstore = load_vectorstore(FAISS_PATH)

        docs = load_documents(DOCS_PATH)
        chunks = _build_chunks(docs)

    else:

        print("[INFO] Building FAISS index...")

        vectorstore, chunks = build_vectorstore(
            DOCS_PATH,
            FAISS_PATH
        )

    ensemble_retriever = build_retrievers(
        vectorstore,
        chunks
    )

    reranker = _build_reranker()

    llm = _build_llm()

    return {
        "ensemble_retriever": ensemble_retriever,
        "reranker": reranker,
        "llm": llm,
    }


def run_tests():

    pipeline = load_pipeline()

    print("\n")
    print("=" * 80)
    print("RUNNING TEST SUITE")
    print("=" * 80)

    for idx, question in enumerate(TEST_QUESTIONS, start=1):

        print("\n")
        print("=" * 80)
        print(f"TEST #{idx}")
        print("=" * 80)

        print(f"QUESTION:\n{question}")

        try:

            result = answer_question(
                query=question,
                ensemble_retriever=pipeline["ensemble_retriever"],
                reranker=pipeline["reranker"],
                llm=pipeline["llm"],
            )

            answer = result["answer"]

            print("\nANSWER:")
            print(answer)

            print("\nSOURCES:")

            if result["sources"]:

                for source in result["sources"]:

                    print(
                        f"- {source.get('policy_name')} "
                        f"(Page {source.get('page_number')})"
                    )

            else:

                print("No sources")

        except Exception as e:

            print(f"\nERROR: {e}")

    print("\n")
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()