import logging
from typing import Literal
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
            streaming=True,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_schema")
    
    decision = router.invoke(f"""
    You are a routing classifier for an enterprise IT support assistant.

    Your task is to classify the user's message into exactly one route:

    1. "direct":
    Use ONLY for:
    - Greetings (Hi, Hello, Good morning).
    - Thanks and acknowledgements.
    - Casual conversations and simple small talk.
    - Simple conversational messages that do not require IT knowledge.

    2. "kb":
    Use for:
    - Company IT policies and procedures.
    - VPN, password resets, and MFA.
    - Laptop and device setup.
    - Software installation and access.
    - Email and network troubleshooting.
    - Cybersecurity and access management.
    - Technical questions and general IT-related information.
    - Any question requiring factual knowledge or explanation.
    - Questions that may require private knowledge base retrieval or web search.

    IMPORTANT RULES:
    - When uncertain, ALWAYS choose "kb".
    - Never choose "direct" for technical or informational questions.
    - Do not answer the user's question.
    - Return only a valid structured routing decision.

    User message: {state['question']}
    """)

    return {"source_used": decision.route, "trace": add_trace(state, f"Router → {decision.route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}

def grade_kb(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_schema"
    )

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in state["kb_docs"]
    )

    grade = grader.invoke(f"""
You are an evidence evaluator for an enterprise IT support assistant.

Your task is to determine whether the retrieved private company
knowledge base documents contain sufficient evidence to answer
the user's question accurately.

User question:
{state['question']}

Retrieved private KB evidence:
{context}

Evaluation criteria:

Return "good" ONLY when:
- The evidence is directly relevant to the question.
- It contains enough information to provide a clear and specific answer.
- Important steps, requirements, or policy details are supported by the evidence.
- The answer can be generated without making assumptions.

Return "weak" when:
- No relevant documents were retrieved.
- The evidence is incomplete or unrelated.
- Important information is missing.
- Answering would require guessing or inventing company-specific details.

IMPORTANT:
- Do not use your general knowledge to fill gaps.
- Do not assume that keyword similarity means sufficient evidence.
- When uncertain, always return "weak".

Return the structured result with exactly one grade:
"good" or "weak".
""")

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}"
        ),
    }

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_schema"
    )

    if not state["web_results"].strip():
        return {
            "web_grade": "weak",
            "trace": add_trace(
                state,
                "Web evidence grade → WEAK (no search results)"
            ),
        }

    grade = grader.invoke(f"""
You are an evidence evaluator for an enterprise IT support assistant.

Your task is to determine whether the external web search results
provide sufficient and reliable information to answer the user's question.

User question:
{state['question']}

External web evidence:
{state['web_results']}

Evaluation criteria:

Return "good" ONLY when:
- The search results are directly relevant to the question.
- The evidence contains enough specific information to answer confidently.
- The information is supported by the provided search results.
- The sources appear relevant and trustworthy.
- The answer can be generated without unsupported assumptions.

Return "weak" when:
- Search results are empty, unrelated, or irrelevant.
- Important information is missing.
- Results contain conflicting or insufficient information.
- The evidence does not support a reliable answer.
- Answering would require guessing.

IMPORTANT:
- Evaluate only the provided web evidence.
- Do not use your general knowledge to fill missing information.
- Do not treat keyword similarity as sufficient evidence.
- When uncertain, always return "weak".

Return the structured result with exactly one grade:
"good" or "weak".
""")

    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()}"
        ),
    }

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
You are a query optimization assistant for an enterprise IT support system.

Rewrite the user's question to improve:
1. Private knowledge base retrieval.
2. External web search.

Instructions:
- Preserve the original intent and meaning.
- Add relevant technical keywords and terminology where appropriate.
- Include useful product names, technologies, or error terminology
  only when supported by the original question.
- Remove unnecessary words and ambiguity.
- Do not answer the question.
- Do not introduce assumptions or unrelated information.
- Return ONLY the rewritten search query.
- Keep the query concise and specific.

Original question:
{state['question']}
""").content.strip()

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}"
        ),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in state["kb_docs"]
    )

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

Answer ONLY using the private company knowledge base provided below.

Instructions:
- Be concise, clear, and actionable.
- Present instructions as numbered steps when appropriate.
- Do not invent facts, procedures, or company policies.
- If the knowledge base does not contain the required information,
  clearly mention the limitation.
- Mention that the answer is based on the company's private knowledge base.
- Do not refer to external web searches.

User question:
{state['question']}

Private company knowledge base:
{context}
""").content

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": [],
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB"
        ),
    }

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

The private company knowledge base did not contain sufficient evidence.
Use ONLY the external web evidence provided below.

Instructions:
- Answer the user's question clearly and accurately.
- Do not invent facts or instructions.
- If the evidence does not fully answer the question, clearly mention the limitation.
- Provide step-by-step instructions when appropriate.
- Clearly state that the answer is based on external web information.
- Warn the user to validate instructions with their IT team before
  making changes to company-managed systems.
- Keep the response concise and professional.

User question:
{state['question']}

External web evidence:
{state['web_results']}
""").content

    return {
        "answer": answer,
        "source_used": "web_search",
        "citations": state.get("citations", []),
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH"
        ),
    }

def direct_answer(state: AgentState):
    answer = llm().invoke(f"""
You are a friendly enterprise IT support assistant.

Respond briefly and naturally to the user's conversational message.
This function is ONLY for greetings, thanks, acknowledgements, and casual small talk.
Do not provide technical instructions or make up company policies.

User message: {state['question']}
""").content

    return {
        "answer": answer,
        "source_used": "direct",
        "citations": [],
        "trace": add_trace(state, "Direct response → no retrieval"),
    }

def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable information in the company "
            "knowledge base or external sources to answer your question confidently. "
            "Please try rephrasing your question, provide more details, "
            "or contact the IT help desk for assistance."
        ),
        "source_used": "insufficient_evidence",
        "citations": [],
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence"
        ),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)
        
    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)

