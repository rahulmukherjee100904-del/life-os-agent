import os
import json
import datetime
from typing import Annotated, TypedDict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ==========================================
# 1. LOCAL DATA STORES & TOOLS DEFINITION
# ==========================================

SCHEDULE_DB = []
EXPENSE_DB = []

@tool
def add_calendar_event(event: str, date_str: str, time_str: str = "12:00") -> str:
    """Schedules a personal event or task. Date format: YYYY-MM-DD, Time format: HH:MM."""
    item = {"event": event, "date": date_str, "time": time_str}
    SCHEDULE_DB.append(item)
    return f"Successfully added event '{event}' on {date_str} at {time_str} to calendar."

@tool
def get_calendar_events() -> str:
    """Retrieves all current scheduled events and tasks."""
    if not SCHEDULE_DB:
        return "Your calendar is currently empty."
    return json.dumps(SCHEDULE_DB, indent=2)

@tool
def log_expense(amount: float, category: str, description: str) -> str:
    """Logs a financial expense with amount, category, and description."""
    entry = {
        "amount": amount,
        "category": category,
        "description": description,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    EXPENSE_DB.append(entry)
    return f"Logged expense of ${amount:.2f} under '{category}' ({description})."

@tool
def calculate_expenses() -> str:
    """Calculates total spending and provides a category breakdown summary."""
    if not EXPENSE_DB:
        return "No expenses logged yet."
    total = sum(e["amount"] for e in EXPENSE_DB)
    categories = {}
    for e in EXPENSE_DB:
        cat = e["category"]
        categories[cat] = categories.get(cat, 0) + e["amount"]
    
    return json.dumps({"total_spent": total, "by_category": categories, "entries": EXPENSE_DB}, indent=2)

tools = [add_calendar_event, get_calendar_events, log_expense, calculate_expenses]

# ==========================================
# 2. LANGGRAPH RE-ACT AGENT GRAPH
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def build_agent():
    # Model Initialization (Requires OPENAI_API_KEY env var)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        system_prompt = SystemMessage(
            content=(
                "You are an autonomous LifeOS Agentic Assistant. Your mission is to help the user manage "
                "their daily calendar, track financial expenses, and structure tasks. "
                "Always reason step-by-step before invoking tools, and provide clear concise answers."
            )
        )
        response = llm_with_tools.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    # State Machine Construction
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

agent_executor = build_agent()

# ==========================================
# 3. FASTAPI BACKEND WEB SERVER
# ==========================================

app = FastAPI(title="LifeOS Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(query: UserQuery):
    try:
        inputs = {"messages": [HumanMessage(content=query.message)]}
        result = agent_executor.invoke(inputs)
        
        last_message = result["messages"][-1]
        
        # Extract execution steps/tool call history
        trace = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    trace.append({"type": "tool_call", "name": tc["name"], "args": tc["args"]})
            elif msg.type == "tool":
                trace.append({"type": "tool_result", "content": msg.content})

        return {
            "response": last_message.content,
            "trace": trace,
            "calendar": SCHEDULE_DB,
            "expenses": EXPENSE_DB
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_system:app", host="0.0.0.0", port=8000, reload=True)