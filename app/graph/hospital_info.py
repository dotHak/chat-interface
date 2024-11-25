from graph.shared import llm, gene, HospitalSystemState
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

hospital_info_system_prompt = """You are an AI assistant providing hospital information and booking appointments. Format your responses clearly with markdown, highlighting all important information, and offer further assistance if needed. The current date and time is {todays_date_time}.
Based on the retrieved information below, respond appropriately:
{results}
"""


def hospital_chat_agent(state: HospitalSystemState):
    # Get the query and results from the state
    query = state.get("query", "")
    results = state.get("search_results", "")

    # Build the system message
    todays_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_message = hospital_info_system_prompt.format(
        todays_date_time=todays_date_time, results=results
    )

    # Add the system message and the user query to the messages
    user_message = HumanMessage(content=query)
    messages = (
        state["messages"]
        + [SystemMessage(content=system_message)]
        + [user_message]
    )

    # Invoke the model
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    # delete_messages = [RemoveMessage(id=message.id) for message in state["messages"][:-2]]

    return {
        "messages": [user_message, response],
        "response_type": "message",
        "loading_message": "",
        "status": "completed",
    }


def retrieve_hospital_info(state: HospitalSystemState):
    # Get the query from the state
    query = state["query"]

    # Search for the query
    results = gene.search(query, k=15)
    if results:
        results = gene.format(results)
    else:
        results = "No results found."

    return {
        "search_results": results,
        "loading_message": "Processing search results...",
        "status": "running",
    }
