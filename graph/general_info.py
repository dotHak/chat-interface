from langchain_core.messages import HumanMessage, SystemMessage
from graph.shared import llm, HospitalSystemState


general_info_system_prompt = """
You are an AI assistant designed to provide general information and handle queries politely. Present the information clearly in a well formatted markdown where necessary.
Follow these rules:

1. **If the user greets you (e.g., "Hi", "Hello") or expresses gratitude (e.g., "Thank you", "Thanks"):
   - Respond politely and appropriately.
   - Example:
     - Greeting: "Hello! How can I assist you today?"
     - Gratitude: "You're welcome! Let me know if there's anything else I can help you with."

2. **For all other general information queries unrelated to greetings or gratitude:
   - Politely inform the user that you are a system designed to assist users with King Faisal Hospital services.
   - Provide an overview of your capabilities:
     - Booking appointments.
     - Retrieving doctor or service information.
     - General hospital information.
   - Encourage the user to ask relevant questions.
"""


def general_info_response(state: HospitalSystemState):
    query = state.get("query", "")

    user_message = HumanMessage(content=query)
    messages = (
        state["messages"]
        + [SystemMessage(content=general_info_system_prompt)]
        + [user_message]
    )

    response = llm.invoke(messages)

    return {
        "messages": [user_message, response],
        "response_type": "message",
        "status": "completed",
        "loading_message": "",
    }
