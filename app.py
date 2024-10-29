from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from datetime import datetime
from typing import Any

import chainlit as cl
import requests
import os
import yaml

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY_SECRET = SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None
MOCK_HOSPITAL_SYSTEM_BASE_URL = (
    os.getenv("MOCK_HOSPITAL_SYSTEM_BASE_URL") or "https://mock-hospital-system.onrender.com"
)
ALLOW_DANGEROUS_REQUEST = True


chat_history = []


def get_mock_hostpital_system_openapi_spec() -> dict[str, Any]:
    res = requests.get(f"{MOCK_HOSPITAL_SYSTEM_BASE_URL}/api/doc")
    if res.status_code != 200:
        raise Exception(f"Error fetching openai spec from mock hospital system: {res.text}")
    return res.json()


def load_openai_spec():
    openai_path = "openapi_spec.yaml"
    if os.path.exists(openai_path):
        with open(openai_path, "r") as f:
            spec = yaml.load(f, Loader=yaml.Loader)
            spec["servers"] = [{"url": MOCK_HOSPITAL_SYSTEM_BASE_URL}]
            return reduce_openapi_spec(spec)
    else:
        openapi_spec = get_mock_hostpital_system_openapi_spec()
        with open(openai_path, "w") as f:
            yaml.dump(openapi_spec, f)

        openapi_spec["servers"] = [{"url": MOCK_HOSPITAL_SYSTEM_BASE_URL}]
        return reduce_openapi_spec(openapi_spec)


def print_message(message: str):
    print("Print =>", message)
    return message


@cl.on_chat_start
def setup_chain():
    llm = ChatOpenAI(api_key=OPENAI_API_KEY_SECRET, model="gpt-4", temperature=0)

    toolkit = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(headers={}),
        allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    )

    tools = toolkit.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    todays_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load your OpenAPI specification
    hospital_api_spec = yaml.dump(load_openai_spec(), sort_keys=False).replace("{id}", "{{id}}")
    system_prompt = f"""
    You are an AI assistant helping users with hospital services, especially for booking appointments and retrieving information about doctors and services. Todays date and time is {todays_date_time}.
    Here is documentation on the API:
    { hospital_api_spec }
    Use the Request Toolkit to interact with the hospital system API and assist users in booking appointments.

    When interacting with the user, follow these steps for appointment booking:

    1. **Detect Intent:**
        - If the user's request involves **booking an appointment** or related, proceed with the following steps.

    2. **Check for Required Information:**
        - Verify if the user has provided the **doctor's name** and a **date range** for the intended appointment.
        - If either the **doctor’s name** or **date range** is missing, highlighting the missing information and proceed to the next step,.

    3. **Search for Doctors:**
       - If the **doctor’s name** is not provided, ask the user:
           - "Please provide the name or specialty of the doctor you are looking for."
       - Use the provided input to **search for matching doctors**.
       - If multiple matches are found, present the list to the user and ask them to select the correct doctor.

    4. **Retrieve Doctor Availability:**
       - Once the doctor is identified, **use the doctor’s ID** to query the available appointment slots.
       - Present the available dates and times to the user in a readable format for selection(AM/PM).
            - Eg: Dr. John Doe is only available on:
                - 1. 31st August 2024 from 10:00 AM to 12:00 PM
                - 2. 31st September 2024 from 2:00 PM to 4:00 PM
        - Ask the user to select a suitable slot.

    5. **Prompt for Missing Information:**
       - If the user selects an available slot, ask them for any **missing details** required for the appointment.
       - State all the required information clearly.
            - Eg. Please provide the following details to book the appointment:
                - 1. Full Name
                - 2. Email Address
                - 3. Reason for Appointment(optional)

            - Prompt these questions **one-by-one** to avoid overwhelming the user.
                - Eg. "What is your full name?"

    6. **Show Appointment Details:**
        - Once all required information is collected, present a summary to the user in a readable format:
            - Here is the summary of your appointment:
                - Doctor: Dr. John Doe
                - Date: 31st August 2024
                - Time: 10:00 AM to 12:00 PM
                - Full Name: John Doe
                - Email: jd@gmail.com
                - Reason: Follow-up checkup

        - Ask the user if they would like to confirm the appointment or make any changes.

    7. **Ask for Final Confirmation:**
       - Ask the user:
         - "Would you like to confirm this appointment?"
       - If the user confirms, proceed to **book the appointment** through the hospital system.

    8. **Handle Errors or Missing Availability:**
        - If no available slots are found for the selected doctor, provide feedback to the user.
        - If the user provides incorrect or missing information, prompt them to provide the correct details.

    9. **Confirm Successful Booking:**
       - Once the appointment is successfully booked, inform the user

    NOTE: All times to the API must follow 24-hour format HH:MM:SS. For example, 3:00 PM should be sent as 15:00:00.
    NOTE: Until the user confirms the appointment, the booking process should be considered incomplete. Ask for confirmation at each step.
    """

    # Define the prompt to use your own questions and answers
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    cl.user_session.set("llm_chain", agent_executor)


@cl.on_message
async def handle_message(message: cl.Message):
    user_message = message.content
    llm_chain: Any = cl.user_session.get("llm_chain")

    result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

    chat_history.extend(
        [
            HumanMessage(content=user_message),
            AIMessage(content=result["output"]),
        ]
    )

    await cl.Message(result["output"]).send()
