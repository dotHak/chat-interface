from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
from typing import cast


from graph.shared import (
    gene,
    llm,
    HospitalSystemState,
    NextHospitalSystemState,
    HospitalSystem,
    PatientIntent,
    PotentialDoctors,
)


patient_intent_system_prompt = """
You are an AI assistant designed to detect the intent behind a patient's request. Classify the intent into one of the following categories:
if the user request is related to question about booking an appointment, it be should treated as hospital information or general information.

1. `booking-appointment`: The user is trying to book an appointment with a doctor or inquire about availability for booking.
2. `hospital-info`: The user is asking for information related to the hospital, such as services, doctor details, or facilities.
3. `general-info`: The user’s request is unrelated to the hospital or medical topics. This includes greetings, casual questions, or irrelevant queries.

Guidelines:
- Carefully analyze the context and keywords in the user's request to determine the intent."""


def detect_patient_intent(state: HospitalSystemState):
    query = state.get("query", "")

    structured_llm = llm.with_structured_output(PatientIntent)

    messages = [SystemMessage(content=patient_intent_system_prompt)] + [
        HumanMessage(content=query)
    ]

    intent = cast(PatientIntent, structured_llm.invoke(messages))

    return {"intent": intent.intent}


def should_continue_to_next_branch(state: HospitalSystemState):
    """Return the next node to execute"""

    intent = state.get("intent", None)

    if intent:
        if intent == "booking-appointment":
            return "preliminary_info_extraction"
        if intent == "hospital-info":
            return "retrieve_hospital_info"
        if intent == "general-info":
            return "general_info_response"

    # Otherwise end
    return "detect_patient_intent"


preliminary_info_system_prompt = """
You are an AI assistant tasked with extracting structured information from user queries related to hospital services and appointments. The current date and time is {todays_date_time}.  Use the following guidelines to extract details:

Information to Extract:
1. appointment_date: Extract the date of the appointment in yyyy-mm-dd format, if mentioned.
2. appointment_start_time: Extract the start time of the appointment in HH:MM:SS 24-hour format, if mentioned.
3. appointment_end_time: Extract the end time of the appointment in HH:MM:SS 24-hour format, if mentioned.
4. appointment_reason: If explicitly mentioned, use it. Otherwise, infer the reason based on symptoms or diseases mentioned in the query.
5. symptoms_description: Extract symptoms or diseases described in the query, if mentioned.
6. doctor_name: Extract the name of the doctor, if specified.
7. specialists: Identify and list the specialists explicitly mentioned in the query. This should be a list of strings.

Notes:
- Only include keys for fields explicitly mentioned or logically inferred.
"""


def extract_preliminary_info(state: HospitalSystemState):
    query = state.get("query", "")

    structured_llm = llm.with_structured_output(HospitalSystem)

    todays_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = [
        SystemMessage(
            content=preliminary_info_system_prompt.format(
                todays_date_time=todays_date_time
            )
        )
    ] + [HumanMessage(content=query)]

    info = cast(HospitalSystem, structured_llm.invoke(messages))

    next_state: NextHospitalSystemState = {}

    if info.appointment_date:
        next_state["appointment_date"] = info.appointment_date
        next_state["start_date"] = info.appointment_date
        next_state["end_date"] = info.appointment_date
    else:
        next_state["start_date"] = ""
        next_state["end_date"] = ""
        next_state["appointment_date"] = ""

    if info.appointment_start_time:
        next_state["start_time"] = info.appointment_start_time
    else:
        next_state["start_time"] = ""

    if info.appointment_end_time:
        next_state["end_time"] = info.appointment_end_time
    else:
        next_state["end_time"] = ""

    if info.appointment_reason:
        next_state["reason"] = info.appointment_reason
    else:
        next_state["reason"] = ""

    if info.doctor_name:
        next_state["doctor_name"] = info.doctor_name
    else:
        next_state["doctor_name"] = ""

    if info.specialists:
        next_state["specialists"] = info.specialists
        next_state["loading_message"] = "Finding doctors..."
        next_state["status"] = "running"
    else:
        next_state["specialists"] = []

    if info.symptoms_description:
        next_state["symptoms_description"] = info.symptoms_description
        next_state["loading_message"] = "Finding doctors..."
        next_state["status"] = "running"
    else:
        next_state["symptoms_description"] = ""

    return next_state


def should_continue_to_find_potential_doctors(state: HospitalSystemState):
    """Return the next node to execute"""

    doctor_name = state.get("doctor_name", None)
    specialists = state.get("specialists", [])
    symptoms_description = state.get("symptoms_description", None)
    appointment_date = state.get("appointment_date", None)

    if specialists or symptoms_description:
        return "find_potential_doctors"

    if doctor_name and appointment_date:
        return "find_doctor"

    return "ask_availability_details"


doctors_recommendation_system_prompt = """
You are an AI assistant tasked with recommending potential doctors based on symptoms or specialist information provided in the user's query.
Follow these rules to generate the response:
1. **Analyze Input**:
   - If a specialist is stated, recommend doctors specializing in that field.
   - If only symptom descriptions are provided, infer the specialist and recommend appropriate doctors. If the specialist is unclear, default to General Internal Medicine doctors.
   - If both a specialist and symptoms are provided, recommend a mixture of doctors covering both.

2. **Specialist Recommendations**:
   - Always explain what the recommended specialists do before listing potential doctors.
   - Avoid recommending surgeons, regardless of symptoms or specialist information.

3. **Output Format**:
   - Present a structured response with the following fields:
     - `doctors`: A list of potential doctors based on the input query. List up to 10 doctors if possible.
     - `prompt_before`: A message explaining the role of the recommended specialists and setting context for the doctor list.
     - `prompt_after`: A message encouraging the user to select a doctor from the list or ask for further assistance.

[DATA]
Symptoms descriptions: {symptoms_description}
Specialists: {specialists}

Search data:
{search_data}
"""


def find_potential_doctors(state: HospitalSystemState):
    specialists = state.get("specialists", [])
    symptoms_description = state.get("symptoms_description", None)

    specialist_search = ""
    symptoms_search = ""

    if specialists:
        specialist_search = gene.search(
            " OR ".join(specialists), k=15, q_filter={"section": "doctors"}
        )
        specialist_search = gene.format(specialist_search)

    if symptoms_description:
        symptoms_search = gene.search(symptoms_description, k=10)
        symptoms_search = gene.format(symptoms_search)

    general_search = gene.format(
        gene.search(
            "General Internal Medicine",
            k=5,
            q_filter={"tag": "general internal medicine"},
        )
    )

    formatted_system_prompt = doctors_recommendation_system_prompt.format(
        symptoms_description=symptoms_description,
        specialists=", ".join(specialists),
        search_data=specialist_search
        + "\n"
        + symptoms_search
        + "\n"
        + general_search,
    )

    structured_llm = llm.with_structured_output(PotentialDoctors)

    messages = [SystemMessage(content=formatted_system_prompt)]

    potential_doctors = cast(PotentialDoctors, structured_llm.invoke(messages))

    return {
        "doctors_list": potential_doctors.doctors,
        "response_before": potential_doctors.prompt_before,
        "response_after": potential_doctors.prompt_after,
        "response_type": "potential_doctors",
        "status": "stopped",
        "loading_message": "",
    }
