from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import cast, Optional
from rapidfuzz import fuzz

import json

from graph.shared import (
    llm,
    gene,
    HospitalSystemState,
    DoctorAvailability,
    Doctor,
    Availability,
    AppointmentDate,
    AppointmentInfo,
    NextHospitalSystemState,
    ConfirmBooking,
)
from utils.api import check_doctor_availabity, book_appointment

START = "detect_patient_intent"

NextState = Optional[HospitalSystemState]

ask_availability_system_prompt = """You are an AI assistant helping patients book doctor appointments. Ask the patient for the`doctor's name`, `date` or `date range`, `symptoms description` or a `specialist` if they don't know the doctor's name. Ensure the patient provides all the necessary details to book an appointment. When key details like the `doctor's name` or the `date` are unknown, ask the patient to provide information in a clear and polite manner. If the doctors name is known, state the name in the prompt.
[Known details]:
{known_details}

[Unkown details]:
{missing_details}

Follow these guidelines:
1. Use well-structured **markdown** to make the prompt user-friendly and visually clear.
2. Provide examples for date inputs such as:
   - A single date: `20 Nov`, `25 Nov 2024`
   - A date range: `from 20 to 25 Nov`
   - Relative dates: `tomorrow`, `next week`
3. IF THE FIELD IS NOT IN UNKNOWN DETAILS, DO NOT ASK FOR IT.

If any field is unknown:
- **Doctor's Name**: Politely ask the patient to specify the doctor's name.
- **Specialist**: Politely ask the patient to specify the specialist if they don't know the doctor's name.
- **Symptoms Description**: Politely ask the patient to provide a brief description of their symptoms if they don't know the doctor's name.
- **Date or Date Range**: Politely ask the patient to provide the date or date range for the appointment.
"""


def ask_availability_details(state: HospitalSystemState):
    # Get the doctor name and dates from the state
    doctor_name = state.get("doctor_name", None)
    start_date = state.get("start_date", None)
    end_date = state.get("end_date", None)
    doctor_not_found = state.get("doctor_not_found", False)
    availability = state.get("availability", None)

    # Check for missing details
    missing_details = []
    known_details = []
    if not doctor_name:
        missing_details.append(" - Doctor's Name")
        missing_details.append(" - Specialist")
        missing_details.append(" - Symptoms description")
    elif doctor_name and doctor_not_found:
        missing_details.append(f" - {doctor_name} is not found.")
        missing_details.append(" - Specialist")
        missing_details.append(" - Symptoms description")
    else:
        known_details.append(f" - Doctor's Name: {doctor_name}")

    if not start_date or not end_date:
        missing_details.append(" - Date or Date Range")
    elif start_date and end_date and not availability:
        missing_details.append(
            f" - No availability found for {doctor_name} from {start_date} to {end_date}"
        )
    else:
        known_details.append(f" - Date: {start_date} to {end_date}")

    # Build the system message
    formatted_system_prompt = ask_availability_system_prompt.format(
        missing_details="\n".join(missing_details),
        known_details="\n".join(known_details),
    )

    messages = [SystemMessage(content=formatted_system_prompt)]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "response_type": "message",
        "status": "stopped",
    }


availability_chat_system_prompt = """You are an AI assistant tasked with extracting structured information from user queries related to doctor availability. The current date and time is {todays_date_time}. Ensure the extracted information is accurate and follows these guidelines:
1. Dates must follow the format `yyyy-mm-dd`.
2. If only one date is provided, set both the `start_date` and `end_date` to the same value.
3. Extract the following details, but ignore any that are not explicitly mentioned in the query:
   - `doctor_name`: The name of the doctor mentioned in the query include their title eg Dr. Sam, or exclude it if not provided.
   - `start_date` and `end_date`: The date range for the availability check, or exclude them if not provided.
   - symptoms_description: Extract symptoms or diseases described in the query, if mentioned.
   - specialists: Identify and list the specialists explicitly mentioned in the query. This should be a list of strings.
4. If the user show any indication of stopping or cancelling or requesting for information, set the `stop_procceing` true or false.
"""


def availability_chat_agent(state: HospitalSystemState):
    # Get the query from the state
    query = state.get("query", "")

    # Enforce structured output
    structured_llm = llm.with_structured_output(DoctorAvailability)

    todays_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_system_prompt = availability_chat_system_prompt.format(
        todays_date_time=todays_date_time
    )

    messages = [SystemMessage(content=formatted_system_prompt)] + [
        HumanMessage(content=query)
    ]

    # Invoke the model
    availability = cast(DoctorAvailability, structured_llm.invoke(messages))

    next_state: NextHospitalSystemState = {
        "messages": [HumanMessage(content=query)],
        "response_type": "message",
        "status": "running",
        "loading_message": "Verifying doctor's name...",
        "from_availability_agent": False,
    }

    if availability.doctor_name:
        next_state["doctor_name"] = availability.doctor_name

    if availability.start_date:
        next_state["start_date"] = availability.start_date

    if availability.end_date:
        next_state["end_date"] = availability.end_date

    if availability.specialists:
        next_state["specialists"] = availability.specialists
        next_state["loading_message"] = "Finding specialists..."
        next_state["from_availability_agent"] = True

    if availability.symptoms_description:
        next_state["symptoms_description"] = availability.symptoms_description
        next_state["loading_message"] = "Finding potential doctors..."
        next_state["from_availability_agent"] = True

    if availability.stop_processing:
        next_state["loading_message"] = "Stopping the process..."
        next_state["restart_graph"] = True
    else:
        next_state["restart_graph"] = False

    return next_state


def should_continue_to_find_doctor(state: HospitalSystemState):
    restart_graph = state.get("restart_graph", False)
    from_availability_agent = state.get("from_availability_agent", False)
    specialists = state.get("specialists", [])
    symptoms_description = state.get("symptoms_description", None)

    if from_availability_agent and (specialists or symptoms_description):
        return "find_potential_doctors"

    if restart_graph:
        return START

    return "find_doctor"


find_doctor_system_prompt = """
You are an AI assistant tasked with identifying a doctor's full name and title from the retrieved information below:
{results}.

Follow these rules:
1. If one of the results closely matches the query, extract the doctor's full name and title.
2. If none of the results match, set both `full_name` and `title` to `None`.
"""


def find_doctor(state: HospitalSystemState):
    # Get the doctor name from the state
    doctor_name = state.get("doctor_name", None)
    doctor = state.get("doctor", None)

    # We know the doctor
    if doctor:
        return {"response_type": "message"}

    # Search for the doctor
    results = gene.search(doctor_name, k=3, q_filter={"section": "doctors"})

    structured_llm = llm.with_structured_output(Doctor)

    formatted_system_prompt = find_doctor_system_prompt.format(
        results=gene.format(results)
    )

    doctor = cast(
        Doctor,
        structured_llm.invoke([SystemMessage(content=formatted_system_prompt)]),
    )

    next_state: NextHospitalSystemState = {
        "response_type": "message",
        "status": "running",
    }

    if (
        doctor.full_name
        and fuzz.ratio(doctor.full_name.lower(), doctor_name.lower()) > 30
    ):
        next_state["doctor"] = doctor
        next_state["doctor_not_found"] = False
        next_state["loading_message"] = "Checking for doctor's availability..."
    else:
        next_state["doctor"] = None
        next_state["doctor_not_found"] = True
        next_state["loading_message"] = "Asking for doctor's name again..."

    return next_state


get_availability_system_prompt = """
You are an AI assistant tasked with generating a structured response for a doctor's availability. Based on the availability data below:
{availability}

Follow these rules:
1. Check if the availability data is empty or populated.
   - If empty, set `response_type` to `"no-availability"` and generate an appropriate response to inform the user that no slots are available.
   - If populated, set `response_type` to `"availability-list"` and generate a clear response with a message before and after the list of available slots asking the user to select or enter a slot.
"""


def should_continue_to_check_availability(state: HospitalSystemState):
    """Return the next node to execute"""

    # Check if the doctor name and dates are provided
    doctor_name = state.get("doctor_name", None)
    start_date = state.get("start_date", None)
    end_date = state.get("end_date", None)
    doctor = state.get("doctor", None)

    if not doctor_name or not start_date or not end_date or not doctor:
        return "ask_availability_details"

    # Otherwise check the doctor's availability
    return "check_doctor_availability"


def check_doctors_availability(state: HospitalSystemState):
    name = cast(str, state.get("doctor").full_name)
    start_date = state.get("start_date")
    end_date = state.get("end_date")

    availability = check_doctor_availabity(name, start_date, end_date)
    doctor_id = availability.get("doctor_id", None)
    availability = availability.get("availability", [])
    formatted_system_prompt = get_availability_system_prompt.format(
        availability=json.dumps(availability, indent=2)
    )

    structured_llm = llm.with_structured_output(Availability)
    messages = [SystemMessage(content=formatted_system_prompt)]

    response = cast(Availability, structured_llm.invoke(messages))

    next_state = {
        "availability": availability,
        "doctor_id": doctor_id,
    }

    if response.response_type == "availability-list":
        next_state["response_type"] = "availability-list"
        next_state["response_before"] = response.response_before
        next_state["response_after"] = response.response_after
        next_state["status"] = "stopped"
        next_state["loading_message"] = ""
    else:
        next_state["messages"] = [AIMessage(content=response.response)]
        next_state["response_type"] = "message"
        next_state["status"] = "running"
        next_state["loading_message"] = "Asking for availability details..."

    return next_state


def should_continue_to_get_appointment_date_and_time(
    state: HospitalSystemState,
):
    """Return the next node to execute"""

    availability = state.get("availability", None)

    if not availability:
        return "ask_availability_details"

    # Otherwise end
    return "get_appointment_date_time"


get_appointment_date_system_prompt = """
You are an AI assistant tasked with extracting structured appointment date information from user queries. This is the initial start date {start_date} and end date {end_date}.
Follow these rules:
1. Extract the following fields if they are mentioned:
   - `date`: The date of the appointment in `yyyy-mm-dd` format.
   - `start_time`: The start time of the appointment in `HH:MM:SS` 24-hour format.
   - `end_time`: The end time of the appointment in `HH:MM:SS` 24-hour format.
2. If `date` is not provided, and only a time range is mentioned, set the `date` to the start date.
3. If any field is not present in the query, exclude it from the output.
4. If the user show any indication of stopping or cancelling or requesting for information, set the `stop_procceing` true or false.
"""


def get_appointment_date_time(state: HospitalSystemState):
    """Get the appointment date and time from the user"""
    query = state.get("query", "")
    start_date = state.get("start_date", "")
    end_date = state.get("end_date", "")

    structured_llm = llm.with_structured_output(AppointmentDate)
    formatted_system_prompt = get_appointment_date_system_prompt.format(
        start_date=start_date, end_date=end_date
    )

    messages = [SystemMessage(content=formatted_system_prompt)] + [
        HumanMessage(content=query)
    ]

    appointment_date = cast(AppointmentDate, structured_llm.invoke(messages))

    next_state: NextHospitalSystemState = {
        "messages": [HumanMessage(content=query)],
    }

    if appointment_date.date:
        next_state["appointment_date"] = appointment_date.date

    if appointment_date.start_time:
        next_state["start_time"] = appointment_date.start_time

    if appointment_date.end_time:
        next_state["end_time"] = appointment_date.end_time

    if len(next_state.keys()) >= 4:
        next_state["status"] = "running"
        next_state["loading_message"] = "Asking for appointment details..."
    else:
        next_state["status"] = "stopped"
        next_state["loading_message"] = ""

    if appointment_date.stop_processing:
        next_state["status"] = "running"
        next_state["restart_graph"] = True
        next_state["loading_message"] = "Stopping the process..."
    else:
        next_state["restart_graph"] = False

    return next_state


def should_continue_to_ask_appointment_info(state: HospitalSystemState):
    """Return the next node to execute"""

    appointment_date = state.get("appointment_date", None)
    start_time = state.get("start_time", None)
    end_time = state.get("end_time", None)
    restart_graph = state.get("restart_graph", False)

    if restart_graph:
        return START

    if not appointment_date or not start_time or not end_time:
        return "get_appointment_date_time"

    # Otherwise end
    return "ask_appointment_info"


ask_appointment_info_system_prompt = """
You are an AI assistant helping patients book doctor appointments. Your task is to ask patients to provide their `full name`, `email`, and `reason` for the appointment. Ask the patient for the information in a clear and polite manner.

[missing]:
{missing_details}

Follow these guidelines:
1. Use well-structured **markdown** to make the prompt user-friendly and visually clear.
2. List all the details that user needs to provide for the appointment.
3. If the is not in the missing details, do not ask for it.
4. Ask for one missing piece of information at a time to avoid overwhelming the user.
5. Do not ask to confirm provided details unless you are unsure. If unsure, state the provided details and ask for confirmation.
6. If multiple fields are missing, list the all missing details clearly.
    - State that they can provide all the details at once but ask for them one at a time.

Ensure the response is polite, concise, and easy to understand.
NEVER STATE THE DETAILS ARE MISSING, ASK FOR THEM INSTEAD.
"""


def ask_appointment_info(state: HospitalSystemState):
    name = state.get("patient_name", None)
    email = state.get("patient_email", None)
    reason = state.get("patient_reason", None)

    missing_details = []

    if not name:
        missing_details.append(" - Full Name")

    if not email:
        missing_details.append(" - Email")

    if not reason:
        missing_details.append(" - Reason")

    formatted_system_prompt = ask_appointment_info_system_prompt.format(
        missing_details="\n".join(missing_details)
    )

    messages = [SystemMessage(content=formatted_system_prompt)]

    response = llm.invoke(messages)

    if missing_details:
        status = "stopped"
        loading_message = ""
    else:
        status = "running"
        loading_message = "Asking for appointment confirmation..."

    return {
        "messages": [response],
        "response_type": "message",
        "status": status,
        "loading_message": loading_message,
    }


def should_continue_to_get_appointment_info(state: HospitalSystemState):
    """Return the next node to execute"""

    name = state.get("patient_name", None)
    email = state.get("patient_email", None)
    reason = state.get("patient_reason", None)

    if not name or not email or not reason:
        return "get_appointment_info"

    # Otherwise end
    return "ask_appointment_confirmation"


get_appointment_info_system_prompt = """
You are an AI assistant tasked with extracting structured appointment information from user queries. Extract the following details if they are provided:

1. `full_name`: The full name of the patient, if name is in the query update the field.
2. `email`: The email address of the patient.
3. `reason`: The reason for the appointment. The reason must be related to a symptom or disease mentioned in the query. if not mentioned, exclude it from the output. Eg. "Appointment with Dr. Sam" cannot be a reason. DO NOT include the doctor's name in the reason. AGAIN DO NOT ASSUME THE REASON UNLESS IT IS EXPLICITLY MENTIONED OR RELATED TO A SYMPTOM OR DISEASE.
4. If the user show any indication of stopping or cancelling or requesting for information, set the `stop_procceing` true or false.
If any field is not mentioned, exclude it from the output."""


def get_appointment_info(state: HospitalSystemState):
    query = state.get("query", "")

    structured_llm = llm.with_structured_output(AppointmentInfo)

    messages = [SystemMessage(content=get_appointment_info_system_prompt)] + [
        HumanMessage(content=query)
    ]

    appointment_info = cast(AppointmentInfo, structured_llm.invoke(messages))

    next_state: NextHospitalSystemState = {
        "messages": [HumanMessage(content=query)],
        "response_type": "message",
    }

    print(appointment_info)

    if appointment_info.full_name:
        next_state["patient_name"] = appointment_info.full_name

    if appointment_info.email:
        next_state["patient_email"] = appointment_info.email

    if appointment_info.reason:
        next_state["patient_reason"] = appointment_info.reason

    if len(next_state.keys()) >= 5:
        next_state["status"] = "running"
        next_state["loading_message"] = "Asking for appointment confirmation..."
    else:
        next_state["status"] = "running"
        next_state["loading_message"] = "Asking for appointment details..."

    if appointment_info.stop_proccessing:
        next_state["restart_graph"] = True
        next_state["status"] = "running"
        next_state["loading_message"] = (
            "Stopping processing the appointment details..."
        )
    else:
        next_state["restart_graph"] = False

    return next_state


def should_continue_to_confirm_appointment(state: HospitalSystemState):
    """Return the next node to execute"""

    name = state.get("patient_name", None)
    email = state.get("patient_email", None)
    reason = state.get("patient_reason", None)
    restart_graph = state.get("restart_graph", False)

    if restart_graph:
        return START

    if not name or not email or not reason:
        return "ask_appointment_info"
    else:
        # Otherwise end
        return "ask_appointment_confirmation"


confirm_appointment_prompt = """
You are an AI assistant helping to finalize appointment bookings. When asking for confirmation, present the details clearly and request the user to confirm or modify the information.
doctor_name: {doctor_name}
appointment_date: {appointment_date}
start_time: {start_time}
end_time: {end_time}
full_name: {full_name}
email: {email}
reason: {reason}

Use the following guidelines:
1. Format the response in **markdown**.
2. Clearly display all details of the appointment for review.
3. Provide clear options for the user to confirm or make changes. """


def ask_appointment_confirmation(state: HospitalSystemState):
    doctor_name = state.get("doctor").full_name
    appointment_date = state.get("appointment_date")
    start_time = state.get("start_time")
    end_time = state.get("end_time")
    full_name = state.get("patient_name")
    email = state.get("patient_email")
    reason = state.get("patient_reason")

    formatted_system_prompt = confirm_appointment_prompt.format(
        doctor_name=doctor_name,
        appointment_date=appointment_date,
        start_time=start_time,
        end_time=end_time,
        full_name=full_name,
        email=email,
        reason=reason,
    )

    messages = [SystemMessage(content=formatted_system_prompt)]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "response_type": "message",
        "status": "stopped",
        "loading_message": "",
    }


get_appointment_confirmation_system_prompt = """
You are an AI assistant tasked with determining if the user has confirmed their booking or want to make changes to appointment details. Extract the confirmation status and/or the appointment details from their response as follows:

- If the user confirms, set `"confirmed": true`. UNLESS EXPLICITLY MENTIONED, DO NOT ASSUME THE CONFIRMATION.
- If the user does not confirm, set `"confirmed": false`.
- In case the user decides to modify the booking, extract all the necessary details for the appointment and ask for confirmation again. The details are  doctor_name, appointment_date, start_time, end_time, full_name, email, reason.
- If the user show any indication of stopping or cancelling or requesting for information, set the `stop_procceing` true or false.
 """


def get_appointment_confirmation(state: HospitalSystemState):
    query = state.get("query", "")
    appointment_date = state.get("appointment_date")
    start_time = state.get("start_time")
    end_time = state.get("end_time")
    full_name = state.get("patient_name")
    email = state.get("patient_email")
    reason = state.get("patient_reason")
    doctor_name = state.get("doctor_name")

    structured_llm = llm.with_structured_output(ConfirmBooking)

    messages = [
        SystemMessage(content=get_appointment_confirmation_system_prompt)
    ] + [HumanMessage(content=query)]

    confirmation = cast(ConfirmBooking, structured_llm.invoke(messages))

    next_state: NextHospitalSystemState = {
        "messages": [HumanMessage(content=query)],
    }

    if confirmation.confirmed:
        next_state["confirmed_booking"] = True
        next_state["next_node"] = "book_appointment"
        next_state["status"] = "running"
        next_state["loading_message"] = "Booking appointment..."
    else:
        next_state["confirmed_booking"] = False

        if confirmation.full_name and full_name != confirmation.full_name:
            next_state["patient_name"] = confirmation.full_name
            next_state["next_node"] = "ask_appointment_confirmation"
            next_state["status"] = "running"
            next_state["loading_message"] = (
                "Asking for appointment confirmation..."
            )

        if confirmation.email and email != confirmation.email:
            next_state["patient_email"] = confirmation.email
            next_state["next_node"] = "ask_appointment_confirmation"
            next_state["status"] = "running"
            next_state["loading_message"] = (
                "Asking for appointment confirmation..."
            )

        if confirmation.reason and reason != confirmation.reason:
            next_state["patient_reason"] = confirmation.reason
            next_state["next_node"] = "ask_appointment_confirmation"
            next_state["status"] = "running"
            next_state["loading_message"] = (
                "Asking for appointment confirmation..."
            )

        if (
            confirmation.appointment_date
            and appointment_date != confirmation.appointment_date
        ):
            next_state["appointment_date"] = confirmation.appointment_date
            next_state["next_node"] = "check_doctor_availability"
            next_state["status"] = "running"
            next_state["loading_message"] = (
                "Checking for doctor's availability..."
            )

        if confirmation.start_time and start_time != confirmation.start_time:
            next_state["start_time"] = confirmation.start_time
            next_state["next_node"] = "check_doctor_availability"
            next_state["status"] = "running"
            next_state["loading_message"] = (
                "Checking for doctor's availability..."
            )

        if confirmation.end_time and end_time != confirmation.end_time:
            next_state["end_time"] = confirmation.end_time
            next_state["next_node"] = "check_doctor_availability"
            next_state["status"] = "running"
            next_state["loading_message"] = (
                "Checking for doctor's availability..."
            )

        if confirmation.doctor_name and doctor_name != confirmation.doctor_name:
            next_state["doctor_name"] = confirmation.doctor_name
            next_state["next_node"] = "find_doctor"
            next_state["status"] = "running"
            next_state["loading_message"] = "Finding doctor..."

    if confirmation.stop_proccessing:
        next_state["restart_graph"] = True
        next_state["status"] = "running"
        next_state["loading_message"] = "Stopping the process..."
    else:
        next_state["restart_graph"] = False

    return next_state


def should_continue_to_book_confirm_appointment(state: HospitalSystemState):
    """Return the next node to execute"""

    confirmed_booking = state.get("confirmed_booking", False)
    next_state = state.get("next_node", None)
    restart_graph = state.get("restart_graph", False)

    if restart_graph:
        return START

    if not confirmed_booking:
        if next_state:
            return next_state
        return "ask_appointment_confirmation"

    # Otherwise end
    return "book_appointment"


book_appointment_system_prompt = """
You are an AI assistant tasked with formatting the response from the booking API and presenting it to the user for confirmation of success in a clear and professional manner.
Response:
{response}

Follow these guidelines:
1. Format the response in **markdown** for clarity.
2. Display the key details of the booking response, ensuring they are easy to read.
3. Ask the user if they needed help with anything else or if they have any questions."""


def book_appointment_with_info(state: HospitalSystemState):
    appointment_date = state.get("appointment_date")
    start_time = state.get("start_time")
    end_time = state.get("end_time")
    full_name = state.get("patient_name")
    email = state.get("patient_email")
    reason = state.get("patient_reason")
    doctor_id = state.get("doctor_id")

    response = book_appointment(
        doctor_id,
        full_name,
        email,
        appointment_date,
        start_time,
        end_time,
        reason,
    )
    formatted_system_prompt = book_appointment_system_prompt.format(
        response=json.dumps(response["data"], indent=2)
    )

    messages = [SystemMessage(content=formatted_system_prompt)]
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "response_type": "message",
        "status": "stopped",
        "loading_message": "",
    }
