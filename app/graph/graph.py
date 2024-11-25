from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from graph.shared import HospitalSystemState

from graph.prelimary import (
    detect_patient_intent,
    extract_preliminary_info,
    find_potential_doctors,
    should_continue_to_next_branch,
    should_continue_to_find_potential_doctors,
)
from graph.general_info import general_info_response
from graph.booking_appointment import (
    availability_chat_agent,
    ask_availability_details,
    find_doctor,
    check_doctors_availability,
    get_appointment_date_time,
    ask_appointment_info,
    get_appointment_info,
    get_appointment_confirmation,
    ask_appointment_confirmation,
    book_appointment_with_info,
    should_continue_to_check_availability,
    should_continue_to_ask_appointment_info,
    should_continue_to_confirm_appointment,
    should_continue_to_get_appointment_info,
    should_continue_to_book_confirm_appointment,
    should_continue_to_get_appointment_date_and_time,
    should_continue_to_find_doctor,
)

from graph.hospital_info import retrieve_hospital_info, hospital_chat_agent


hospital_memory = MemorySaver()


def build_hospital_system_graph():
    hospital_builder = StateGraph(HospitalSystemState)

    # Preliminary info
    hospital_builder.add_node("detect_patient_intent", detect_patient_intent)
    hospital_builder.add_node(
        "preliminary_info_extraction", extract_preliminary_info
    )
    hospital_builder.add_node("find_potential_doctors", find_potential_doctors)
    hospital_builder.add_node("general_info_response", general_info_response)

    hospital_builder.add_edge(START, "detect_patient_intent")
    hospital_builder.add_edge(
        "find_potential_doctors", "availability_chat_agent"
    )
    hospital_builder.add_edge("general_info_response", END)

    hospital_builder.add_conditional_edges(
        "detect_patient_intent",
        should_continue_to_next_branch,
        [
            "preliminary_info_extraction",
            "detect_patient_intent",
            "retrieve_hospital_info",
            "general_info_response",
        ],
    )
    hospital_builder.add_conditional_edges(
        "preliminary_info_extraction",
        should_continue_to_find_potential_doctors,
        ["find_potential_doctors", "find_doctor", "ask_availability_details"],
    )

    # Hospital Info
    hospital_builder.add_node("hospital_chat_agent", hospital_chat_agent)
    hospital_builder.add_node("retrieve_hospital_info", retrieve_hospital_info)

    hospital_builder.add_edge("retrieve_hospital_info", "hospital_chat_agent")
    hospital_builder.add_edge("hospital_chat_agent", END)

    # Booking appointment
    hospital_builder.add_node(
        "availability_chat_agent", availability_chat_agent
    )
    hospital_builder.add_node(
        "ask_availability_details", ask_availability_details
    )
    hospital_builder.add_node("find_doctor", find_doctor)
    hospital_builder.add_node(
        "check_doctor_availability", check_doctors_availability
    )
    hospital_builder.add_node(
        "get_appointment_date_time", get_appointment_date_time
    )
    hospital_builder.add_node("ask_appointment_info", ask_appointment_info)
    hospital_builder.add_node("get_appointment_info", get_appointment_info)
    hospital_builder.add_node(
        "ask_appointment_confirmation", ask_appointment_confirmation
    )
    hospital_builder.add_node(
        "get_appointment_confirmation", get_appointment_confirmation
    )
    hospital_builder.add_node("book_appointment", book_appointment_with_info)

    hospital_builder.add_edge(
        "ask_availability_details", "availability_chat_agent"
    )
    # hospital_builder.add_edge("availability_chat_agent", "find_doctor")
    hospital_builder.add_edge(
        "ask_appointment_confirmation", "get_appointment_confirmation"
    )
    hospital_builder.add_edge("book_appointment", END)

    hospital_builder.add_conditional_edges(
        "availability_chat_agent",
        should_continue_to_find_doctor,
        ["find_doctor", "detect_patient_intent"],
    )
    hospital_builder.add_conditional_edges(
        "find_doctor",
        should_continue_to_check_availability,
        ["ask_availability_details", "check_doctor_availability"],
    )
    hospital_builder.add_conditional_edges(
        "check_doctor_availability",
        should_continue_to_get_appointment_date_and_time,
        ["ask_availability_details", "get_appointment_date_time"],
    )
    hospital_builder.add_conditional_edges(
        "ask_appointment_info",
        should_continue_to_get_appointment_info,
        ["get_appointment_info", "ask_appointment_confirmation"],
    )
    hospital_builder.add_conditional_edges(
        "get_appointment_date_time",
        should_continue_to_ask_appointment_info,
        [
            "get_appointment_date_time",
            "ask_appointment_info",
            "detect_patient_intent",
        ],
    )
    hospital_builder.add_conditional_edges(
        "get_appointment_info",
        should_continue_to_confirm_appointment,
        [
            "ask_appointment_info",
            "ask_appointment_confirmation",
            "detect_patient_intent",
        ],
    )
    hospital_builder.add_conditional_edges(
        "get_appointment_confirmation",
        should_continue_to_book_confirm_appointment,
        [
            "ask_appointment_confirmation",
            "book_appointment",
            "find_doctor",
            "check_doctor_availability",
            "detect_patient_intent",
        ],
    )

    hospital_graph = hospital_builder.compile(
        checkpointer=hospital_memory,
        interrupt_before=[
            "availability_chat_agent",
            "get_appointment_date_time",
            "get_appointment_info",
            "get_appointment_confirmation",
        ],
    )

    return hospital_graph


def get_memory_config(id: str) -> RunnableConfig:

    return {"configurable": {"thread_id": id}}


def save_graph_image_to_file(file_path: str):
    graph = build_hospital_system_graph()
    graph.get_graph(xray=1).draw_mermaid_png(output_file_path=file_path)
