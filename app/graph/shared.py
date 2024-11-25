from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage
from utils.gene import Gene
from utils.get_text_data import get_all_data_documents
from typing import List, Literal, Union, Dict, NotRequired, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

llm = ChatOpenAI(model="gpt-4o", temperature=0)
gene = Gene(get_all_data_documents())


class DoctorAvailability(BaseModel):
    doctor_name: Union[str, None] = Field(
        None, description="The name of the doctor to check the availability."
    )
    start_date: Union[str, None] = Field(
        None, description="The start date of the availability check."
    )
    end_date: Union[str, None] = Field(
        None, description="The end date of the availability check."
    )
    stop_processing: bool = Field(
        False, description="Stop processing the request."
    )


class Doctor(BaseModel):
    full_name: Union[str, None] = Field(
        None, description="The full name of the doctor."
    )
    title: Union[str, None] = Field(
        None, description="The title of the doctor."
    )

    def to_dict(self) -> dict:
        return {"name": self.full_name, "title": self.title}


class Availability(BaseModel):
    response_type: Literal["availability-list", "no-availability"] = Field(
        description="The type of response."
    )
    response: str = Field(description="The response message.")
    response_before: str = Field(
        description="The response message before the availability list."
    )
    response_after: str = Field(
        description="The response message after the availability list."
    )


class AppointmentDate(BaseModel):
    date: Union[str, None] = Field(
        None, description="The date of the appointment."
    )
    start_time: Union[str, None] = Field(
        None, description="The start time of the appointment."
    )
    end_time: Union[str, None] = Field(
        None, description="The end time of the appointment."
    )
    stop_processing: bool = Field(
        False, description="Stop processing the request."
    )


class AppointmentInfo(BaseModel):
    full_name: Union[str, None] = Field(
        None, description="The full name of the patient."
    )
    email: Union[str, None] = Field(
        None, description="The email of the patient."
    )
    reason: Union[str, None] = Field(
        None, description="The reason for the appointment."
    )
    stop_proccessing: bool = Field(
        False, description="Stop processing the request."
    )


class ConfirmBooking(BaseModel):
    confirmed: bool = Field(
        False, description="The confirmation status of the booking."
    )
    appointment_date: Union[str, None] = Field(
        None, description="The date of the appointment."
    )
    start_time: Union[str, None] = Field(
        None, description="The start time of the appointment."
    )
    end_time: Union[str, None] = Field(
        None, description="The end time of the appointment."
    )
    doctor_name: Union[str, None] = Field(
        None, description="The name of the doctor."
    )
    full_name: Union[str, None] = Field(
        None, description="The full name of the patient."
    )
    email: Union[str, None] = Field(
        None, description="The email of the patient."
    )
    reason: Union[str, None] = Field(
        None, description="The reason for the appointment."
    )
    stop_proccessing: bool = Field(
        False, description="Stop processing the request."
    )


class PatientIntent(BaseModel):
    intent: Literal["booking-appointment", "hospital-info", "general-info"] = (
        Field(description="The detected intent of the patient's request.")
    )


class HospitalSystem(BaseModel):
    appointment_date: Union[str, None] = Field(
        None, description="The date of the appointment."
    )
    appointment_start_time: Union[str, None] = Field(
        None, description="The start time of the appointment."
    )
    appointment_end_time: Union[str, None] = Field(
        None, description="The end time of the appointment."
    )
    appointment_reason: Union[str, None] = Field(
        None, description="The reason for the appointment."
    )
    symptoms_description: Union[str, None] = Field(
        None, description="The description of the symptoms."
    )
    doctor_name: Union[str, None] = Field(
        None, description="The name of the doctor."
    )
    specialists: List[str] = Field(
        [], description="The list of specialists listed in the query."
    )


class PotentialDoctors(BaseModel):
    doctors: List[Doctor] = Field(
        [], description="The list of potential doctors."
    )
    prompt_before: str = Field(
        "", description="The prompt message before the list of doctors."
    )
    prompt_after: str = Field(
        "", description="The prompt message after the list of doctors."
    )


class HospitalSystemState(MessagesState):
    query: str
    intent: str
    start_time: str
    end_time: str
    start_date: str
    end_date: str
    doctor_name: str
    specialists: List[str]
    symptoms_description: str
    reason: str
    doctors_list: List[Doctor]
    doctor: Doctor
    doctor_not_found: bool
    doctor_id: int
    availability: List[Dict[str, str]]
    response_type: str
    response_before: str
    response_after: str
    appointment_date: str
    patient_name: str
    patient_email: str
    patient_reason: str
    confirmed_booking: bool
    next_node: str
    search_results: str
    loading_message: str
    status: Literal["stopped", "completed", "running"]
    restart_graph: bool


class NextHospitalSystemState(TypedDict):
    messages: NotRequired[List[AnyMessage]]
    query: NotRequired[str]
    intent: NotRequired[str]
    start_time: NotRequired[str]
    end_time: NotRequired[str]
    start_date: NotRequired[str]
    end_date: NotRequired[str]
    doctor_name: NotRequired[str]
    specialists: NotRequired[List[str]]
    symptoms_description: NotRequired[str]
    reason: NotRequired[str]
    doctors_list: NotRequired[List[Doctor]]
    doctor: NotRequired[Union[Doctor, None]]
    doctor_not_found: NotRequired[bool]
    doctor_id: NotRequired[int]
    availability: NotRequired[List[Dict[str, str]]]
    response_type: NotRequired[str]
    response_before: NotRequired[str]
    response_after: NotRequired[str]
    appointment_date: NotRequired[str]
    patient_name: NotRequired[str]
    patient_email: NotRequired[str]
    patient_reason: NotRequired[str]
    confirmed_booking: NotRequired[bool]
    next_node: NotRequired[str]
    search_results: NotRequired[str]
    loading_message: NotRequired[str]
    status: NotRequired[Literal["stopped", "completed", "running"]]
    restart_graph: NotRequired[bool]
