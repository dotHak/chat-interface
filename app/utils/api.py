import requests
import os
from typing import Union

MOCK_HOSPITAL_SYSTEM_BASE_URL = (
    os.getenv("MOCK_HOSPITAL_SYSTEM_BASE_URL")
    or "https://mock-hospital-system.onrender.com"
)


def search_doctor_by_name(name: str) -> dict:
    """Search for a doctor by name in the mock hospital system.

    Args:
        name (str): The name of the doctor to search for.
    """

    if not name or name == "":
        return {"error": "Doctor name cannot be empty"}

    url = f"{MOCK_HOSPITAL_SYSTEM_BASE_URL}/api/doctors/search"
    query_string = {"name": name}
    try:
        res = requests.get(url, params=query_string)
        return res.json()
    except Exception as e:
        return {"error": str(e)}


def check_doctor_availabity(
    doctor_name: str, start_date: str, end_date: str
) -> dict:
    """Check the availability of a doctor in the mock hospital system.

    Args:
        doctor_name (str): The name of the doctor to check availability for.
        start_date (str): The start date of the availability check.
        end_date (str): The end date of the availability check.
    """

    if not doctor_name or doctor_name == "":
        return {"error": "Doctor name cannot be empty"}

    if not start_date:
        return {"error": "Start date cannot be empty"}

    if not end_date:
        return {"error": "End date cannot be empty"}

    doctor = search_doctor_by_name(doctor_name)

    if "error" in doctor:
        return doctor
    elif not doctor or len(doctor) == 0:
        return {"error": "Doctor not found"}

    doctor_id = doctor[0]["id"]

    url = f"{MOCK_HOSPITAL_SYSTEM_BASE_URL}/api/availability/{doctor_id}"
    query_string = {"startDate": start_date, "endDate": end_date}
    try:
        res = requests.get(url, params=query_string)
        return {
            "doctor_id": doctor[0]["id"],
            "availability": res.json(),
        }
    except Exception as e:
        return {"error": str(e)}


def book_appointment(
    doctor_id: int,
    patient_name: str,
    email: str,
    appointment_date: str,
    start_time: str,
    end_time: str,
    reason: Union[str, None] = None,
) -> dict:
    """Book an appointment with a doctor in the mock hospital system.

    Args:
        doctor_id (int): The ID of the doctor to book an appointment with.
        patient_name (str): The name of the patient booking the appointment.
        email (str): The email of the patient booking the appointment.
        appointment_date (str): The date of the appointment.
        start_time (str): The start time of the appointment.
        end_time (str): The end time of the appointment.
        status (str): The status of the appointment.
    """

    if not doctor_id:
        return {"error": "Doctor ID cannot be empty"}

    if not patient_name:
        return {"error": "Patient name cannot be empty"}

    if not email:
        return {"error": "Email cannot be empty"}

    if not appointment_date:
        return {"error": "Appointment date cannot be empty"}

    if not start_time:
        return {"error": "Start time cannot be empty"}

    if not end_time:
        return {"error": "End time cannot be empty"}

    url = f"{MOCK_HOSPITAL_SYSTEM_BASE_URL}/api/appointments"
    payload = {
        "doctorId": int(doctor_id),
        "patientName": patient_name,
        "email": email,
        "appointmentDate": appointment_date,
        "startTime": start_time,
        "endTime": end_time,
        "status": "booked",
        "reason": reason,
    }
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(url, json=payload, headers=headers)
        return {"data": res.json()}
    except Exception as e:
        return {"error": str(e)}
