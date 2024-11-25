from typing import TypedDict
import json
import os
from langchain.schema import Document
from uuid import uuid4
from typing import List
from rapidfuzz import fuzz


START_DELIMITOR = "["
END_DELIMITOR = "]"

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_about_info_documents() -> List[Document]:
    file_path = os.path.join(project_dir, "data", "about.json")
    documents: List[Document] = []
    with open(file_path, "r") as file:
        data = json.load(file)

        for key, value in data.items():
            str_data = (
                f"{START_DELIMITOR}{key.upper()}{END_DELIMITOR}\n{value}\n\n"
            )
            id = str(uuid4())
            metadata = {"section": "about", "tag": key.lower(), "id": id}
            documents.append(
                Document(page_content=str_data, metadata=metadata, id=id)
            )

    return documents


def get_contact_info_documents() -> List[Document]:
    file_path = os.path.join(project_dir, "data", "contact.json")
    documents: List[Document] = []
    with open(file_path, "r") as file:
        data = json.load(file)

        for key, value in data.items():
            str_data = (
                f"{START_DELIMITOR}{key.upper()}{END_DELIMITOR}:\n{value}\n\n"
            )
            id = str(uuid4())
            metadata = {"section": "contact", "tag": key.lower(), "id": id}
            documents.append(
                Document(page_content=str_data, metadata=metadata, id=id)
            )

    return documents


def get_doctor_services(name: str, services: List[dict]):
    d_services = []
    for service in services:
        for doctor in service["doctors"]:
            if fuzz.ratio(name.lower(), doctor["name"].lower()) > 90:
                d_services.append(service["title"].lower())

    return d_services


def get_doctors_info_documents() -> List[Document]:
    file_path = os.path.join(project_dir, "data", "doctors.json")
    services_file_path = os.path.join(project_dir, "data", "services.json")
    with open(services_file_path, "r") as file:
        services = json.load(file)

    documents: List[Document] = []
    with open(file_path, "r") as file:
        data = json.load(file)

        for doctor in data:
            str_data = f"{START_DELIMITOR}{doctor['name']}{END_DELIMITOR}\n"
            for key, value in doctor.items():
                if key != "name":
                    str_data += f"{key.upper()}: {value}\n"
            str_data += "\n"
            id = str(uuid4())
            d_services = get_doctor_services(doctor["name"], services)
            metadata = {
                "section": "doctors",
                "tag": doctor["name"].lower(),
                "id": id,
                "services": d_services,
            }
            documents.append(
                Document(page_content=str_data, metadata=metadata, id=id)
            )

    return documents


Doctor = TypedDict("Doctor", {"name": str, "title": str})


class ServiceDict(TypedDict):
    title: str
    context: str
    doctors: List[Doctor]


def get_services_info_documents() -> List[Document]:
    file_path = os.path.join(project_dir, "data", "services.json")
    documents: List[Document] = []
    with open(file_path, "r") as file:
        data: List[ServiceDict] = json.load(file)

        for service in data:
            str_data = (
                f"{START_DELIMITOR}{service['title'].upper()}{END_DELIMITOR}\n"
            )
            str_data += f"DESCRIPTION: {service['context']}\n"
            str_data += "DOCTORS: "

            for doctor in service["doctors"]:
                d_name = doctor["name"]
                d_title = doctor["title"]
                str_data += f"{d_name}-{d_title},\t"

            str_data += "\n"

            id = str(uuid4())
            metadata = {
                "section": "services",
                "tag": service["title"].lower(),
                "id": id,
            }
            documents.append(
                Document(page_content=str_data, metadata=metadata, id=id)
            )

    return documents


def get_useful_links_info_documents() -> List[Document]:
    file_path = os.path.join(project_dir, "data", "usefulLinks.json")
    documents: List[Document] = []
    with open(file_path, "r") as file:
        data = json.load(file)

        for link_data in data:
            str_data = f"{START_DELIMITOR}{link_data['title'].upper()}{END_DELIMITOR}\n"
            str_data += f"{link_data['context']}\n"
            # str_data += f"LINK: {link_data['link']}\n"

            str_data += "\n"

            id = str(uuid4())
            metadata = {
                "section": "about",
                "tag": link_data["title"].lower(),
                "id": id,
            }
            documents.append(
                Document(page_content=str_data, metadata=metadata, id=id)
            )

    return documents


def get_doctor_speciality_documents() -> List[Document]:
    file_path = os.path.join(project_dir, "data", "doctor_speciality.json")
    documents: List[Document] = []
    with open(file_path, "r") as file:
        data = json.load(file)

        for speciality in data:
            str_data = f"{START_DELIMITOR}{speciality['type'].upper()}{END_DELIMITOR}\n"
            for key, value in speciality.items():
                if key != "type":
                    str_data += f"{key.upper()}: {value}\n"
            str_data += "\n"

            id = str(uuid4())
            metadata = {
                "section": "doctor_speciality",
                "tag": speciality["type"].lower(),
                "id": id,
            }
            documents.append(
                Document(page_content=str_data, metadata=metadata, id=id)
            )

    return documents


def get_all_data_documents() -> List[Document]:
    documents = get_about_info_documents()
    documents.extend(get_contact_info_documents())
    documents.extend(get_doctors_info_documents())
    documents.extend(get_services_info_documents())
    documents.extend(get_useful_links_info_documents())
    documents.extend(get_doctor_speciality_documents())

    return documents
