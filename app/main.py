from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from dotenv import load_dotenv
from typing import Any
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

load_dotenv()

from graph.graph import build_hospital_system_graph, get_memory_config
from db.feedback_db import FeedbackRequest, Feedback, get_db

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://api.yourdoc.click",
        "https://chat.yourdoc.click",
        "https://yourdoc.click",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

graph = build_hospital_system_graph()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket, chat_id: str):
        self.active_connections.remove(websocket)

    async def send(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)


manager = ConnectionManager()


async def process_event(
    event: dict[str, Any] | Any,
    manager: ConnectionManager,
    websocket: WebSocket,
    client_id: str,
):
    response_type = event.get("response_type", None)
    status = event.get("status", "")
    loading_message = event.get("loading_message", "")
    # print(status, loading_message)
    # config = get_memory_config(client_id)
    # graph_state = graph.get_state(config)
    # print(graph_state.next)
    if status == "completed" or status == "stopped":
        if response_type == "message":
            # get last message
            message = event["messages"][-1]
            await manager.send(
                {
                    "message": message.content,
                    "type": "chat-message",
                    "role": "system",
                    "status": "success",
                },
                websocket,
            )
        elif response_type == "availability-list":
            availability = event.get("availability", [])
            message_before = event.get("response_before", "")
            message_after = event.get("response_after", "")

            await manager.send(
                {
                    "availability": availability,
                    "type": "availability-list",
                    "role": "system",
                    "status": "success",
                    "messageBefore": message_before,
                    "messageAfter": message_after,
                },
                websocket,
            )

        elif response_type == "potential_doctors":
            doctors_list = event.get("doctors_list", [])
            message_before = event.get("response_before", "")
            message_after = event.get("response_after", "")

            await manager.send(
                {
                    "doctors": [doc.to_dict() for doc in doctors_list],
                    "type": "doctors-list",
                    "role": "system",
                    "status": "success",
                    "messageBefore": message_before,
                    "messageAfter": message_after,
                },
                websocket,
            )

    elif status == "running" and loading_message:
        await manager.send(
            {
                "type": "loading-state",
                "state": "loading",
                "message": loading_message,
            },
            websocket,
        )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/feedback")
async def submit_feedback(
    feedback_data: FeedbackRequest, db: Session = Depends(get_db)
):
    # Create feedback object
    feedback_entry = Feedback(
        feedback=feedback_data.feedback,
        comments=feedback_data.comments,
        message_before=feedback_data.message_before,
        message_after=feedback_data.message_after,
        message_type=feedback_data.message_type,
        message=feedback_data.message,
        user_message=feedback_data.user_message,
    )
    db.add(feedback_entry)
    db.commit()
    db.refresh(feedback_entry)
    return {"status": "success", "data": feedback_entry.id}


# GET endpoint to retrieve all feedback
@app.get("/feedback")
async def get_feedback(db: Session = Depends(get_db)):
    feedback_list = db.query(Feedback).all()
    return {"status": "success", "data": [f.__dict__ for f in feedback_list]}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive message from the client
            user_json = await websocket.receive_json()
            user_message = user_json["message"]
            should_restart = user_json.get("restart", False)

            # Send loading state to the client immediately
            await manager.send(
                {
                    "type": "loading-state",
                    "state": "loading",
                    "message": "Thinking...",
                },
                websocket,
            )

            config = get_memory_config(client_id)
            graph_state = graph.get_state(config)

            if graph_state.next and not should_restart:
                graph.update_state(
                    config, {"query": user_message, "status": "running"}
                )
                async for event in graph.astream(
                    None,
                    config,
                    stream_mode="values",
                ):
                    await process_event(event, manager, websocket, client_id)

            else:
                # Process the LLM invocation asynchronously
                async for event in graph.astream(
                    {"query": user_message, "status": "running"},
                    config,
                    stream_mode="values",
                ):
                    await process_event(event, manager, websocket, client_id)

    except WebSocketDisconnect:
        print("websocket disconnected")
        manager.disconnect(websocket, client_id)
