from sqlalchemy import create_engine, Integer, String, Text
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
import os


# SQLite Database setup
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./feedback.db"
DATABASE_AUTH_TOKEN = os.getenv("DATABASE_AUTH_TOKEN")
if DATABASE_AUTH_TOKEN:
    dbUrl = (
        f"sqlite+{DATABASE_URL}/?authToken={DATABASE_AUTH_TOKEN}&secure=true"
    )
    engine = create_engine(dbUrl, connect_args={"check_same_thread": False})
else:
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# Database model
class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    feedback: Mapped[str] = mapped_column(String, nullable=False)
    comments: Mapped[str] = mapped_column(Text, nullable=False)
    message_before: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_after: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_type: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return (
            f"Feedback(id={self.id!r}, feedback={self.feedback!r}, "
            f"comments={self.comments!r}, message_before={self.message_before!r}, "
            f"message_after={self.message_after!r}, message_type={self.message_type!r}, "
            f"message={self.message!r}, user_message={self.user_message!r})"
        )


# Create database tables
Base.metadata.create_all(bind=engine)


class FeedbackRequest(BaseModel):
    feedback: str = Field(description="The feedback message.")
    comments: str = Field(description="The comments for the feedback.")
    message_before: str | None = Field(
        None, description="The message before the main message."
    )
    message_after: str | None = Field(
        None, description="The message after the main message."
    )
    message: str = Field(description="The main message.")
    message_type: str = Field(description="The type of the message.")
    user_message: str = Field(description="The user message.")


# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
