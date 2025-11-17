#To store chat logs( quesry and response) in a database

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

Base= declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_query= Column(Text, nullable=False)
    bot_response= Column(Text, nullable=False)
    timestamp=Column(DateTime(timezone=True), server_default=func.now())
    