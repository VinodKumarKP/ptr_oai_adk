from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List
from datetime import datetime


class EmailAnalysis(BaseModel):
    """
    Represents the structured analysis of an email's content.
    This model is used to extract key information and determine the required actions.
    """
    summary: str = Field(
        description="A concise, one-line summary of the email's main topic.",
        min_length=1
    )
    requires_urgent_response: bool = Field(
        description="Set to true if the email requires an immediate or time-sensitive response.",
        default=False
    )
    sentiment: str = Field(
        description="The overall sentiment of the email. Must be one of 'positive', 'negative', or 'neutral'.",
        enum=['positive', 'negative', 'neutral']
    )


class FlightConfirmation(BaseModel):
    """
    Represents the confirmation details for a successfully booked flight.
    This model is used as the structured output after a flight reservation is made.
    """
    departure_airport: str = Field(description="The 3-letter IATA code for the departure airport (e.g., 'BOS').")
    arrival_airport: str = Field(description="The 3-letter IATA code for the arrival airport (e.g., 'JFK').")
    airline: str = Field(description="The name of the airline operating the flight (e.g., 'Jet Blue').")
    date: str = Field(description="The date of the flight in YYYY-MM-DD format.")
    flight_id: str = Field(description="The unique identifier for the booked flight.")
    price: int = Field(description="The total price of the flight booking in USD.")
    user_id: str = Field(description="The user ID associated with the booking.")


class HotelConfirmation(BaseModel):
    """
    Represents the confirmation details for a successfully booked hotel.
    This model is used as the structured output after a hotel reservation is made.
    """
    location: str = Field(description="The city where the hotel is located (e.g., 'New York').")
    name: str = Field(description="The name of the booked hotel (e.g., 'The Plaza').")
    neighborhood: str = Field(description="The neighborhood where the hotel is located (e.g., 'Midtown').")
    hotel_id: str = Field(description="The unique identifier for the booked hotel.")
    price_per_night: int = Field(description="The price per night for the hotel room in USD.")
    user_id: str = Field(description="The user ID associated with the booking.")


class CompleteItinerary(BaseModel):
    hotel_confirmation: HotelConfirmation = Field(
        description="The confirmation details for a successful booked hotel.")
    flight_confirmation: FlightConfirmation = Field(
        description="The confirmation details for a successful booked flight.")


class GenericAgentResponse(BaseModel):
    """
    A generic response model that works with any agent.
    This flexible model can capture responses from different types of agents
    with varying output structures.
    """
    status: str = Field(
        description="The status of the agent's execution. Must be one of 'success', 'partial', 'error', or 'pending'.",
        enum=['success', 'partial', 'error', 'pending'],
        default='success'
    )
    message: str = Field(
        description="A human-readable message summarizing the agent's response or action taken.",
        default=""
    )
    data: Optional[Dict[str, Any]] = Field(
        description="The main response data/payload from the agent. Can contain any structure.",
        default=None
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="Additional metadata about the response such as execution time, token counts, etc.",
        default=None
    )
    errors: Optional[List[str]] = Field(
        description="A list of error messages if any issues occurred during execution.",
        default=None
    )
    timestamp: datetime = Field(
        description="The timestamp when the response was generated.",
        default_factory=datetime.utcnow
    )
    agent_name: Optional[str] = Field(
        description="The name or identifier of the agent that generated this response.",
        default=None
    )
    session_id: Optional[str] = Field(
        description="The session identifier for tracking multi-turn conversations or agent executions.",
        default=None
    )
    confidence: Optional[float] = Field(
        description="A confidence score (0.0 to 1.0) indicating the agent's confidence in the response.",
        default=None,
        ge=0.0,
        le=1.0
    )

    class Config:
        """Pydantic config for the GenericAgentResponse model."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Task completed successfully",
                "data": {
                    "result": "example output",
                    "details": "additional information"
                },
                "metadata": {
                    "execution_time_ms": 1234,
                    "tokens_used": 512
                },
                "errors": None,
                "timestamp": "2026-03-19T10:30:45.123456",
                "agent_name": "example_agent",
                "session_id": "session_123",
                "confidence": 0.95
            }
        }
