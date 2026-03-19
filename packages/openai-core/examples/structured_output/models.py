from pydantic import BaseModel, Field


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
    departure_airport: str = Field(description="The 3-letter IATA code for the booked departure airport (e.g., 'BOS').")
    arrival_airport: str = Field(description="The 3-letter IATA code for the booked arrival airport (e.g., 'JFK').")
    airline: str = Field(description="The name of the airline operating the booked flight (e.g., 'Jet Blue').")
    date: str = Field(description="The booked date of the flight in YYYY-MM-DD format.")
    flight_id: str = Field(description="The unique identifier for the booked flight.")
    price: str = Field(description="The total price of the flight booking in USD.")
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
    price_per_night: str = Field(description="The price per night for the hotel room in USD.")
    user_id: str = Field(description="The user ID associated with the booking.")


class CompleteItinerary(BaseModel):
    hotel_confirmation: HotelConfirmation = Field(
        description="The confirmation details for a successful booked hotel.")
    flight_confirmation: FlightConfirmation = Field(
        description="The confirmation details for a successful booked flight.")