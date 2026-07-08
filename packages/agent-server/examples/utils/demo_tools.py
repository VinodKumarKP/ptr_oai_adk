# Mock data for tools
import datetime
from collections import defaultdict
from typing import Dict, List, Optional

from langchain.tools import tool

RESERVATIONS = defaultdict(lambda: {"flight_info": {}, "hotel_info": {}})
TOMORROW = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
FLIGHTS = [
    # BOS -> JFK
    {"departure_airport": "BOS", "arrival_airport": "JFK", "airline": "Jet Blue", "date": TOMORROW, "id": "1", "price": 300},
    {"departure_airport": "BOS", "arrival_airport": "JFK", "airline": "Delta", "date": TOMORROW, "id": "2", "price": 320},
    {"departure_airport": "BOS", "arrival_airport": "JFK", "airline": "American", "date": TOMORROW, "id": "3", "price": 310},
    {"departure_airport": "BOS", "arrival_airport": "JFK", "airline": "United", "date": TOMORROW, "id": "4", "price": 305},
    {"departure_airport": "BOS", "arrival_airport": "JFK", "airline": "Spirit", "date": TOMORROW, "id": "5", "price": 290},

    # LAX -> SFO
    {"departure_airport": "LAX", "arrival_airport": "SFO", "airline": "Delta", "date": TOMORROW, "id": "6", "price": 150},
    {"departure_airport": "LAX", "arrival_airport": "SFO", "airline": "United", "date": TOMORROW, "id": "7", "price": 160},
    {"departure_airport": "LAX", "arrival_airport": "SFO", "airline": "Southwest", "date": TOMORROW, "id": "8", "price": 140},
    {"departure_airport": "LAX", "arrival_airport": "SFO", "airline": "Alaska", "date": TOMORROW, "id": "9", "price": 155},
    {"departure_airport": "LAX", "arrival_airport": "SFO", "airline": "Jet Blue", "date": TOMORROW, "id": "10", "price": 165},

    # LHR -> CDG
    {"departure_airport": "LHR", "arrival_airport": "CDG", "airline": "British Airways", "date": TOMORROW, "id": "11", "price": 200},
    {"departure_airport": "LHR", "arrival_airport": "CDG", "airline": "Air France", "date": TOMORROW, "id": "12", "price": 210},
    {"departure_airport": "LHR", "arrival_airport": "CDG", "airline": "EasyJet", "date": TOMORROW, "id": "13", "price": 180},
    {"departure_airport": "LHR", "arrival_airport": "CDG", "airline": "Ryanair", "date": TOMORROW, "id": "14", "price": 175},
    {"departure_airport": "LHR", "arrival_airport": "CDG", "airline": "Lufthansa", "date": TOMORROW, "id": "15", "price": 220},
]

HOTELS = [
    # New York
    {"location": "New York", "name": "McKittrick Hotel", "neighborhood": "Chelsea", "id": "1", "price_per_night": 250, "image_url": "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=500"},
    {"location": "New York", "name": "The Plaza", "neighborhood": "Midtown", "id": "2", "price_per_night": 800, "image_url": "https://images.unsplash.com/photo-1551882547-ff40eb0d1e73?w=500"},
    {"location": "New York", "name": "Empire Hotel", "neighborhood": "Upper West Side", "id": "3", "price_per_night": 300, "image_url": "https://images.unsplash.com/photo-1542314831-c6a4d45c30c2?w=500"},
    {"location": "New York", "name": "Ace Hotel", "neighborhood": "NoMad", "id": "4", "price_per_night": 280, "image_url": "https://images.unsplash.com/photo-1496417263034-38ec4f0b665a?w=500"},
    {"location": "New York", "name": "The Standard", "neighborhood": "High Line", "id": "5", "price_per_night": 350, "image_url": "https://images.unsplash.com/photo-1522798514-97ceb8c4f1c8?w=500"},

    # Chicago
    {"location": "Chicago", "name": "The Palmer House", "neighborhood": "Loop", "id": "6", "price_per_night": 180, "image_url": "https://images.unsplash.com/photo-1611892440504-42a792e24d32?w=500"},
    {"location": "Chicago", "name": "The Drake", "neighborhood": "Gold Coast", "id": "7", "price_per_night": 220, "image_url": "https://images.unsplash.com/photo-1518733057094-95b53143d2a7?w=500"},
    {"location": "Chicago", "name": "The Langham", "neighborhood": "River North", "id": "8", "price_per_night": 400, "image_url": "https://images.unsplash.com/photo-1535827841776-24afc1e255ac?w=500"},
    {"location": "Chicago", "name": "LondonHouse", "neighborhood": "Loop", "id": "9", "price_per_night": 350, "image_url": "https://images.unsplash.com/photo-1582719508461-905c673771fd?w=500"},
    {"location": "Chicago", "name": "Viceroy", "neighborhood": "Gold Coast", "id": "10", "price_per_night": 450, "image_url": "https://images.unsplash.com/photo-1596386461350-326ccb383e9f?w=500"},

    # Paris
    {"location": "Paris", "name": "Hotel Ritz", "neighborhood": "Place Vendome", "id": "11", "price_per_night": 1200, "image_url": "https://images.unsplash.com/photo-1505691938895-1758d7def51a?w=500"},
    {"location": "Paris", "name": "Le Meurice", "neighborhood": "Tuileries", "id": "12", "price_per_night": 1100, "image_url": "https://images.unsplash.com/photo-1554647286-f365d7defc2d?w=500"},
    {"location": "Paris", "name": "Shangri-La", "neighborhood": "Iena", "id": "13", "price_per_night": 1000, "image_url": "https://images.unsplash.com/photo-1590490359683-658d3d23f972?w=500"},
    {"location": "Paris", "name": "The Peninsula", "neighborhood": "Kleber", "id": "14", "price_per_night": 950, "image_url": "https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=500"},
    {"location": "Paris", "name": "Mandarin Oriental", "neighborhood": "Opera", "id": "15", "price_per_night": 1050, "image_url": "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=500"},
]

# Flight tools

def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    date: Optional[str] = None,
) -> List[Dict]:
    """Search flights.

    Args:
        departure_airport: 3-letter airport code for the departure airport.
        arrival_airport: 3-letter airport code for the arrival airport.
        date: YYYY-MM-DD date
    """
    results = []
    for flight in FLIGHTS:
        if departure_airport and flight["departure_airport"] != departure_airport:
            continue
        if arrival_airport and flight["arrival_airport"] != arrival_airport:
            continue
        if date and flight["date"] != date:
            continue
        results.append(flight)
        
    # If no specific filters, return all (for demo purposes)
    if not results and not departure_airport and not arrival_airport:
        return FLIGHTS
        
    return results

@tool
def book_flight(
    flight_id: str,
    user_id: str = "default_user"
) -> str:
    """Book a flight.
    Args:
        flight_id: The flight ID
        user_id: The user ID
    """
    matches = [flight for flight in FLIGHTS if flight["id"] == flight_id]
    if not matches:
        return f"Error: Flight ID {flight_id} not found."
        
    flight = matches[0]
    RESERVATIONS[user_id]["flight_info"] = flight
    return f"Successfully booked flight {flight_id} ({flight['airline']} from {flight['departure_airport']} to {flight['arrival_airport']})"


# Hotel tools
@tool
def search_hotels(location: str) -> list[dict]:
    """Search hotels.

    Args:
        location: offical, legal city name (proper noun)
    """
    results = [h for h in HOTELS if h["location"].lower() == location.lower()]
    # Fallback for demo if exact match fails but we have data
    if not results and location.lower() in ["nyc", "ny", "new york city"]:
        results = [h for h in HOTELS if h["location"] == "New York"]
        
    return results

@tool
def book_hotel(
    hotel_id: str,
    user_id: str = "default_user"
) -> str:
    """Book a hotel.
    Args:
        hotel_id: The hotel ID
        user_id: The user ID
    """
    matches = [hotel for hotel in HOTELS if hotel["id"] == hotel_id]
    if not matches:
        return f"Error: Hotel ID {hotel_id} not found."
        
    hotel = matches[0]
    RESERVATIONS[user_id]["hotel_info"] = hotel
    return f"Successfully booked hotel {hotel_id} ({hotel['name']} in {hotel['location']})"

@tool
def read_file(file_path):
    """
    Read file and return its content
    Args:
        file_path: Path to the file to read
    """
    with open(file_path, 'r') as f:
        return f.read()

# Itinerary Tools

@tool
def get_weather(location: str) -> dict:
    """Get the weather for a given city.
    
    Args:
        location: City name
    """
    # Mock data for weather
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "Sunny"
    }

@tool
def search_attractions(location: str) -> list[dict]:
    """Search local attractions for a given city.
    
    Args:
        location: City name
    """
    # Mock data for attractions
    return [
        {
            "name": f"Famous Park in {location}",
            "description": "A beautiful and iconic green space perfect for a relaxing afternoon.",
            "image_url": "https://images.unsplash.com/photo-1542314831-c6a4d45c30c2?w=500"
        },
        {
            "name": f"Historic Museum of {location}",
            "description": "Explore the rich history and culture of the region with stunning exhibits.",
            "image_url": "https://images.unsplash.com/photo-1518733057094-95b53143d2a7?w=500"
        }
    ]

@tool
def search_restaurants(location: str) -> list[dict]:
    """Search local restaurants for a given city.
    
    Args:
        location: City name
    """
    # Mock data for restaurants
    return [
        {
            "name": f"The Local Grill ({location})",
            "cuisine": "American / Local",
            "price_range": "$$",
            "image_url": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=500"
        },
        {
            "name": f"Cafe {location}",
            "cuisine": "Cafe / Bakery",
            "price_range": "$",
            "image_url": "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=500"
        }
    ]