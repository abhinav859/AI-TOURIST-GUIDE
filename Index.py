"""
Kerala Hidden Gems Recommender - Enterprise MVP
-----------------------------------------------
This application serves as a comprehensive travel recommendation engine,
utilizing Machine Learning (TF-IDF, Cosine Similarity) and external APIs 
(Google Gemini, OpenWeatherMap, OpenRouteService) to deliver context-aware, 
conversational travel itineraries for the state of Kerala.

Features:
- External CSV Data Loading & Validation
- Conversational RAG Chatbot
- Dynamic AI Itinerary Generation
- Real-Time Weather Contextualization
- Isochrone Drive-Time Polygon Mapping
- Image Carousels & Visual UI Integration
"""

# --- Standard Library Imports ---
import os
from typing import Tuple, Optional, List, Any, Dict

# --- Third-Party Library Imports ---
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import google.generativeai as genai

# --- Machine Learning Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. API Configuration & Global Variables
# ==========================================

# Replace these strings with your actual API keys in a production environment.
# For Streamlit deployment, it is highly recommended to use st.secrets instead.
GEMINI_API_KEY: str = "AIzaSyBu7JUdRXrzCxkZKZhs85CTuuK9yyOhYlk"
WEATHER_API_KEY: str = "553a12892f87093cb1bb478cc9a67799"
ORS_API_KEY: str = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImJiNWI1MDYxNzI2ZDQ0ZGI5ODRmYjI2Zjg0M2Y0NjgxIiwiaCI6Im11cm11cjY0In0="

# Initialize the Gemini Large Language Model
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    # Configure the API client with the provided key
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Instantiate the specific generative model we wish to use
    llm_model: Optional[genai.GenerativeModel] = genai.GenerativeModel('gemini-2.5-flash')
else:
    # Set to None to gracefully degrade functionality if no key is provided
    llm_model = None


# ==========================================
# 2. UI Setup & Page Configuration
# ==========================================

st.set_page_config(
    page_title="Kerala AI Travel Guide", 
    page_icon="🌴", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 3. Data Ingestion & Validation
# ==========================================

@st.cache_data
def load_data(filepath: str = "kerala_tourism.csv") -> pd.DataFrame:
    """
    Loads the tourism dataset strictly from an external CSV file.
    Includes strict schema validation to ensure all required columns are present.
    
    Args:
        filepath (str): The relative or absolute path to the CSV file.
        
    Returns:
        pd.DataFrame: A validated Pandas DataFrame containing the location data.
        
    Stops Execution:
        If the file does not exist or if required columns are missing,
        the Streamlit app will halt and display an error message.
    """
    
    # 3.1 Check if the file exists on the disk
    if not os.path.exists(filepath):
        st.error(f"❌ **Database Error:** The external file `{filepath}` was not found.")
        st.info("Please place your CSV file in the same directory as this script.")
        st.markdown(
            "**Required Columns:** `name`, `category`, `tags`, "
            "`latitude`, `longitude`, `rating`, `review_count`, `image_url`"
        )
        st.stop()
        
    try:
        # 3.2 Attempt to parse the CSV file into memory
        loaded_dataframe: pd.DataFrame = pd.read_csv(filepath)
        
        # 3.3 Define the strict schema required for the application to function
        required_columns: List[str] = [
            'name', 
            'category', 
            'tags', 
            'latitude', 
            'longitude', 
            'rating', 
            'review_count', 
            'image_url'
        ]
        
        # 3.4 Validate that all required columns are present in the loaded dataframe
        missing_cols: List[str] = [
            col for col in required_columns if col not in loaded_dataframe.columns
        ]
        
        # 3.5 Halt execution if validation fails
        if len(missing_cols) > 0:
            st.error(
                f"❌ **Format Error:** Your CSV is missing the following "
                f"required columns: {missing_cols}"
            )
            st.stop()
            
        return loaded_dataframe
        
    except Exception as read_error:
        # Catch any pandas parsing errors (e.g., corrupted file, wrong encoding)
        st.error(f"❌ **Error reading the CSV file:** {read_error}")
        st.stop()

# Load the external data into the global scope
app_dataframe: pd.DataFrame = load_data()


# ==========================================
# 4. External API Integration Functions
# ==========================================

def get_live_weather(lat: float, lon: float) -> Tuple[str, float]:
    """
    Fetches real-time weather data from the OpenWeatherMap API to 
    contextualize AI recommendations.
    
    Args:
        lat (float): The latitude of the target location.
        lon (float): The longitude of the target location.
        
    Returns:
        Tuple[str, float]: A tuple containing the main weather condition 
                           (e.g., "Clear", "Rain") and the temperature in Celsius.
    """
    # 4.1 Check for missing API key and return mock data if necessary
    if WEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
        return "Sunny", 28.0 
        
    # 4.2 Construct the API endpoint URL with metric units for Celsius
    api_endpoint_url: str = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    )
    
    try:
        # 4.3 Execute the HTTP GET request
        response: requests.Response = requests.get(api_endpoint_url, timeout=3)
        response_json: Dict[str, Any] = response.json()
        
        # 4.4 Parse the desired metrics from the JSON payload
        weather_condition: str = response_json['weather'][0]['main']
        current_temperature: float = response_json['main']['temp']
        
        return weather_condition, current_temperature
        
    except Exception:
        # Fallback values if the API fails, times out, or returns invalid JSON
        return "Unknown", 28.0


def get_isochrone_polygon(lat: float, lon: float, drive_time_mins: int = 60) -> Optional[List[List[float]]]:
    """
    Fetches a drive-time boundary polygon (isochrone) from the OpenRouteService API.
    This boundary represents how far a vehicle can travel within the specified time.
    
    Args:
        lat (float): The latitude of the starting location.
        lon (float): The longitude of the starting location.
        drive_time_mins (int): The maximum allowed driving time in minutes.
        
    Returns:
        Optional[List[List[float]]]: A list of [longitude, latitude] coordinate pairs 
                                     forming the polygon, or None if the request fails.
    """
    # 4.5 Check for missing API key and gracefully degrade to None
    if ORS_API_KEY == "YOUR_OPENROUTESERVICE_API_KEY":
        return None 
        
    # 4.6 Configure the OpenRouteService API request payload
    api_endpoint_url: str = "https://api.openrouteservice.org/v2/isochrones/driving-car"
    
    request_headers: Dict[str, str] = {
        'Authorization': ORS_API_KEY, 
        'Content-Type': 'application/json'
    }
    
    # ORS expects coordinates in [Longitude, Latitude] format
    # Range must be converted from minutes to seconds
    request_body: Dict[str, Any] = {
        "locations": [[lon, lat]], 
        "range": [drive_time_mins * 60]
    }
    
    try:
        # 4.7 Execute the HTTP POST request
        response: requests.Response = requests.post(
            url=api_endpoint_url, 
            json=request_body, 
            headers=request_headers, 
            timeout=5
        )
        response_json: Dict[str, Any] = response.json()
        
        # 4.8 Traverse the GeoJSON response to extract the polygon geometry
        polygon_coordinates: List[List[float]] = response_json['features'][0]['geometry']['coordinates'][0]
        
        return polygon_coordinates
        
    except Exception:
        # Return None so the mapping logic can fallback to a standard circular radius
        return None


def generate_ai_itinerary(start_loc: str, destination: str, tags: str) -> str:
    """
    Leverages the Google Gemini Large Language Model to dynamically author
    a customized, context-aware travel itinerary.
    
    Args:
        start_loc (str): The name of the user's current location.
        destination (str): The name of the recommended hidden gem.
        tags (str): The comma-separated descriptive tags for the destination.
        
    Returns:
        str: The Markdown-formatted itinerary generated by the AI.
    """
    # 4.9 Validate that the model is instantiated
    if llm_model is None:
        warning_message: str = (
            "⚠️ **API Key Required:** Please enter your Gemini API key in the code "
            "to generate live, dynamic AI itineraries."
        )
        return warning_message
        
    # 4.10 Construct the prompt instruction set for the LLM
    generation_prompt: str = (
        f"Act as an expert Kerala tour guide. "
        f"I am driving from {start_loc} to {destination}. "
        f"The destination is known for these characteristics: {tags}. "
        f"Write a short, bulleted 1-day itinerary including local food recommendations. "
        f"Keep the total length under 150 words."
    )
    
    try:
        # 4.11 Request content generation from Gemini
        ai_response = llm_model.generate_content(generation_prompt)
        generated_text: str = ai_response.text
        
        return generated_text
        
    except Exception as generation_error:
        # Catch quota limits, network errors, or API rejections
        error_message: str = f"❌ AI generation failed: {generation_error}"
        return error_message


# ==========================================
# 5. Core Application UI & Layout
# ==========================================

st.title("🌴 Kerala AI Travel Guide & Copilot")
st.markdown("Navigate the hidden beauty of Kerala with the power of LLMs and Geospatial AI.")

# Create a two-column layout for the primary interface
# The left column is for the chatbot, the right is for controls and maps
left_chat_column, right_map_column = st.columns([1, 2])

# ------------------------------------------
# 5.1 Chatbot Interface (Left Column)
# ------------------------------------------

with left_chat_column:
    st.subheader("💬 AI Travel Copilot")
    st.info("Chat with the AI to find your perfect spot.")
    
    # Initialize the chat history within the Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! Where are you currently located in Kerala, and what kind of vibe are you looking for today?"
            }
        ]
        
    # Render all previous chat messages to the UI
    for chat_message in st.session_state.messages:
        role: str = chat_message["role"]
        content: str = chat_message["content"]
        st.chat_message(role).write(content)

    # Capture the user's latest input query
    user_query: Optional[str] = st.chat_input("E.g., I'm in Munnar and want a quiet waterfall nearby...")
    
    if user_query is not None and user_query.strip() != "":
        
        # Append and display the user's message
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        # Extremely basic NLP extraction for location matching
        # In a fully-fledged RAG app, an LLM would parse this intent
        detected_location: Optional[str] = None
        unique_locations: np.ndarray = app_dataframe['name'].unique()
        
        for loc_name in unique_locations:
            if loc_name.lower() in user_query.lower():
                detected_location = loc_name
                break
                
        # Determine the appropriate chatbot response based on extraction
        if detected_location is not None:
            copilot_reply: str = (
                f"I see you're near {detected_location}. "
                f"Let me scan the map for hidden gems matching your vibe..."
            )
            # Update the global app state with the detected location
            st.session_state.current_location = detected_location
        else:
            copilot_reply: str = (
                "I couldn't detect a starting location in your message. "
                "Could you select one from the map settings on the right?"
            )
            
        # Append and display the assistant's reply
        st.session_state.messages.append({"role": "assistant", "content": copilot_reply})
        st.chat_message("assistant").write(copilot_reply)


# ------------------------------------------
# 5.2 Control Panel & Map (Right Column)
# ------------------------------------------

with right_map_column:
    
    # Sub-columns for the control inputs
    control_col_1, control_col_2 = st.columns(2)
    
    with control_col_1:
        # Determine the default index based on chat state or default to the first row
        default_location_name: str = st.session_state.get('current_location', app_dataframe['name'].iloc[0])
        default_index_position: int = list(app_dataframe['name']).index(default_location_name)
        
        target_place_name: str = st.selectbox(
            label="Current Location:", 
            options=app_dataframe['name'].unique(), 
            index=default_index_position
        )
        
    with control_col_2:
        max_drive_time_mins: int = st.slider(
            label="Max Drive Time (Minutes)", 
            min_value=30, 
            max_value=180, 
            value=60,
            step=10
        )

    # ==========================================
    # 6. Recommendation Logic Processing
    # ==========================================
    
    # Locate the target coordinates in the dataframe
    target_row_indices: List[int] = app_dataframe.index[app_dataframe['name'] == target_place_name].tolist()
    primary_target_idx: int = target_row_indices[0]
    
    target_latitude: float = app_dataframe.loc[primary_target_idx, 'latitude']
    target_longitude: float = app_dataframe.loc[primary_target_idx, 'longitude']

    with st.spinner("Fetching Live Weather & Mapping Drive Times..."):
        
        # 6.1 Weather Context Integration
        weather_condition, temperature = get_live_weather(lat=target_latitude, lon=target_longitude)
        
        st.write(f"🌤️ **Live Weather at {target_place_name}:** {weather_condition}, {temperature}°C")

        # 6.2 TF-IDF and Cosine Similarity Calculation
        # Initialize the vectorizer to remove common English stop words
        tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')
        
        # Fill missing tags with an empty string to prevent fitting errors
        filled_tags_series: pd.Series = app_dataframe['tags'].fillna(value="")
        
        # Fit the vectorizer and transform the text into a sparse matrix
        tfidf_feature_matrix = tfidf_vectorizer.fit_transform(raw_documents=filled_tags_series)
        
        # Extract the vector representation of the chosen start location
        target_location_vector = tfidf_feature_matrix[primary_target_idx]
        
        # Calculate cosine similarity across all rows
        similarity_matrix = cosine_similarity(X=target_location_vector, Y=tfidf_feature_matrix)
        flattened_similarities: np.ndarray = similarity_matrix.flatten()
        
        # Assign scores back to the dataframe
        app_dataframe['similarity_score'] = flattened_similarities

        # 6.3 Dynamic Weather Filtering
        # Create a pool of candidates by removing the target location itself
        candidate_pool: pd.DataFrame = app_dataframe[app_dataframe['name'] != target_place_name].copy()
        
        # If it is raining, apply penalties to outdoor water-based activities
        if "Rain" in weather_condition:
            
            # Identify rows categorized as Beach or Waterfall
            is_water_activity = candidate_pool['category'].isin(['Beach', 'Waterfall'])
            
            # Reduce their similarity score drastically
            candidate_pool.loc[is_water_activity, 'similarity_score'] *= 0.2
            
            st.warning("☔ It's raining! The AI is prioritizing indoor or safe hill-station activities.")

        # 6.4 Final Composite Scoring
        # The score is a weighted blend: 60% Vibe Match (Similarity), 40% Obscurity (Review Count)
        maximum_reviews_in_dataset = candidate_pool['review_count'].max()
        
        # Calculate components
        weighted_similarity: pd.Series = candidate_pool['similarity_score'] * 0.6
        normalized_obscurity: pd.Series = (1.0 - (candidate_pool['review_count'] / maximum_reviews_in_dataset)) * 0.4
        
        # Sum components to create the final gem score
        candidate_pool['gem_score'] = weighted_similarity + normalized_obscurity
        
        # Sort values descending and slice the top 3 recommendations
        sorted_candidates: pd.DataFrame = candidate_pool.sort_values(by='gem_score', ascending=False)
        top_recommendations: pd.DataFrame = sorted_candidates.head(3)

        # ==========================================
        # 7. Map Rendering & Isochrone Processing
        # ==========================================
        
        st.subheader("🗺️ Isochrone Drive-Time Map")
        
        # Initialize the Folium interactive map
        interactive_map: folium.Map = folium.Map(
            location=[target_latitude, target_longitude], 
            zoom_start=10
        )
        
        # Attempt to retrieve the actual drive-time boundary from ORS
        isochrone_polygon_coordinates: Optional[List[List[float]]] = get_isochrone_polygon(
            lat=target_latitude, 
            lon=target_longitude, 
            drive_time_mins=max_drive_time_mins
        )
        
        if isochrone_polygon_coordinates is not None:
            # Reformat coordinates from ORS [lon, lat] to Folium [lat, lon]
            folium_formatted_coords: List[List[float]] = [
                [coord[1], coord[0]] for coord in isochrone_polygon_coordinates
            ]
            
            # Draw the irregular polygon onto the map
            folium.Polygon(
                locations=folium_formatted_coords, 
                color='blue', 
                fill=True, 
                fill_opacity=0.2, 
                popup=f"{max_drive_time_mins}-Min Drive Zone"
            ).add_to(interactive_map)
            
        else:
            # Fallback to a perfect mathematical circle if the API call failed
            # Approximation: 600 meters per minute of driving time
            approximate_radius_meters: float = max_drive_time_mins * 600.0
            
            folium.Circle(
                location=[target_latitude, target_longitude], 
                radius=approximate_radius_meters, 
                color='blue', 
                fill=True, 
                popup="Estimated Drive Zone"
            ).add_to(interactive_map)

        # Plot the starting location marker in red
        start_icon = folium.Icon(color="red", icon="home")
        folium.Marker(
            location=[target_latitude, target_longitude], 
            icon=start_icon,
            popup="Start Location"
        ).add_to(interactive_map)
        
        # Plot the recommended hidden gem markers in green
        for index_val, gem_row in top_recommendations.iterrows():
            
            gem_lat: float = gem_row['latitude']
            gem_lon: float = gem_row['longitude']
            gem_name: str = gem_row['name']
            
            gem_icon = folium.Icon(color="green", icon="star")
            
            folium.Marker(
                location=[gem_lat, gem_lon], 
                popup=gem_name, 
                icon=gem_icon
            ).add_to(interactive_map)
        
        # Render the map within the Streamlit layout container
        st_folium(fig=interactive_map, width=700, height=400)

        # ==========================================
        # 8. Visual Results & AI Itinerary Generation
        # ==========================================
        
        st.subheader("Top Hidden Gems")
        
        # Iterate through the top recommendations to display their details
        for index_val, gem_row in top_recommendations.iterrows():
            
            gem_name: str = gem_row['name']
            gem_category: str = gem_row['category']
            gem_image_url: Any = gem_row.get('image_url')
            
            # Create an expanding accordion menu for each location
            with st.expander(label=f"💎 {gem_name} - {gem_category}"):
                
                is_valid_image = pd.notna(gem_image_url) and str(gem_image_url).strip() != ""
                if is_valid_image:
                    # Split the accordion into image and text columns
                    image_column, details_column = st.columns([1, 2])
                    with image_column:
                        st.image(image=gem_image_url, use_container_width=True)
                else:
                    details_column = st.container()
                        
                with details_column:
                    # Display metadata statistics
                    st.write(f"⭐ {gem_row['rating']} | 📝 {gem_row['review_count']} Reviews")
                    st.write(f"**Vibe:** {gem_row['tags']}")
                    
                    # Create a unique key for the button to prevent Streamlit rendering errors
                    button_key_identifier: str = f"btn_{gem_name}"
                    button_label: str = f"✨ Generate AI Road Trip to {gem_name}"
                    
                    # Manage state for this specific gem
                    state_key = f"itinerary_{gem_name}"
                    
                    # Render the button and execute generation if clicked
                    if st.button(label=button_label, key=button_key_identifier):
                        
                        with st.spinner("Gemini AI is writing your itinerary..."):
                            
                            # Call the Generative AI function
                            generated_itinerary: str = generate_ai_itinerary(
                                start_loc=target_place_name, 
                                destination=gem_name, 
                                tags=gem_row['tags']
                            )
                            
                            # Save to session state
                            st.session_state[state_key] = generated_itinerary
                            
                    # If an itinerary exists in session state, show it
                    if state_key in st.session_state:
                        itinerary_text = st.session_state[state_key]
                        
                        # Display the output to the user
                        st.success(itinerary_text)
