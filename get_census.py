# NOTICE: This is now depricated in favor of the full census module!!

import requests
import os

def get_2020_census_population_by_county(county_name):
    API_KEY = os.environ.get('CENSUS_API_KEY')
    if not API_KEY:
        return "API key not set in environment variables."
    
    BASE_URL = "https://api.census.gov/data/2020/nat"
    params = {
        "get": "NAME,POP",
        "for": f"county:*",
        "key": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        raw_response = response.text
        print(raw_response)
        data = response.json()
        print(data)
        for record in data[1:]:  # Skip the header row
            name, population, _, _ = record
            if county_name.lower() in name.lower():
                return f"The 2020 census population for {name} is {population}."
        return f"County named {county_name} not found."
    else:
        return f"Error {response.status_code}: {response.text}"

# Example usage
# Make sure to set the environment variable before running, e.g., in terminal:
# export CENSUS_API_KEY="YOUR_CENSUS_API_KEY"

county_name = "Middlesex"
print(get_2020_census_population_by_county(county_name))