import requests
import os
import pandas as pd

class census_data(object):

    def __init__(self) -> None:

        # Get the API key from environment variable
        API_KEY = open('C:/Users/wchupp/Documents/CENSUS_API_KEY.txt').read()
        if(API_KEY != "" or API_KEY is not None):
            self.API_KEY = API_KEY
            print("Census API key found!")
        else:
            print("No census API key found!")

    # Helper functions --------------------------------------------------------------------------------
    def read_csv_to_dataframe(self, filepath):
        # Reads the provided CSV file and returns a pandas DataFrame.
        return pd.read_csv(filepath)
    
    def write_dataframe_to_csv(self, dataframe, filepath):
        # Writes a pandas DataFrame to a CSV file.
        try:
            dataframe.to_csv(filepath, index=False)
            print(f"DataFrame successfully written to {filepath}")
        except Exception as e:
            print(f"Error writing DataFrame to CSV: {str(e)}")

    # Main functions -----------------------------------------------------------------------------------
    def get_population_data_by_county(self, state_code):
        # The endpoint for the specific API you are interested in
        # Here we are assuming you're interested in the 2019 Population Estimates API
        endpoint = f'https://api.census.gov/data/2019/pep/population'

        # Define the parameters for your request
        params = {
            'get': 'NAME,POP',
            'for': f'county:*',
            'in': f'state:{state_code}',
            'key': self.API_KEY
        }

        # Send the request
        response = requests.get(endpoint, params=params)

        # Check for a valid response
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Convert the data to a Pandas DataFrame
            df = pd.DataFrame(data[1:], columns=["Name","Population","State", "County"])
            
            # Convert the "Population" column to integers
            df["Population"] = df["Population"].astype(int)
            
            return df
        else:
            print(f'Failed to retrieve data: {response.status_code}')
            return None

    def get_population_data_by_place(self, state_code):
        # The endpoint for the specific API you are interested in
        # Here we are assuming you're interested in the 2019 Population Estimates API
        endpoint = f'https://api.census.gov/data/2019/pep/population'

        # Define the parameters for your request
        params = {
            'get': 'NAME,POP',
            'for': f'place:*',
            'in': f'state:{state_code}',
            'key': self.API_KEY
        }

        # Send the request
        response = requests.get(endpoint, params=params)

        # Check for a valid response
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Convert the data to a Pandas DataFrame
            df = pd.DataFrame(data[1:], columns=["Name", "Population", "State", "Place"])
            
            # Convert the "Population" column to integers
            df["Population"] = df["Population"].astype(int)
            
            return df
        else:
            print(f'Failed to retrieve data: {response.status_code}')
            return None
    
    def get_population_data_by_city(self, state_code):
        # The endpoint for the specific API you are interested in
        # Here we are assuming you're interested in the 2019 Population Estimates API
        endpoint = f'https://api.census.gov/data/2019/pep/population'

        # Define the parameters for your request
        params = {
            'get': 'NAME,POP',
            'for': f'consolidated city:*',
            'in': f'state:{state_code}',
            'key': self.API_KEY
        }

        # Send the request
        response = requests.get(endpoint, params=params)

        # Check for a valid response
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Convert the data to a Pandas DataFrame
            df = pd.DataFrame(data[1:], columns=["Name", "Population", "State", "Place"])
            
            # Convert the "Population" column to integers
            df["Population"] = df["Population"].astype(int)
            
            return df
        else:
            print(f'Failed to retrieve data: {response.status_code}')
            return None

    def get_pop_data_from_file(self, filename, state, county, year):
        # grabs population estimate data from US census output that has been saved to csv
        pop_df = self.read_csv_to_dataframe(filename)
        towns = pop_df[(pop_df['COUNTY'] == int(county)) & (pop_df['STATE'] == int(state)) & (pop_df['SUMLEV'] != 50)]
        final_output = towns[['NAME', 'POPESTIMATE' + str(year)]]
        return final_output
