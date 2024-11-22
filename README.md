# README: U.S. DOT ROADII -- Traffic and Mobility Counts Prediction for Transportation Planning
1. Project Description
2. Prerequisites
3. Usage
	* Building
	* Testing
	* Execution
4. Additional Notes
5. Version History and Retention
6. License
7. Contributing to the Code
8. Contact Information
9. Acknowledgements
10. README Version History

(UPDATE OUTLINE ONCE README IS FINALIZED @REMI)
# 1. Project Description

### ROADII Background

Research, Operational, and Artificial Intelligence Data Integration Initiative (ROADII) is a multi-year initiative led by the United States Department of Transportation (U.S. DOT) Intelligent Transportation Systems Joint Program Office (ITS JPO).

ROADII’s vision is to expand U.S. transportation agencies’ (regional, state, local, tribal, etc.) access to advanced data analytics knowledge and resources including Artificial Intelligence (AI) and Machine Learning (ML). The ROADII team:
- Identifies and evaluates **use cases** that can benefit from advanced data analytics, AI, and ML
- Develops **proofs-of-concept** for use cases
- **Engages stakeholders** with proofs-of-concept and refine based on stakeholder feedback
- **Makes advanced data analytics, AI, and ML tools** available to the public at a central location (e.g., ITS CodeHub) 

The processes and tools developed under ROADII will enable data scientists, researchers, and data providers to test and share new transportation-related AI algorithms; to develop high-value and well-documented AI training datasets; to reduce the barriers of applying AI approaches to transportation data; and to train future transportation researchers.

For more information, visit ITS JPO's website [here](https://www.its.dot.gov/).

### ROADII Use Case 29 - Mobility Counts Prediction System (MCPS)

- **Full Title:** “High-Resolution Mobility Traffic Count Estimation for Modeling, Planning, and Environmental Impact Applications” 
- **Purpose and goals of the project:** The Mobility Traffic Counts code base geographically matches **traffic counting station data** with **probe-collected speed data** on the U.S. National Highway System (NHS), to produce training datasets for roadway traffic volume prediction across the entire road system. The code provides a Graphical User Interface (GUI) to easily load input data, select input and target columns, and train a model using basic AI neural network methods.

Figure 1 shows traffic speed data on NHS roadway links. The speed data originate from U.S. DOT National Performance Management Research Dataset (NPMRDS) managed by the Regional Integrated Transportation Information System (RITIS).

<img src="resources/01_NHSSpeed.png" width="600">

*Figure 1. NHS Roadway Links with Speed Data*


Figure 2 shows the locations of over 8,000 U.S. Federal Highway Administration (FHWA), Travel Monitoring Analysis System (TMAS) stations for traffic counting and classification.

<img src="resources/02_TMASStations.png" width="600">

*Figure 2. TMAS Traffic Counting Stations*


Figure 3 shows the locations of NHS roadway links having TMAS traffic counting stations.

<img src="resources/03_NHS_TMAS.png" width="600">

*Figure 3. NHS Roadway Links with TMAS Traffic Counting Stations*


Figure 4 shows U.S. Census 2020 Population Density by County as an example of the Census data used in the code base. The code base uses NHS roadway links where traffic counts and speed data are available, along with Census data; to perform prediction for NHS roadway links having similar Census data characteristics where traffic counts and/or speed data are not available.

<img src="resources/04_USCensus2020PopDensityExample.png" width="600">

*Figure 4. U.S. Census 2020 Population Density by County (Retrieved from https://maps.geo.census.gov/ddmv/map.html)*


- **Purpose of the source code and how it relates to the overall goals of the project:** This tool will enable more accurate calculation and prediction of on-road traffic volumes for planning and highway project analysis purposes. Traditional methods of calculating traffic counts on roadways where there are no continuous counting stations require in-person measurements or inaccurate averaging methods. Future traffic volume forecasting may also be difficult or inaccurate due to reliance on traffic count measurements from many years ago or incomplete data.

    The MCPS tool attempts to ease the burden on planning and transportation agencies by providing a simple neural network model to output historical traffic count estimates on roadways where there are no continuous count data available, as is the case for most National Highway System network links. The intended user base includes state and local agencies looking to produce and use more complete traffic speed and traffic volume datasets. Applications of these resulting datasets and the code in this repository include highway planning projects and highway management projects, as well as future traffic forecasting efforts.

- **Length of the project:** The code base is currently in a stable state. The ROADII Team may push updates irregularly as new data become available that might be used to update the model

# 2. Prerequisites

Requires:
- Installation of Python 3.6.0 or later
- Installation of Python packages listed in *requirements.txt*
- Command prompt application to run the source code from the current working directory

# 3. Usage

### Starting up the Streamlit Interface

The steps to run the streamlit interface

1) Clone the GitHub envionment to a location of your choosing on your machine
2) In a command prompt application (e.g., Anaconda Prompt), execute the line **--> pip install -r requirements.txt** to ensure all necessary Python packages are up to date and installed in your working environment
3) Run *main.py* to produce the Streamlit GUI. This can be done by executing the following line in a command prompt application **-->python -m streamlit run main.py** (ensuring that the dependencides in requirements.txt have been installed in a location accessible by this version of Python)
4) The Streamlit Interface should now open in a browser window (See additional instructions for using the interface below)

### Using the Streamlit Interface (A lot to add in here @Remi)

The following section details the function of each tab within the Streamlit Interface

- "0 - Introduction": Provides the user with background context on required datasets and terminology. Details the workflow of the Streamlit Interface. 
- "1 - Generate Dataset":
    * Generate a dataset to predict traffic volumes on roads with no existing traffic counting stations (TMAS).
    * Generate a dataset to predict traffic volumes on roads with existing traffic counting stations (TMAS). This is used for testing the performance of the model or training a new model.
- "2 - Use a Traffic Counts Model": Allows the user to select a pre-trained AI Model (included with GitHub distribution) or one they've trained using the "Train Model" tab, anf apply that model to a generated dataset to obtain traffic volume predictions.
    * If a generated dataset with TMAS data was used, there will be perforance metrics generated in the "Results" tab comparing the predicted values to the measured values from the TMAS data.
- "3 - Results": View performance metrics comparing actual and predicted traffic volumes or view predictions for roads with no measured traffic volumes. Explore an interactive map of station locations from generated dataset.
- "4 - Train Model": Train a new model using your generated dataset with the ability to select custom input features and target variable.
- "5 - About": User views links for helpful information related to this source code e.g., Points of Contacts, GitHub link, and README download link

### Source code included in the Mobility Counts Prediction System (MCPS)

The [ml](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/tree/main/ml) folder contains the modules and classes to read in the requisite training data for building a MCPS model, using a pre-trained model, and generating a dataset to test model performance. The source code can be used if the user desires to interact directly with the model architecture, train custom versions of the MCPS model or adapt the functionality to another use case. Direct use of files other than the *main.py* file is not required to use the tool.

The following modules are contained therein:

- **<main.py>:** Produces a Streamlit GUI that reads the training data files, normalizes all columns to numerical types, and runs a training loop on the normalized data to produce a neural network to predict the user-chosen target column
	* The user’s default web browser (e.g., Google Chrome, Microsoft Edge) opens a Streamlit application with the address "localhost:8501"
- **<use_model.py>:** Uses a cached model file (.pt format) to produce traffic count estimates. Also provides an easier, script based methodology to train a new model version without using the Streamlit application or interface. This is useful for more rapid model iteration
- **<setup_funcs.py>:** Sets up the various data sources for training a model
- **<module_data.py>:** Reads, formats, and joins the various data sources into a single training dataset. This includes the Traffic Monitoring and Analysis System traffic volume data and the National Performance Measurement Research Data Set speed data
- **<module_census.py>:** Connects the training data to census information to improve model performance
- **<module_ai.py>:** Defines the AI/ML training loop, the model architecture, and saves the resulting model for later use. Also provides methods to use a saved or cached model file 

## Detailed Usage Instructions

### Predicting Traffic Counts

Figure 5 shows the Streamlit GUI’s main banner at the top, with the "Generate Dataset" tab activated. This tab provides the user some instructions for downloading NPMRDS speed data from the source website. The user can then choose the correct files to upload to the GUI and join the files together.

<img src="resources/05_StreamlitSourceDataFile.png" width="600">

*Figure 5. Streamlit GUI – User Chooses Source Data File*


After the user chooses a source data file, the user should navigate to the "Predict Traffic Counts" tab to predict traffic counts. The interface will guide the user through picking the correct combined data file that the user generated in Tab 1. Pressing "Run model and predict traffic counts" will run the model on the input dataset and produce a '.csv" file in the selected output folder, as well as a ".pkl" file that the user can use to load data into Tab 3, or into another python environment for faster follow-on analysis. 

<img src="resources/06_StreamlitUserViewsInputData.png" width="600">

*Figure 6. Streamlit GUI – Predicting Traffic Counts*

The "results" tab provides some information about the resulting traffic count predictions that were produced in Tab 2, with some simple visualizations and filters for users to drill down into the result of the model. The results are 

<img src="resources/06-5_StreamlitUserViewsInputData.png" width="600">

*Figure 7. Streamlit GUI – Traffic Count Prediction Results*

### Training a custom version of the model

Figure 8 shows the user’s ability to choose input data columns and the target data column in AI model training – in the “Train Model” tab. The input data columns should not include the target column. After the user chooses input data columns and the target data column and clicks “Train Model” – then AI model training is initiated and the user will start to see in-progress results in their command prompt application. After AI model training is complete, the code base saves an AI model file to the sub-directory “..\models.”

<img src="resources/07_StreamlitUserTrainAIModel.png" width="600">

*Figure 8. Streamlit GUI – User Selects Input Data for AI Model Training on a Targeted Metric*


In the “Train Model” tab, if the user chooses to train a new AI model, Figure 9 shows the AI model training progress with losses updating with each epoch on the Streamlit GUI. The x-axis is the number of AI training epochs; the user may set the number of training epochs in the source code, and the AI model training process ends once the number of epochs is reached. The y-axis is the logarithmic loss of the AI model training.

<img src="resources/09_StreamlitTrainAIModel_loss.png" width="450">

*Figure 10. Streamlit GUI – AI Model Training Losses vs. Epoch*


In addition to the training loss plot in the Streamlit GUI, Figure 10 shows the AI model training process with a periodically updating graph in the command prompt application. In Figure 10, the x-axis is the percent difference (absolute value) between AI Model Training (i.e., Predicted Value) and Input Data (i.e., Expected Value), and the y-axis is the number of occurrences in a percent difference histogram bin. The bin size in Figure 9 is two (2) percent.

<img src="resources/10_StreamlitTrainAIModel_pctdiff.png" width="600">

*Figure 11. AI Model Training – Histogram of Percent Difference (Absolute Value) between AI Model Training (i.e., Predicted Value) and Input Data (i.e., Expected Value)*

Figure 11 compares the predicted data versus input data and visually depicts the statistical correlation or R^2 value. The closer the scatterplot looks to a line with slope of 1, the closer the R^2 value is to 1. The x-axis is the expected value while the y-axis is the predicted value.

<img src="resources/11_StreamlitTrainAIModel_accuracy.png" width="600">

*Figure 12. AI Model Training – AI Model Training (i.e., Predicted Value) versus Input Data (i.e., Expected Value)*


Figure 12 shows real-time updating relative importance of the input training features (i.e., columns of the input dataframe).

<img src="resources/12_StreamlitTrainAIModel_importance.png" width="600">

*Figure 13. AI Model Training – Relative Importance of Input Training Features*

After the AI model training completes, the following outputs are seen on the command prompt application. The logarithmic loss, tensor, test loss, and R-squared values provide a high-level summary of the AI model training. The AI model is saved to “..\models.”

-------------------

Epoch [2500/2500],

Logarithmic Loss: 104156.6484375,

tensor([[2079.9739],
            [ 271.9318],
            [4203.4741],
            ...,
            [ 647.0178],
            [3022.9729],
            [ 242.9163]])
	    
tensor([[2846.],
            [ 394.],
            [5372.],
            ...,
            [2676.],
            [2528.],
            [ 103.]])
	    
Test Loss: 382541.6875,

R2: 0.8152651190757751,

36.359431140945375% are within 15.0 Percent of Expected,

Model weights saved to ../models/model__20240329_194737

Model file saved to ../models/model__20240329_194737

-------------------

The "Results" tab will include additional comparison plots, metrics, and visualizations after training a custom version of the model. The user may choose a date and time range to display the input data and the predicted data using that input dataset. Figure 13 shows the datepicker widget in the Streamlit GUI.

<img src="resources/13_StreamlitResults_datepicker.png" width="450">

*Figure 13. Results Tab - Date Picker to Filter Results (courtesy https://github.com/imdreamer2018/streamlit-date-picker)*

After the user has chosen a date to show results, Figure 14 shows the display of input data on a Folium map in the Streamlit GUI whose timestamp falls within the selected date. Each road segment is indicated by a Folium icon and line segment. The icon and line segment are colored with respect to traffic volume, a darker shade of blue (input data) indicates that the traffic volume at that road segment is in a higher quintile within the entire set of traffic-volume-per-road-segment data. Figure 15, Figure 16, and Figure 17 are example close-up views of an urban, suburban, and rural road segment, respectively. Note that in some of the following figures – the icon color and design may not reflect the latest release of the code base.

<img src="resources/14_StreamlitResults_inputdata.png" width="600">

*Figure 14. Results Tab - Display of Input Data*

<img src="resources/15_StreamlitResults_inputdata_urban.png" width="450">

*Figure 15. Results Tab - Display of Input Data (Urban Road Segment)*

<img src="resources/16_StreamlitResults_inputdata_suburban.png" width="450">

*Figure 16. Results Tab - Display of Input Data (Suburban Road Segment)*

<img src="resources/17_StreamlitResults_inputdata_rural.png" width="450">

*Figure 17. Results Tab - Display of Input Data (Rural Road Segment)*

<img src="resources/18_StreamlitResults_compareinputtopredicted.png" width="450">

*Figure 18. Results Tab - Compare Input Data and Predicted Data for a Traffic Station*

# 4. Additional Notes

The geographic region that the algorithms use to train the model is determined by the NPMRDS data input into the code. Additional updates and improvements are planned in future releases and iterations.

**Known Issues:** None identified, this use case is still in development and future updates will be tested sufficiently before being released. 

**Associated datasets:** This use case incorporates NPMRDS, TMAS, U.S. Census, and other data sources to train the model discussed herein.


# 5. Version History and Retention

**Status:** This project is in active development phase. 

**Release Frequency:** This project will be updated when there are stable developments. This will be approximately every month. 

**Retention:** This project will likely remain publicly accessible indefinitely. 


# 6. License

This project is licensed under the Creative Commons 1.0 Universal (CC0 1.0) License - see the [License.md](https://github.com/usdot-jpo-codehub/codehub-readme-template/blob/master/LICENSE) file for more details. 


# 7. Contributing to the Code

Please read [Contributing.md](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/blob/main/Contributing.MD) for details on our Code of Conduct, the process for submitting pull requests to us, and how contributions will be released.


# 8. Contact Information

Contact Name: Billy Chupp

Contact Information: William.Chupp@dot.gov

Contact Name: Eric Englin

Contact Information: Eric.Englin@dot.gov


### Citing this code

Users may cite our code base and/or associated publications. Below is a sample citation for the code base:

> ROADII Team. (2024). _ROADII README Template_ (0.1) [Source code]. Provided by ITS JPO through GitHub.com. Accessed yyyy-mm-dd from https://doi.org/xxx.xxx/xxxx.

When you copy or adapt from this code, please include the original URL you copied the source code from and date of retrieval as a comment in your code. Additional information on how to cite can be found in the [ITS CodeHub FAQ](https://its.dot.gov/code/#/faqs).


# 9. Acknowledgements

- Billy Chupp (Volpe), William.Chupp@dot.gov
- Eric Englin (Volpe), Eric.Englin@dot.gov
- RJ Rittmuller (Volpe), Robert.Rittmuller@dot.gov
- Michael Barzach (Volpe), Michael.Barzach@dot.gov
- Jason Lu (Volpe), Jason.Lu@dot.gov

This project is funded by the U.S. DOT, ITS JPO under IAA HWE3A122. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the ITS JPO.

### Languages

-Python [100.0%](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/search?l=python)

### About

This repository provides code for using ML methods to join national traffic datasets. One of these traffic data sets measure speed, and the other measures traffic volumes.


# 10. README Version History

*Table 1. README Version History*

<img src="resources/ReadmeVersionHistory.PNG" width="450">


