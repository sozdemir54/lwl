# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, concat
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error as mae
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="LWL Predictor", page_icon="🚰")

# Custom CSS to style the navigation title
css = """
<style>
    .stApp h1 {
        display: flex;
        align-items: center;
    }

    .stApp h1 img {
        margin-right: 10px;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# Function to display existing data
def dashboard():
    st.markdown("<h1 style='text-align: center; color: black;'>Dashboard</h1>", unsafe_allow_html=True)
    st.write("👇Check last week averages compared to total average.")
    existing_data = pd.read_excel('your_dataset.xlsx')

    # creating KPIs
    avg_precipitation = np.mean(existing_data["precipitation"])
    avg_max_temp = np.mean(existing_data["max_temp"])
    avg_min_temp = np.mean(existing_data["min_temp"])
    avg_avg_temp = np.mean(existing_data["avg_temp"])
    avg_withdrawal = np.mean(existing_data["withdrawal"])
    avg_water_level = np.mean(existing_data["water_level"])

    last_week_avg_water_level = np.mean(existing_data["water_level"].iloc[-5:])
    last_week_avg_max_temp = np.mean(existing_data["max_temp"].iloc[-5:])
    last_week_avg_min_temp = np.mean(existing_data["min_temp"].iloc[-5:])
    last_week_avg_avg_temp = np.mean(existing_data["avg_temp"].iloc[-5:])
    last_week_avg_withdrawal = np.mean(existing_data["withdrawal"].iloc[-5:])
    last_week_avg_precipitation = np.mean(existing_data["precipitation"].iloc[-5:])
        
    # creating a single-element container
    placeholder = st.empty()

    with placeholder.container():

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Lake Water Level 🌉",
            value=round(last_week_avg_water_level, 2),
            delta=round(last_week_avg_water_level-avg_water_level, 2)
        )
        
        kpi2.metric(
            label="Precipitation 🌂",
            value=round(last_week_avg_precipitation, 2),
            delta=round(last_week_avg_precipitation-avg_precipitation, 2)
        )
        
        kpi3.metric(
            label="Maximum Temperature 🌞",
            value=round(last_week_avg_max_temp, 2),
            delta=round(last_week_avg_max_temp-avg_max_temp, 2)
        )

        # create three columns
        kpi4, kpi5, kpi6 = st.columns(3)

        kpi4.metric(
            label="Average Temperature 🌝",
            value=round(last_week_avg_avg_temp, 2),
            delta=round(last_week_avg_avg_temp-avg_avg_temp, 2)
        )
        
        kpi5.metric(
            label="Minimum Temperature 🌖",
            value=round(last_week_avg_min_temp, 2),
            delta=round(last_week_avg_min_temp-avg_min_temp, 2)
        )
        
        kpi6.metric(
            label="Withdrawal 🌊",
            value=round(last_week_avg_withdrawal, 2),
            delta=round(last_week_avg_withdrawal-avg_withdrawal, 2)
        )

        st.write("👇Radar chart for average values of variables.")
        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            df = pd.DataFrame(dict(
                r=[avg_water_level, avg_max_temp, avg_min_temp, avg_avg_temp, avg_precipitation],
                theta=['Water Level','Max Temp','Min Temp', 'Avg Temp','Precipitation']))
            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, height=400)

        with fig_col2:
            df = pd.DataFrame(dict(
                r=[avg_max_temp, avg_min_temp, avg_avg_temp],
                theta=['Max Temp','Min Temp', 'Avg Temp']))
            fig2 = px.line_polar(df, r='r', theta='theta', line_close=True)
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True, height=400)

        st.write("👇Line chart to compare several variables.")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=existing_data["date"],
            y=existing_data["water_level"],
            name = 'Water Level', 
        ))
        #fig2.add_trace(go.Scatter(
        #   x=existing_data["date"],
        #    y=existing_data["precipitation"],
        #    name='Precipitation',
        #))
        fig2.add_trace(go.Scatter(
            x=existing_data["date"],
            y=existing_data["max_temp"],
            name='Max Temperature',
        ))        
        fig2.add_trace(go.Scatter(
            x=existing_data["date"],
            y=existing_data["min_temp"],
            name='Min Temperature',
        ))
        fig2.add_trace(go.Scatter(
            x=existing_data["date"],
            y=existing_data["avg_temp"],
            name='Avg Temperature',
        ))
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=False, height=400)



        st.write("👇Time series with range slider and selectors")
        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(existing_data.date), y=list(existing_data.water_level)))
        # Set title
        fig.update_layout(
            title_text="Water Level"
        )
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                visible=True
                ),
                type="date"
            )
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, height=400)

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(existing_data.date), y=list(existing_data.precipitation)))
        # Set title
        fig.update_layout(
            title_text="Precipitation"
        )
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                visible=True
                ),
                type="date"
            )
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, height=400)

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(existing_data.date), y=list(existing_data.max_temp)))
        # Set title
        fig.update_layout(
            title_text="Maximum Temperature"
        )
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                visible=True
                ),
                type="date"
            )
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, height=400)

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(existing_data.date), y=list(existing_data.avg_temp)))
        # Set title
        fig.update_layout(
            title_text="Average Temperature"
        )
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                visible=True
                ),
                type="date"
            )
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, height=400)

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(existing_data.date), y=list(existing_data.min_temp)))
        # Set title
        fig.update_layout(
            title_text="Minimum Temperature"
        )
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                visible=True
                ),
                type="date"
            )
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, height=400)

        # Create figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(existing_data.date), y=list(existing_data.withdrawal)))
        # Set title
        fig.update_layout(
            title_text="Withdrawal"
        )
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                visible=True
                ),
                type="date"
            )
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, height=400)

        st.write("👇See current dataset.")

        st.markdown("### Detailed Data View")
        st.dataframe(existing_data)

# Function to add data manually
def manual_entry_page():
    st.title("Manual Entry")
    st.write("Use this page to manually add, modify, or delete data in your dataset.")
    # Load the existing Excel sheet into a DataFrame
    existing_data = pd.read_excel('your_dataset.xlsx')

    col1, col2 = st.columns(2)
    with col1: 
        # Optionally, display the updated data
        if st.checkbox('Show Current Data'):
            st.dataframe(existing_data)
    with col2: 
        edit_mode = st.checkbox("Edit Mode") 

    # Collect user input for 'Column1'
    user_input1 = st.number_input("Input for Water Flow", min_value=0.0, value=0.0, max_value=999999.0)

    # Collect user input for 'Column2'
    user_input_column2 = st.number_input("Input for Precipitation:", min_value=0.0, value=0.0, max_value=999999.0)

    # Collect user input for 'Column3'
    user_input_column3 = st.number_input("Input for Maximum Temperature:", min_value=0.0, value=0.0, max_value=999999.0)

    # Collect user input for 'Column4'
    user_input_column4 = st.number_input("Input for Minimum Temperature:", min_value=0.0, value=0.0, max_value=999999.0)

    # Collect user input for 'Column5'
    user_input_column5 = st.number_input("Input for Average Temperature:", min_value=0.0, value=0.0, max_value=999999.0)

    # Collect user input for 'Column6'
    user_input_column6 = st.number_input("Input for Lake Water Level:", min_value=0.0, value=0.0, max_value=999999.0)

    # Collect user input for the custom date
    user_input_date = st.date_input("Input for Date:", value=datetime(2023, 9, 9))

    # Option to edit or delete data
    edit_index = st.number_input("Enter the index of the row you want to edit (leave empty to add new data):", min_value=0, max_value=len(existing_data)-1, step=1)

       
    # Add new data
    if not edit_mode or edit_index is None:
        if st.button('ADD'):
            # Create a new DataFrame with the user-submitted data
            new_data = pd.DataFrame({
                'date': [user_input_date],
                'withdrawal': [user_input1],
                'precipitation': [user_input_column2],
                'max_temp': [user_input_column3],
                'min_temp': [user_input_column4],
                'avg_temp': [user_input_column5],
                'water_level': [user_input_column6]
            })

            # Append the new data to the existing DataFrame
            existing_data = existing_data.append(new_data, ignore_index=True)

            # Save the updated DataFrame back to the Excel file
            existing_data.to_excel('your_dataset.xlsx', index=False)

            # Display a success message
            st.success('Data added successfully!')

    if edit_mode and edit_index is not None:
        # Modify existing data
        if st.button('EDIT'):
            existing_data.at[edit_index, 'date'] = user_input_date
            existing_data.at[edit_index, 'withdrawal'] = user_input1
            existing_data.at[edit_index, 'precipitation'] = user_input_column2
            existing_data.at[edit_index, 'max_temp'] = user_input_column3
            existing_data.at[edit_index, 'min_temp'] = user_input_column4
            existing_data.at[edit_index, 'avg_temp'] = user_input_column5
            existing_data.at[edit_index, 'water_level'] = user_input_column6

            # Save the updated DataFrame back to the Excel file
            existing_data.to_excel('your_dataset.xlsx', index=False)

            # Display a success message
            st.success('Data edited successfully!')

        # Delete existing data
        delete_data = st.checkbox("Delete Data")
        if delete_data:
            if st.button('DELETE'):
                existing_data.drop(index=edit_index, inplace=True)
                existing_data.reset_index(drop=True, inplace=True)

                # Save the updated DataFrame back to the Excel file
                existing_data.to_excel('your_dataset.xlsx', index=False)

                # Display a success message
                st.success('Data deleted successfully!')

# Function to make predictions
def predict_page_30():

    st.title("Predict")
    st.write("Use this page to make predictions using your dataset.")

    # Read data from CSV file
    data = pd.read_excel('your_dataset.xlsx')  # Assuming the CSV file has columns 'precipitation', 'water_level', 'max_temp', 'min_temp', 'avg_temp', 'withdrawal'
 
    # Select the relevant columns for prediction
    data = data[['water_level', 'max_temp', 'min_temp', 'avg_temp', 'precipitation', 'withdrawal']]
    st.write(f"Today's water level value is: {data['water_level'].iloc[-1]:.2f}")
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)


    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if isinstance(data, list) else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    # Prepare the data for supervised learning
    timesteps = 30
    features = data.shape[1]
    reframed = series_to_supervised(scaled_data, timesteps, 30)
    print(reframed.head())

    # Split the data into train and test sets
    values = reframed.values
    train_size = 348
    val_size = 464
    train = values[:train_size, :]
    val = values[train_size:val_size, :]
    test = values[val_size:, :]

    # Split the data into input and output variables
    train_X, train_y = train[:, :-features * 30], train[:, -features:]
    val_X, val_y = val[:, :-features * 30], val[:, -features:]
    test_X, test_y = test[:, :-features * 30], test[:, -features:]

    # Reshape the input data to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timesteps, features))
    val_X = val_X.reshape((val_X.shape[0], timesteps, features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, features))

    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)


    # Gets first item from inner array of an array. Array halinde 4 parametrenin de sonucunu çıkardığı 
    # için bu şekilde yapmak zorunda kaldım.
    def Extract(lst):
        return [item[0] for item in lst]


    # Define the GRU model
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(features))
    model.compile(loss='mae', optimizer='adam')

    # Fit the model
    history = model.fit(train_X, train_y, epochs=100, batch_size=64,
                    validation_data=(val_X, val_y), verbose=2, shuffle=False)

    # Make predictions
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    test_pred = model.predict(test_X)

    # Invert the predictions and the original data
    train_pred = scaler.inverse_transform(train_pred)
    val_pred = scaler.inverse_transform(val_pred)
    test_pred = scaler.inverse_transform(test_pred)
    train_y_inv = scaler.inverse_transform(train_y)
    val_y_inv = scaler.inverse_transform(val_y)
    test_y_inv = scaler.inverse_transform(test_y)
                
    # Calculate MAE or in other words, accuracy
    error = 100*mae(Extract(test_y_inv), Extract(test_pred))
    accuracy = 100-error


    # Use the last sequence in the test set for prediction (assuming one prediction per sequence)
    last_sequence = test_X[-1].reshape(1, timesteps, features)
    next_day_prediction = model.predict(last_sequence)

    # Invert the prediction to the original scale
    next_day_prediction = scaler.inverse_transform(next_day_prediction)

    # Display the prediction result and RMSE
    st.write(f"Predicted Value for Next 30th Day: {next_day_prediction[0][0]:.2f}")
    if next_day_prediction[0][0] < 29.9:
        st.error("The water level is going to be under threshold")
    else:
        st.success("The water level is going to be above threshold")
    st.write("Accuracy: %", "%.2f" % accuracy)
   
    fig2, ax = plt.subplots(1,1)
    ax.plot(Extract(test_y_inv), marker='.', label="actual")
    ax.plot(Extract(test_pred), 'r', label="prediction")
    # plt.tick_params(left=False, labelleft=True) #remove ticks

    sns.despine(top=True)

    ax.set_ylabel('Lake Water Level', size=15)
    ax.set_xlabel('Time step', size=15)
    ax.legend(fontsize=15)
    st.pyplot(fig2);  

# Function to make predictions
def predict_page_60():

    # Read data from CSV file
    data = pd.read_excel('your_dataset.xlsx')  # Assuming the CSV file has columns 'precipitation', 'water_level', 'max_temp', 'min_temp', 'avg_temp', 'withdrawal'
 
    # Select the relevant columns for prediction
    data = data[['water_level', 'max_temp', 'min_temp', 'avg_temp', 'precipitation', 'withdrawal']]
    st.write(f"Today's water level value is: {data['water_level'].iloc[-1]:.2f}")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)


    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if isinstance(data, list) else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    # Prepare the data for supervised learning
    timesteps = 60
    features = data.shape[1]
    reframed = series_to_supervised(scaled_data, timesteps, 60)
    print(reframed.head())

    # Split the data into train and test sets
    values = reframed.values
    train_size = 1802
    val_size = 2403
    train = values[:train_size, :]
    val = values[train_size:val_size, :]
    test = values[val_size:, :]

    # Split the data into input and output variables
    train_X, train_y = train[:, :-features * 60], train[:, -features:]
    val_X, val_y = val[:, :-features * 60], val[:, -features:]
    test_X, test_y = test[:, :-features * 60], test[:, -features:]

    # Reshape the input data to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timesteps, features))
    val_X = val_X.reshape((val_X.shape[0], timesteps, features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, features))

    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)


    # Gets first item from inner array of an array. Array halinde 4 parametrenin de sonucunu çıkardığı 
    # için bu şekilde yapmak zorunda kaldım.
    def Extract(lst):
        return [item[0] for item in lst]


    # Define the GRU model
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(features))
    model.compile(loss='mae', optimizer='adam')

    # Fit the model
    history = model.fit(train_X, train_y, epochs=100, batch_size=64,
                    validation_data=(val_X, val_y), verbose=2, shuffle=False)

    # Make predictions
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    test_pred = model.predict(test_X)

    # Invert the predictions and the original data
    train_pred = scaler.inverse_transform(train_pred)
    val_pred = scaler.inverse_transform(val_pred)
    test_pred = scaler.inverse_transform(test_pred)
    train_y_inv = scaler.inverse_transform(train_y)
    val_y_inv = scaler.inverse_transform(val_y)
    test_y_inv = scaler.inverse_transform(test_y)
                
    # Calculate MAE or in other words, accuracy
    error = 100*mae(Extract(test_y_inv), Extract(test_pred))
    accuracy = 100-error


    # Use the last sequence in the test set for prediction (assuming one prediction per sequence)
    last_sequence = test_X[-1].reshape(1, timesteps, features)
    next_day_prediction = model.predict(last_sequence)

    # Invert the prediction to the original scale
    next_day_prediction = scaler.inverse_transform(next_day_prediction)

    # Display the prediction result and RMSE
    st.write(f"Predicted Value for Next 60th Day: {next_day_prediction[0][0]:.2f}")
    if next_day_prediction[0][0] < 29.9:
        st.error("The water level is going to be under threshold")
    else:
        st.success("The water level is going to be above threshold")
    st.write("Accuracy: %", "%.2f" % accuracy)
   
    fig2, ax = plt.subplots(1,1)
    ax.plot(Extract(test_y_inv), marker='.', label="actual")
    ax.plot(Extract(test_pred), 'r', label="prediction")
    # plt.tick_params(left=False, labelleft=True) #remove ticks

    sns.despine(top=True)

    ax.set_ylabel('Lake Water Level', size=15)
    ax.set_xlabel('Time step', size=15)
    ax.legend(fontsize=15)
    st.pyplot(fig2); 

# Function to make predictions
def predict_page_120():

    # Read data from CSV file
    data = pd.read_excel('your_dataset.xlsx')  # Assuming the CSV file has columns 'precipitation', 'water_level', 'max_temp', 'min_temp', 'avg_temp', 'withdrawal'
 
    # Select the relevant columns for prediction
    data = data[['water_level', 'max_temp', 'min_temp', 'avg_temp', 'precipitation', 'withdrawal']]
    st.write(f"Today's water level value is: {data['water_level'].iloc[-1]:.2f}")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)


    # convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if isinstance(data, list) else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    # Prepare the data for supervised learning
    timesteps = 120
    features = data.shape[1]
    reframed = series_to_supervised(scaled_data, timesteps, 120)
    print(reframed.head())

    # Split the data into train and test sets
    values = reframed.values
    train_size = 1802
    val_size = 2403
    train = values[:train_size, :]
    val = values[train_size:val_size, :]
    test = values[val_size:, :]

    # Split the data into input and output variables
    train_X, train_y = train[:, :-features * 120], train[:, -features:]
    val_X, val_y = val[:, :-features * 120], val[:, -features:]
    test_X, test_y = test[:, :-features * 120], test[:, -features:]

    # Reshape the input data to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timesteps, features))
    val_X = val_X.reshape((val_X.shape[0], timesteps, features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, features))

    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)


    # Gets first item from inner array of an array. Array halinde 4 parametrenin de sonucunu çıkardığı 
    # için bu şekilde yapmak zorunda kaldım.
    def Extract(lst):
        return [item[0] for item in lst]


    # Define the GRU model
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(features))
    model.compile(loss='mae', optimizer='adam')

    # Fit the model
    history = model.fit(train_X, train_y, epochs=100, batch_size=64,
                    validation_data=(val_X, val_y), verbose=2, shuffle=False)

    # Make predictions
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)
    test_pred = model.predict(test_X)

    # Invert the predictions and the original data
    train_pred = scaler.inverse_transform(train_pred)
    val_pred = scaler.inverse_transform(val_pred)
    test_pred = scaler.inverse_transform(test_pred)
    train_y_inv = scaler.inverse_transform(train_y)
    val_y_inv = scaler.inverse_transform(val_y)
    test_y_inv = scaler.inverse_transform(test_y)
                
    # Calculate MAE or in other words, accuracy
    error = 100*mae(Extract(test_y_inv), Extract(test_pred))
    accuracy = 100-error


    # Use the last sequence in the test set for prediction (assuming one prediction per sequence)
    last_sequence = test_X[-1].reshape(1, timesteps, features)
    next_day_prediction = model.predict(last_sequence)

    # Invert the prediction to the original scale
    next_day_prediction = scaler.inverse_transform(next_day_prediction)

    # Display the prediction result and RMSE
    st.write(f"Predicted Value for Next 120th Day: {next_day_prediction[0][0]:.2f}")
    if next_day_prediction[0][0] < 29.9:
        st.error("The water level is going to be under threshold")
    else:
        st.success("The water level is going to be above threshold")
    st.write("Accuracy: %", "%.2f" % accuracy)
   
    fig2, ax = plt.subplots(1,1)
    ax.plot(Extract(test_y_inv), marker='.', label="actual")
    ax.plot(Extract(test_pred), 'r', label="prediction")
    # plt.tick_params(left=False, labelleft=True) #remove ticks

    sns.despine(top=True)

    ax.set_ylabel('Lake Water Level', size=15)
    ax.set_xlabel('Time step', size=15)
    ax.legend(fontsize=15)
    st.pyplot(fig2); 

# Add a sidebar navigation menu
menu_choice = st.sidebar.selectbox("Navigation", ('📊 Dashbord', '📝 Manual Entry', '📈 Predict 1️⃣ month', '📈 Predict 2️⃣ months', '📈 Predict 4️⃣ months'))

# Based on the menu choice, display the respective page
if menu_choice == '📝 Manual Entry':
    manual_entry_page()
elif menu_choice == '📈 Predict 1️⃣ month':
    predict_page_30()
elif menu_choice == '📈 Predict 2️⃣ months':
    predict_page_60()
elif menu_choice == '📈 Predict 4️⃣ months':
    predict_page_120()
else:
    dashboard()
