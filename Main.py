import random
import pandas as pd
import numpy as np
from Model_Adaboost_ import Model_Adaboost
from Model_CNN import Model_CNN
from Model_Capsnet import Model_Capsnet
from Model_ELM import Model_ELM
from Model_Ensemble_Learning import Model_Ensemble_Learning
from Plot_Results import *


# Convert categorical columns to numerical representations based on unique values
def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df

def generate_building_info():
    building_types = ["Multi-story building", "Office", "House", "Warehouse", "School"]
    dimensions_of_room = (random.randint(5, 20), random.randint(5, 15))  # Tuple for room dimensions
    num_windows = random.randint(1, 5)
    window_sizes = [(random.uniform(1, 2.5), random.uniform(1, 2.5)) for _ in
                    range(num_windows)]  # Random window sizes (width, height)
    window_surface_areas = [round(width * height, 2) for width, height in
                            window_sizes]  # Calculate window surface areas (width * height)
    total_window_surface_area = round(sum(window_surface_areas), 2)  # Total window surface area

    thermal_resistance_window = round(random.uniform(0.5, 2), 2)  # Thermal resistance of windows in m²·K/W
    thermal_resistance_roof = round(random.uniform(2, 6), 2)  # Thermal resistance of roof in m²·K/W
    thermal_resistance_wall = round(random.uniform(1, 3), 2)  # Thermal resistance of walls in m²·K/W
    solar_radiation = round(random.uniform(150, 350), 2)  # Solar radiation in W/m²

    building_type = random.choice(building_types)

    entry_time = random.randint(8, 10)
    exit_time = random.randint(5, 7)
    Power = round(random.uniform(1, 10), 2)  # Random power consumption in kW
    Energy_Consumed = round(random.uniform(0.01, 0.5), 2)  # Random energy consumed in kWh

    holidays = ["New Year", "Pongal", "Thiruvalluvar Day", "Thai Poosam", "Republic Day", "Good Friday",
                "Telugu New Year Day", "Ramzan", "Tamil New Years Day", "Mahaveer Jayanthi", "Bakrid", "Muharram",
                "Independence Day", "Krishna Jayanthi", "Vinayakar Chathurthi", "Milad-un-Nabi", "Gandhi Jayanthi",
                "Ayutha Pooja", "Vijaya Dasami", "Deepavali", "Christmas"]
    activities = ["Meetings", "Breaks", "Workshops", "Training sessions"]
    daily_activities = ", ".join(
        random.sample(activities, k=random.randint(1, 3)))  # Join activities into a single string

    building_info = {
        "Type of building": building_type,
        # "Dimensions of room": f"{dimensions_of_room[0]}m x {dimensions_of_room[1]}m x {dimensions_of_room[2]}m",
        "Room D1": dimensions_of_room[0],  # Room dimension D1
        "Room D2": dimensions_of_room[1],  # Room dimension D2
        "Number of windows": num_windows,
        "Total window surface area (m²)": total_window_surface_area,
        "Thermal resistance (windows)(m²·K/W)": thermal_resistance_window,
        "Thermal resistance (roof)(m²·K/W)": thermal_resistance_roof,
        "Thermal resistance (walls) (m²·K/W)": thermal_resistance_wall,
        "Solar radiation (W/m²)": solar_radiation,
        "Entry time (am)": entry_time,
        "Exit time (pm)": exit_time,
        "Holidays": random.choice(holidays),
        "Activities & Schedule": daily_activities,
        "Power (kW)": Power,
        "Energy Consumed (kWh)": Energy_Consumed
    }

    return building_info


# Read the Dataset
an = 0
if an == 1:
    data = []
    for _ in range(2500):
        building_info = generate_building_info()
        data.append(building_info)
    df = pd.DataFrame(data)
    csv_file_path = "./Dataset/generated_building_info.csv"
    df.to_csv(csv_file_path, index=False)
    datas = pd.read_csv("./Dataset/generated_building_info.csv")
    datas = datas.drop('Energy Consumed (kWh)', axis=1)
    columns_to_convert = ['Type of building', 'Holidays', 'Activities & Schedule']
    datas = convert_to_numeric(datas, columns_to_convert)
    Target = df['Energy Consumed (kWh)']
    Data = np.asarray(datas)
    Targets = np.reshape(np.asarray(Target), (-1, 1))
    np.save('Data.npy', Data)
    np.save('Target.npy', Targets)


# Classification by Varying Activation fuction
an = 0
if an == 1:
    EVAL = []
    Feat = np.load('Data.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Activation = ['linear', 'relu', 'leaky relu', 'tanH', 'sigmoid', 'softmax']
    for Act in range(len(Activation)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:]
        Eval = np.zeros((5, 25))
        Eval[0, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Activation[Act])
        Eval[1, :], pred2 = Model_Adaboost(Train_Data, Train_Target, Test_Data, Test_Target, Activation[Act])
        Eval[2, :], pred3 = Model_Capsnet(Train_Data, Train_Target, Test_Data, Test_Target, Activation[Act])
        Eval[3, :], pred4 = Model_ELM(Train_Data, Train_Target, Test_Data, Test_Target, Activation[Act])
        Eval[4, :], pred5 = Model_Ensemble_Learning(Train_Data, Train_Target, Test_Data, Test_Target, Activation[Act])
        EVAL.append(Eval)
    np.save('Eval_all_Act.npy', np.asarray(EVAL))  # Save the Eval all


Plot_Batch_size()
Plot_Activation()
