import streamlit as st
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import random

df = pd.read_csv("adap-ecosys-dataset.csv")
df = df.drop(["SpeciesName"], axis='columns')

df = pd.get_dummies(df, columns=['Diet'])
df = pd.get_dummies(df, columns=['Habitat'])

def norm(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def create_prox_matrix(df):
    matrix = np.zeros([len(df), len(df)])
    for i in np.arange(len(df)):
        for j in np.arange(len(df)):
            matrix[i, j] = norm(np.array(df.iloc[i].tolist()), np.array(df.iloc[j].tolist()))
    return matrix

def proximity_to_condensed(matrix):
    return sch.distance.squareform(matrix)

def generate_dendrogram(condensed_dist_matrix, timestep, method='centroid'):
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(condensed_dist_matrix, method=method))
    plt.title(f'Dendrogram (TimeStep = {timestep})')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    st.pyplot(plt.gcf())

st.title("Ecosystem Animal Clustering Tool")


st.sidebar.header("Add New Animal")

size = st.sidebar.number_input("Size", min_value=0.0, max_value=1000.0, value=5.0)
speed = st.sidebar.number_input("Speed", min_value=0.0, max_value=1000.0, value=5.0)
color = st.sidebar.number_input("Color", min_value=0.0, max_value=1000.0, value=50.0)
aggression = st.sidebar.number_input("Aggression", min_value=0.0, max_value=1000.0, value=5.0)
timestep = st.sidebar.slider("TimeStep", min_value=0, max_value=19, value=0)

diet = st.sidebar.selectbox("Diet", options=["Carnivore", "Herbivore"])
habitat = st.sidebar.selectbox("Habitat", options=["Desert", "Forest", "Mountain", "Ocean", "Plains", "Wetlands"])

if st.sidebar.button("Add New Animal"):
    new_animal = pd.DataFrame({
        'Size': [size],
        'Speed': [speed],
        'Color': [color],
        'Aggression': [aggression],
        'TimeStep': [timestep],
        'SpeciesID': random.random(),
        'Diet_Carnivore': [1 if diet == "Carnivore" else 0],
        'Diet_Herbivore': [1 if diet == "Herbivore" else 0],
        'Habitat_Desert': [1 if habitat == "Desert" else 0],
        'Habitat_Forest': [1 if habitat == "Forest" else 0],
        'Habitat_Mountain': [1 if habitat == "Mountain" else 0],
        'Habitat_Ocean': [1 if habitat == "Ocean" else 0],
        'Habitat_Plains': [1 if habitat == "Plains" else 0],
        'Habitat_Wetlands': [1 if habitat == "Wetlands" else 0],
    })
    df = pd.concat([new_animal, df]).reset_index(drop=True)
    st.sidebar.success("New animal added successfully!")

st.sidebar.header("Mutate Existing Animal")

species_id_input = st.sidebar.text_input("Enter SpeciesID to mutate:", "", key="species_id_input")

if species_id_input:
    try:
        species_id = int(species_id_input)
        if species_id in df['SpeciesID'].values:
            animal_to_mutate = df[df['SpeciesID'] == species_id].iloc[0]

            size = st.sidebar.number_input("Size", min_value=0.0, max_value=1000.0, value=float(animal_to_mutate['Size']), key="mutate_size")
            speed = st.sidebar.number_input("Speed", min_value=0.0, max_value=1000.0, value=float(animal_to_mutate['Speed']), key="mutate_speed")
            color = st.sidebar.number_input("Color", min_value=0.0, max_value=1000.0, value=float(animal_to_mutate['Color']), key="mutate_color")
            aggression = st.sidebar.number_input("Aggression", min_value=0.0, max_value=1000.0, value=float(animal_to_mutate['Aggression']), key="mutate_aggression")
            timestep = st.sidebar.slider("TimeStep", min_value=0, max_value=19, value=int(animal_to_mutate['TimeStep']), key="mutate_timestep")  # Ensure value is an integer

            diet = st.sidebar.selectbox(
                "Diet", 
                options=["Carnivore", "Herbivore"], 
                index=0 if animal_to_mutate['Diet_Carnivore'] == 1 else 1, 
                key="mutate_diet"
            )

            habitat_options = ["Desert", "Forest", "Mountain", "Ocean", "Plains", "Wetlands"]
            habitat_index = np.where(animal_to_mutate.filter(like='Habitat_').values == 1)[0]
            habitat_index = habitat_index[0] if len(habitat_index) > 0 else 0  # Default to 0 if none found
            habitat = st.sidebar.selectbox(
                "Habitat",
                options=habitat_options,
                index=int(habitat_index),
                key="mutate_habitat"
            )

            if st.sidebar.button("Update Animal"):
                # Update the animal in the DataFrame
                df.loc[df['SpeciesID'] == species_id, ['Size', 'Speed', 'Color', 'Aggression', 'TimeStep', 'Diet_Carnivore', 'Diet_Herbivore', 'Habitat_Desert', 'Habitat_Forest', 'Habitat_Mountain', 'Habitat_Ocean', 'Habitat_Plains', 'Habitat_Wetlands']] = [
                    size, speed, color, aggression, timestep,
                    1 if diet == "Carnivore" else 0,
                    1 if diet == "Herbivore" else 0,
                    1 if habitat == "Desert" else 0,
                    1 if habitat == "Forest" else 0,
                    1 if habitat == "Mountain" else 0,
                    1 if habitat == "Ocean" else 0,
                    1 if habitat == "Plains" else 0,
                    1 if habitat == "Wetlands" else 0
                ]
                st.sidebar.success("Animal updated successfully!")
        else:
            st.sidebar.error("Invalid SpeciesID. Please enter a valid ID.")
    except ValueError:
        st.sidebar.error("Invalid input. Please enter a numeric SpeciesID.")

subset_size = 100
selected_timestep = st.slider("Select TimeStep for Dendrogram", 0, 19, 0)

df_time = df[df['TimeStep'] == selected_timestep]
df_time = df_time.astype(float)

proximity_matrix = create_prox_matrix(df_time[:subset_size  ])
condensed_dist_matrix = proximity_to_condensed(proximity_matrix)

generate_dendrogram(condensed_dist_matrix, selected_timestep)