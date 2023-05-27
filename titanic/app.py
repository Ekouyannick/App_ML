import pandas as pd
import streamlit as st
import numpy as np
from joblib import load

def layout():
    # Sidebar
    st.sidebar.title("Liste des Paramètres ")
    st.sidebar.write("*Tous les champs sont obligatoires*")
    
@st.cache
def lire_fichier():
    path_data = 'titanic/X_train_up.csv'
    df = pd.read_csv(path_data)
    return df

def parametres(df):
    
    defaut = 'TOUT'
    
    #### CLASSE #####  
    pclass_val = sorted(list(df['pclass'].unique()))
    pclass_options = st.sidebar.selectbox("Catégorie de Classe", pclass_val, index = 0)
    
    #### SEXE #####  
    sex_val = sorted(list(df['sex'].unique()))
    sex_options = st.sidebar.selectbox("Sexe ( 0.Féminin, 1.Masculin )", sex_val, index = 0)
    
    #### AGE #####
    min_age, max_age = 0, 100
    age_options = st.sidebar.slider('Age du Passager ( 0 à 100 ): ', min_age, max_age, value = min_age, step = 1)

    #### PRIX DU BILLET #####
    prix_options = st.sidebar.number_input('Prix du billet ($)', min_value = 5, max_value = 600, step = 10)
    
    #### ADULTE MALE #####  
    adulte_male_val = sorted(list(df['adult_male'].unique()))
    adulte_male_options = st.sidebar.selectbox("Adulte Mâle ( 0.Non, 1.Oui )", adulte_male_val, index = 0)
    
    #### SEUL #####  
    seul_val = sorted(list(df['alone'].unique()))
    seul_options = st.sidebar.selectbox("Êtes vous seul ? ( 0.Non, 1.Oui )", seul_val, index = 0)
    
    #### Southampton #####  
    embark_val = sorted(list(df['Southampton'].unique()))
    embark_options = st.sidebar.selectbox("Embarquer a Southampton ( 0.Non, 1.Oui )", embark_val, index = 0)
    
    #### Famille #####
    famille_options = st.sidebar.number_input('Nombre Personne(s) de la famille', min_value = 0, max_value = 10, step = 1)

    return pclass_options, sex_options, age_options, prix_options, adulte_male_options, seul_options, embark_options, famille_options


def load_model():
    model = load(filename='titanic/RF_model.joblib')
    return model
    
def prediction(model, data):
    resultat = model.predict(data)
    return resultat



########################################### MAIN ###########################################################
############################################################################################################
if __name__ == "__main__":
    st.set_page_config(
        page_title="App Titanic",
        layout="centered"
    )
    st.title("Application de survie sur le Titanic")
    st.header("*# Ekou Yannick*")
    st.subheader("@2023")
    
    layout()
    
    ############ READ DATA ##########################
    df = lire_fichier()
    # st.write(df.head())
    model = load_model()
    
    ############ PARAMETRES ##########################
    formulaire = parametres(df)
    
    st.caption('Ci-dessous - Dataframe d\'informations du passager du Titanic.')
    
    ############ Données Formuliaire ##########################
        # Création du DataFrame à partir du tuple
    donnees = [formulaire]
    df_formulaire = pd.DataFrame(donnees, columns=['pclass', 'sex', 'age', 'fare', 'adult_male', 'alone', 'Southampton', 'family'])
    st.write(df_formulaire)
    
    # Bouton pour appeller le Modèle
    if st.sidebar.button('Valider'):
        resultat = prediction(model, df_formulaire)
        st.info(f'Resultat: {resultat}')
        
        if resultat[0] == 1:
            st.success('Le Passager est un miraculeux il est en vie !')     
        else:
            st.error('Mes Condoléances, le Passager n\'a pas survécu au naufrage')
    
    
    