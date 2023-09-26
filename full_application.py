# -*- coding: utf-8 -*-
"""
@author: thomas TRANG
"""
#Basics
import numpy as np
import pandas as pd
import base64
import os
import math
import warnings
from unidecode import unidecode
import requests
import subprocess
import re
#IA
from annoy import AnnoyIndex
#Déploiement    
import hydralit_components as hc
import streamlit as st 
import hydralit as hy
#Visualisation
import seaborn as sns
import matplotlib.pyplot as plt


app = hy.HydraApp(title="Football application by Thomas Trang",hide_streamlit_markers=True,layout='wide')


@app.addapp(title='Système de recommandation')
def Accueil():    
    warnings.simplefilter("ignore")
    
    # variables
    all_name = "All"
    photo_profile_dir = "profile_photo/"
    
    for file in os.listdir():
        if file.endswith(".png"):
            os.remove(file)
    
    # load data
        
    def value_to_float(x):
        if type(x) == float or type(x) == int:
            return x
        if 'K' in x:
            if len(x) > 1:
                return float(x.replace('K', '')) * 1000
            return 1000.0
        if 'M' in x:
            if len(x) > 1:
                return float(x.replace('M', '')) * 1000000
            return 1000000.0
        if 'B' in x:
            return float(x.replace('B', '')) * 1000000000
        return x
    
    @st.cache_data()
    # Data processing & importation
    def load_data():
        df = pd.read_csv('FIFA22_official_data.csv')
        df["Name"] = df["Name"].apply(lambda name: unidecode(name))
        # df["positions_list"] = df["positions"].apply(lambda x: x.split(","))
        df=df[df['Overall']>75]
        # df["contract"] = df["contract"].astype(int)
    
        df['Club_cat']=df['Club'].astype('category').cat.codes
        df['Nationality_cat']=df['Nationality'].astype('category').cat.codes
        df['Wage']=df['Wage'].str.replace('€', '').apply(value_to_float)
        df['Value']=df['Value'].str.replace('€', '').apply(value_to_float)
        
        df['Height']=df['Height'].str.replace('[dA-Za-z]', '').astype('int')
        df['Weight']=df['Weight'].str.replace('[dA-Za-z]', '').astype('int')
        df=df.rename(columns={"Value": "Value (€)", "Wage": "Wage (€)"})
        df['Value (€)']=df['Value (€)'].astype('float')
        df['Wage (€)']=df['Wage (€)'].astype('float')
        df['zone_terrain']=df['Best Position'].apply(lambda x : "Milieu" if x.find('M')!=-1 else ( 'Defenseur' if (x.find('B')!=-1)  else ( "Gardien" if (x.find('K')!=-1) else 'Attaquant') ))
        df=df[~df['Name'].str.contains(r'\d')]
        return df
    
    
    df = load_data()
    df=df.sort_values('Overall',ascending=False)
    nationality_list = list(df["Nationality"].unique())
    player_list = list(df["Name"].unique())
    positions_list=list(df["Best Position"].unique())
    
    
    
    cols_to_keep=['Name','Age', 'Photo', 'Nationality', 'Best Position', 'Flag', 'Club', 'Club Logo', 'Jersey Number', 'Overall', 'Height',
     'Weight', 'Interceptions', 'GKReflexes', 'ShotPower', 'Jumping', 'GKHandling', 'SlidingTackle', 'HeadingAccuracy', 'ShortPassing',
     'Stamina', 'Balance', 'Vision', 'Aggression', 'BallControl', 'GKPositioning', 'Potential', 'Acceleration', 'StandingTackle', 'Release Clause',
     'GKDiving', 'Dribbling', 'Positioning', 'Penalties', 'Wage', 'Value', 'Preferred Foot', 'LongPassing', 'Agility', 'Composure', 'Finishing',
     'Strength', 'GKKicking', 'Volleys', 'FKAccuracy', 'DefensiveAwareness', 'LongShots', 'Crossing', 'Curve', 'Reactions', 'SprintSpeed']
    
    show_columns = [
        "Photo",
        "Name",
        "Club Logo",
        'Flag',
        "Age",
        "Best Position",
        "Overall",
        "Potential",
        "Preferred Foot",
        "Value (€)",
        
    ]
    
    show_columns_target = [
        "Photo",
        "Name",
        "Club Logo",
        'Flag',
        "Age",
        "Best Position",
        "Overall",
        "Potential",
        "Preferred Foot",
        
        
    ]
    
    default_columns_to_compare = ['Overall','StandingTackle','SlidingTackle',
        "Potential",
        "Finishing",
        "Dribbling",
        "BallControl",
        "Acceleration",
        "Agility",
    
    
    ]
    default_positions = ["ST", "CF", 'CAM','CM', "RW", "LW",'GK','CB','LB','CDM']
    
    
    
    
    
    possible_columns_to_compare = [
        "Overall",
        "Potential",
        "Crossing",
        "Finishing",
        "HeadingAccuracy",
        "ShortPassing",
        "Volleys",
        "Dribbling",
        "Curve",
        "FKAccuracy",
        "LongPassing",
        "BallControl",
        "Acceleration",
        "SprintSpeed",
        "Agility",
        "Reactions",
        "Balance",
        "ShotPower",
        "Jumping",
        "Stamina",
        "Strength",
        "LongShots",
        "Aggression",
        "Interceptions",
        "Positioning",
        "Vision",
        "Penalties",
        "Composure",
        "DefensiveAwareness",
        "StandingTackle",
        "SlidingTackle",
        "GKDiving",
        "GKHandling",
        "GKKicking",
        "GKPositioning",
        "GKReflexes",
    ]
    over_theme = {'txc_inactive': '#FFFFFF'}

    
    
    
    ################################################################
    # css style
    hide_streamlit_style = """
                <style>
                    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 463px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 500px;
            margin-left: -500px;
        }
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    
    
    ##################################################################
    # sidebar filters
    st.sidebar.title(" &#128270; Filtres")
    
    st.sidebar.title("Sélectionnez le joueur à comparer:")
    
    target_player_name = st.sidebar.selectbox("Joueur:", [""] + player_list)
    
    target_player_name = target_player_name.strip()
    
    st.sidebar.title("Caractéristiques du joueur:")
    
    
    
    
    age = st.sidebar.slider("Max. age:", min_value=15, max_value=42, value=36)
    
    positions = st.sidebar.multiselect("Position:", options=positions_list, default=default_positions)
    
    transfer_fee = 1000000 * float(
        st.sidebar.text_input("Prix Maximal (€M):", "500")
    )
    
    wage = 1000 * float(st.sidebar.text_input("Salaire maximal (en kilo €):", "1000"))
    
    columns_to_compare = st.sidebar.multiselect(
        "Caractéristiques à comparer :", possible_columns_to_compare, default=default_columns_to_compare
    )
    
    top_K = st.sidebar.slider("Nombre de joueurs similaires", min_value=0, max_value=15, value=5)
    
    is_scan = st.sidebar.button("LANCER LA RECHERCHE")
    
    
    st.sidebar.header(" &#128204; Contact Info")
    st.sidebar.info("LinkedIn : [https://www.linkedin.com/in/thomas-trang100/](https://www.linkedin.com/in/thomas-trang100/)")
    st.sidebar.info("GitHub  : [https://github.com/thomastrg](https://github.com/thomastrg)")
    
    
    
    ##############################################################################
    # if detect button is clicked, then show the main components of the dashboard
    
    
    def filter_positions(row, positions):
        for p in positions:
            if p in row["Best Position"]:
                return True
        return False
    
    
    def upload_local_photo(file):
        file_ = open(file, "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        return data_url
    
    
    def download_photo_url(url):
        photo_name = "_".join(url.split("/")[-3:])
    
        r = requests.get(url, allow_redirects=True)
        open(photo_name, "wb").write(r.content)
    
        return photo_name
    
    
    def create_table(data, width=100, class_="", image_height=105, image_width=105):
        if len(class_) > 0:
            table = f'<table class="{class_}" style="text-align: center; width:{width}%">'
        else:
            table = f'<table style="text-align: center; width:{width}%">'
    
        # create header row
        header_html = "<tr>"
        for col in data.columns:
            if col == "Photo":
                header_html = header_html + "<th>Photo</th>"
            elif col == "Value (€)":
                header_html = header_html + "<th>Valeur (€M)</th>"
            #elif col == "player_hashtags":
              #  header_html = header_html + "<th>Caractéristiques</th>"
            else:
                header_html = header_html + f"<th>{col.capitalize()}</th>"
        header_html = header_html + "<tr>"
    
        all_rows_html = ""
        for row_index in range(len(data)):
            row_html = "<tr>"
            row = data.iloc[row_index]
            for col in data.columns:
                if col == "Photo":
                    local_photo = download_photo_url(row[col])
                    data_url = upload_local_photo(local_photo)
                    row_html = (
                        row_html
                        + f'<td><img src="data:image/gif;base64,{data_url}" height="{image_height} width="{image_width}"></img></td>'
                    )
                elif col == "Flag":
                    local_photo = download_photo_url(row[col])
                    data_url = upload_local_photo(local_photo)
                    row_html = (
                        row_html
                        + f'<td><img src="data:image/gif;base64,{data_url}" height="54 width="54"></img></td>'
                    )
                elif col == "Club Logo":
                    local_photo = download_photo_url(row[col])
                    data_url = upload_local_photo(local_photo)
                    row_html = (
                        row_html
                        + f'<td><img src="data:image/gif;base64,{data_url}" height="60 width="60"></img></td>'
                    )
                elif row[col] == None:
                    row_html = row_html + "<td></td>"
                #elif col == "Best Position":
                 #   row_html = row_html + f'<td>{", ".join(eval(row[col]))}</td>'
                else:
                    row_html = row_html + f"<td>{row[col]}</td>"
            row_html = row_html + "</tr>"
            all_rows_html = all_rows_html + row_html
    
        table = table + header_html + all_rows_html + "</table>"
        st.markdown(table, unsafe_allow_html=True)
    
    
    # @st.cache(allow_output_mutation=True)
    def scan(target_player,positions, transfer_fee, wage, age):
        df = load_data()
    
        target_player_KPIs = target_player[columns_to_compare].to_numpy()[0]
    
        df = df.loc[df["Name"] != target_player_name]
        df = df[df["Age"] <= age]
        df = df[(df["Value (€)"] <= transfer_fee) & (df["Wage (€)"] <= wage)]
    
        df["filter_positions"] = df.apply(
            lambda row: filter_positions(row, positions), axis=1
        )
        search_space = df.loc[df["filter_positions"] == True]
        search_space.reset_index(drop=True, inplace=True)
    
        # search_space["label"] = pd.Series(list(clf.fit_predict(X)))
        # search_space["score"] = pd.Series(list(clf.score_samples(X)))
        # search_space.sort_values(by=["score"], inplace=True)
    
        # calculate ANNOY
        
        #Colonnes comparées dans KNN
        annoy = AnnoyIndex(len(columns_to_compare), "euclidean")
        search_space_array = search_space[columns_to_compare].to_numpy()
    
        for i in range(search_space_array.shape[0]):
            annoy.add_item(i, search_space_array[i, :])
        annoy.build(n_trees=1000)
    
        indices = annoy.get_nns_by_vector(target_player_KPIs, top_K)
        return pd.concat([search_space.iloc[index : index + 1, :] for index in indices])
    
    

    
    
    
    import time
    
    if is_scan:
        
    
    # a dedicated single loader 
    
        target_player = df.loc[df["Name"] == target_player_name]
        target_player["Value (€)"]=target_player["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        target_player_age = target_player["Age"].iloc[0]
        target_player_teams = target_player["Club"].iloc[0]
        url = target_player["Photo"].iloc[0]
        local_photo = download_photo_url(url)
        data_url = upload_local_photo(local_photo)
        st.title("Joueur sélectionné:")
    
        joueur_ligne=pd.DataFrame(target_player)
        
        #joueur_ligne["Value (€)"] = target_player["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        
        if joueur_ligne.shape[0]>1:
            create_table(target_player[show_columns].head(1))
        else :
            create_table(target_player[show_columns])
    
        with hc.HyLoader(f"Calcul en cours ... l'IA recherche les {top_K} joueurs les plus similaires",hc.Loaders.standard_loaders,index=3):
            time.sleep(5)
            
        result = scan(target_player,positions, transfer_fee, wage, age)
        st.subheader("")
        st.subheader(f"\n**Les _{top_K}_ joueurs les plus similaires sont **:")
        
        result["Value (€)"] = result["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        create_table(result[show_columns])
    else:
        st.subheader("Bienvenue, cette application lie mes 2 passions : la Data Science et le football &#9917. L'application est composée de 3 onglets.")
        st.subheader("")
        st.subheader(" ")       
        st.title('&#128373; Le Système de Recommandation de Footballeurs')
        st.subheader(" ")
        st.subheader("Ce système de recommandation vous permettra de trouver le joueur qui correspondra à chacun de vos critères. ")
        st.subheader("")
        st.subheader("Entrez le nom d'un joueur similaire à celui que vous recherchez.")
        st.subheader("")
        st.subheader(
            "Puis adaptez les caractéristiques en fonction de ce que vous cherchez chez vos potentielles recrues.")
        st.subheader("")
        st.subheader("L'algorithme vous proposera une liste de joueurs correspondant à vos attentes." )
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.subheader("")

        
        st.subheader(
            "Si vous avez des questions ou des feedbacks, n'hésitez pas à me contacter via LinkedIn : https://www.linkedin.com/in/thomas-trang100/"
        )
        st.title(
            "Thomas TRANG"
        )

@app.addapp(title='Data Visualisation')
def app3():
    import pandas as pd
    import numpy as np
    
    import re
    import requests
    import base64
    from unidecode import unidecode
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    
    from annoy import AnnoyIndex
    
    def value_to_float(x):
        if type(x) == float or type(x) == int:
            return x
        if 'K' in x:
            if len(x) > 1:
                return float(x.replace('K', '')) * 1000
            return 1000.0
        if 'M' in x:
            if len(x) > 1:
                return float(x.replace('M', '')) * 1000000
            return 1000000.0
        if 'B' in x:
            return float(x.replace('B', '')) * 1000000000
        return x
    
    @st.cache_data()
    # Data processing & importation
    def load_data():
        df = pd.read_csv('FIFA22_official_data.csv')
        df["Name"] = df["Name"].apply(lambda name: unidecode(name))
        # df["positions_list"] = df["positions"].apply(lambda x: x.split(","))
        df=df[df['Overall']>75]
        # df["contract"] = df["contract"].astype(int)
    
        df['Club_cat']=df['Club'].astype('category').cat.codes
        df['Nationality_cat']=df['Nationality'].astype('category').cat.codes
        df['Wage']=df['Wage'].str.replace('€', '').apply(value_to_float)
        df['Value']=df['Value'].str.replace('€', '').apply(value_to_float)
        
        df['Height']=df['Height'].str.replace('[dA-Za-z]', '').astype('int')
        df['Weight']=df['Weight'].str.replace('[dA-Za-z]', '').astype('int')
        df=df.rename(columns={"Value": "Value (€)", "Wage": "Wage (€)"})
        df['Value (€)']=df['Value (€)'].astype('float')
        df['Wage (€)']=df['Wage (€)'].astype('float')
        df['zone_terrain']=df['Best Position'].apply(lambda x : "Milieu" if x.find('M')!=-1 else ( 'Defenseur' if (x.find('B')!=-1)  else ( "Gardien" if (x.find('K')!=-1) else 'Attaquant') ))
        df=df[~df['Name'].str.contains(r'\d')]
        df['Evolution']=df['Potential']-df['Overall']
        return df
    df = load_data()
    
    st.title("&#127760; Bienvenue sur l'onglet Data Visualisation")
    st.subheader('')
    st.subheader("Choisissez une visualisation à travers le menu déroulant ci-dessous.")
    st.subheader("")
    st.subheader('')
    viz_list=["Classement des 100 meilleurs joueurs du monde en 2022",
              "Comparaison des classements des meilleurs joueurs par club",
              "Comparaison des classements des meilleurs joueurs par pays",          
              'Classement du joueur le plus cher par pays en 2022',
              "Classement des pays ayant la plus grande valeur marchande", 
              "Situer les 75 meilleurs joueurs du monde sur une carte", 
              "Classement des joueurs avec les meilleurs potentiels d'évolution en 2022",
              "Distribution des tailles et poids des n meilleurs joueurs du monde",
              "Classement des clubs ayant les effectifs les plus chers du monde",
              "Classement des clubs ayant les meilleurs espoirs du monde",
              "Classement des clubs ayant les plus grands joueurs de taille en moyenne",
              "Classement des clubs ayant les joueurs les plus lourds en moyenne", 
              "La répartition des meilleurs pieds des n meilleurs joueurs du monde"]
    viz = st.selectbox("Sélectionnez une visualisation que vous désirez afficher :", options=viz_list)
    st.subheader("")
    st.subheader("")
    st.subheader("")
    
    
    def upload_local_photo(file):
        file_ = open(file, "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        return data_url
    
    
    def download_photo_url(url):
        photo_name = "_".join(url.split("/")[-3:])
    
        r = requests.get(url, allow_redirects=True)
        open(photo_name, "wb").write(r.content)
    
        return photo_name
    def create_table(data, width=100, class_="", image_height=105, image_width=105):
        if len(class_) > 0:
            table = f'<table class="{class_}" style="text-align: center; width:{width}%">'
        else:
            table = f'<table style="text-align: center; width:{width}%">'
    
        # create header row
        header_html = "<tr>"
        for col in data.columns:
            if col == "Photo":
                header_html = header_html + "<th>Photo</th>"
            elif col == "Value (€)":
                header_html = header_html + "<th>Valeur (€M)</th>"
            #elif col == "player_hashtags":
              #  header_html = header_html + "<th>Caractéristiques</th>"
            else:
                header_html = header_html + f"<th>{col.capitalize()}</th>"
        header_html = header_html + "<tr>"
    
        all_rows_html = ""
        for row_index in range(len(data)):
            row_html = "<tr>"
            row = data.iloc[row_index]
            for col in data.columns:
                if col == "Photo":
                    local_photo = download_photo_url(row[col])
                    data_url = upload_local_photo(local_photo)
                    row_html = (
                        row_html
                        + f'<td><img src="data:image/gif;base64,{data_url}" height="{image_height} width="{image_width}"></img></td>'
                    )
                elif col == "Flag":
                    local_photo = download_photo_url(row[col])
                    data_url = upload_local_photo(local_photo)
                    row_html = (
                        row_html
                        + f'<td><img src="data:image/gif;base64,{data_url}" height="{54} width="{54}"></img></td>'
                    )
                elif col == "Club Logo":
                    local_photo = download_photo_url(row[col])
                    data_url = upload_local_photo(local_photo)
                    row_html = (
                        row_html
                        + f'<td><img src="data:image/gif;base64,{data_url}" height="60 width="60"></img></td>'
                    )
                elif row[col] == None:
                    row_html = row_html + "<td></td>"
                #elif col == "Best Position":
                 #   row_html = row_html + f'<td>{", ".join(eval(row[col]))}</td>'
                else:
                    row_html = row_html + f"<td>{row[col]}</td>"
            row_html = row_html + "</tr>"
            all_rows_html = all_rows_html + row_html
    
        table = table + header_html + all_rows_html + "</table>"
        st.markdown(table, unsafe_allow_html=True)
    
    
    top100_valuable_players=df.sort_values('Value (€)',ascending=False).head(200)
    # Le joueur le plus cher de chaque pays parmi les 200 joueurs les plus chers du marché en 2022
    def joueur_le_plus_cher_de_chaque_pays():
        df_agg = top100_valuable_players.groupby(["Nationality",'Name']).agg({'Value (€)':sum})
        g = df_agg['Value (€)'].groupby('Nationality', group_keys=False)
        res = g.apply(lambda x: x.sort_values(ascending=False).head(1))
        res=pd.DataFrame(res).sort_values('Value (€)',ascending=False)
        res["Value (€)"] = res["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        res=res.reset_index()
        test=df[['Photo','Flag','Age','Name','Club Logo']]
        res=pd.merge(res,test)
        res=res[["Photo",'Name','Age','Flag',"Value (€)"]]
        
        res.index = np.arange(1, len(res) + 1)
        res.index.name='Classement'
        res=res.reset_index()
        create_table(res)
        
    def meilleurs_potentiels(nb_joueurs):
        df['Evolution']=df['Potential']-df['Overall']
        a=df[df['Wage (€)']!=0].sort_values('Evolution',ascending=False)[['Name','Age','Club','Overall','Potential','Evolution','Value (€)']].head(nb_joueurs)
        test=df[['Photo','Flag','Age','Name','Club Logo']]
        res=pd.merge(test,a).sort_values('Evolution',ascending=False)
        res=res[["Photo",'Name','Age','Club Logo','Flag','Overall','Potential','Evolution',"Value (€)"]]
        res["Value (€)"] = res["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        res.index = np.arange(1, len(res) + 1)
        res.index.name='Classement'
        res=res.reset_index()
        create_table(res)
        
    def x_meilleurs_joueurs_dumonde():
        res=df.sort_values('Overall',ascending=False)[['Photo','Name','Age',"Club Logo","Flag",'Best Position','Value (€)']].head(100)
        res["Value (€)"] = res["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        res.index = np.arange(1, len(res) + 1)
        res.index.name='Classement'
        res=res.reset_index()
        create_table(res)

    def x_plus_chers_joueurs_dumonde():
        res=df.sort_values("Value (€)",ascending=False)[['Photo','Name','Age',"Club Logo","Flag",'Best Position','Value (€)']].head(100)
        res["Value (€)"] = res["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        res.index = np.arange(1, len(res) + 1)
        res.index.name='Classement'
        res=res.reset_index()
        create_table(res)

    def meilleurs_joueurs_equipes():
        st.sidebar.title("Filtres")
        res=df[df['Club'].isin(clubs)].sort_values('Overall',ascending=False)[['Photo','Name','Age','Overall',"Club Logo","Flag",'Best Position','Value (€)']]
        create_table(res)
        
        
        
    def viz_somme_des_valeurs_des_pays():
        a=pd.DataFrame(top100_valuable_players.groupby(["Nationality"]).sum()['Value (€)'].sort_values(ascending=False))
        st.title('Somme des valeurs marchandes des 200 joueurs les plus chers du monde regroupés par pays')
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x=a['Value (€)'],y=a.index,data=a, palette='Spectral')
        st.pyplot(fig)
        
        st.header('On voit que les nationalités ont un impact sur le prix et le talent de ses joueurs.')
        
            
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    #Plot map
    import folium
    from streamlit_folium import folium_static
    from folium.plugins import MarkerCluster
    
    geolocator = Nominatim(user_agent="full_application.py")
    def geolocate(country):
            try:
                geocode = RateLimiter(geolocator.geocode, min_delay_seconds=4)
                # Geolocate the center of the country
                loc = geolocator.geocode(country)
                # And return latitude and longitude
                return (loc.latitude, loc.longitude)
            except:
                # Return missing value
                return None
    
    def placer_sur_une_carte_les_joueurs():
        df['Nationality']=df['Nationality'].replace(['Korea Republic'],'Korea')
        top75_bestplayers=df[df['Overall']>85]
        #Get lat & long of countries
        top75_bestplayers['geolocate']=top75_bestplayers['Nationality'].apply(geolocate)
        top75_bestplayers[['latitude', 'longitude']]=pd.DataFrame(top75_bestplayers['geolocate'].tolist(), index=top75_bestplayers.index)

        
        #empty map
        world_map= folium.Map(tiles="cartodbpositron")
        marker_cluster = MarkerCluster().add_to(world_map)
        #for each coordinate, create circlemarker of user percent
        for i in range(len(top75_bestplayers)):
                lat = top75_bestplayers.iloc[i]['latitude']
                long = top75_bestplayers.iloc[i]['longitude']
                radius=10
                popup_text = """{}, {}
                            Rating : {}
                            """
                popup_text = popup_text.format(top75_bestplayers.iloc[i]['Name'],top75_bestplayers.iloc[i]['Club'],
                                           top75_bestplayers.iloc[i]['Overall']
                                           )
                folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
        
        folium_static(world_map,width=1500,height=700)
        
    def meilleur_pied(nb):
        res= df.sort_values('Overall',ascending=False).head(nb)
        fig = plt.figure(figsize = (10, 5))
        res['Preferred Foot'].value_counts().plot(kind='pie',autopct='%1.0f%%', pctdistance=0.5, labeldistance=1.1)
        
        st.pyplot(fig,width=1,height=2)


    def distrib_poids_taille(nb):
        res= df.sort_values('Overall',ascending=False).head(nb)
        res=res.rename(columns={'Height':'Taille (cm)','Weight':'Poids (kg)'})
        fig = plt.figure(figsize = (10, 5))
        res['Taille (cm)'].plot.kde()
        res['Poids (kg)'].plot.kde()
        plt.legend()        
        st.pyplot(fig,width=1,height=2)


    def show_values(axs, orient="v", space=.01):
        def _single(ax):
            if orient == "v":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                    value = '{:.1f}'.format(p.get_height())
                    ax.text(_x, _y, value, ha="center") 
            elif orient == "h":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                    value = '{:.0f}'.format(p.get_width())
                    ax.text(_x, _y, value, ha="left")
    
        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _single(ax)
        else:
            _single(axs)
            
    def equipe_les_plus_cheres(nb):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        a=pd.DataFrame(df.groupby(['Club']).sum()['Value (€)'].sort_values(ascending=False)).head(nb)
        a["Value (€)"] = a["Value (€)"].apply(lambda v: str(float(v) / 1000000))
        a["Value (€)"]=a["Value (€)"].astype('float')
        f, ax = plt.subplots(figsize=(17, 7))
        p=sns.barplot(x=a['Value (€)'],y=a.index,data=a, palette='Spectral')
        ax.set(xlabel='Valeur (€M) ', ylabel='Club')
        st.pyplot(show_values(p, "h", space=0))
        
    
    def equipe_les_meilleurs_espoirs(nb):
        
        a=pd.DataFrame(df.groupby(['Club']).sum()['Evolution'].sort_values(ascending=False)).head(nb)
        f, ax = plt.subplots(figsize=(17, 7))
        p=sns.barplot(x=a['Evolution'],y=a.index,data=a, palette='Spectral')
        ax.set(xlabel="Marge d'évolution", ylabel='Club')
        st.pyplot(show_values(p, "h", space=0))
        
    def equipe_les_plus_grands(nb):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        a=pd.DataFrame(df.groupby(['Club']).mean()['Height'].sort_values(ascending=False)).head(nb)
        f, ax = plt.subplots(figsize=(17, 7))
        p=sns.barplot(x=a['Height'],y=a.index,data=a, palette='Spectral')
        ax.set(xlabel="Taille (cm)", ylabel='Club')
        st.pyplot(show_values(p, "h", space=0))
    
    
    def equipe_les_plus_lourds(nb):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        a=pd.DataFrame(df.groupby(['Club']).mean()['Weight'].sort_values(ascending=False)).head(nb)
        f, ax = plt.subplots(figsize=(17, 7))
        p=sns.barplot(x=a['Weight'],y=a.index,data=a, palette='Spectral')
        ax.set(xlabel="Poids (kg)", ylabel='Club')
        st.pyplot(show_values(p, "h", space=0))
    
    
        

    is_scan = st.button("LANCER LA RECHERCHE")

    if viz == "Comparaison des classements des meilleurs joueurs par club":
            st.subheader("")
            st.subheader("Choisissez les clubs dont vous souhaitez comparer les meilleurs joueurs puis cliquez sur le bouton 'Lancer la recherche'")
            list_club=pd.DataFrame(df.groupby(['Club']).sum()['Value (€)'].sort_values(ascending=False)).index
            clubs = st.multiselect("Sélectionnez le(s) club(s) de votre choix:", options=list_club)
            if is_scan:
                res=df[df['Club'].isin(clubs)].sort_values('Overall',ascending=False)[['Photo','Name','Age','Overall',"Club Logo","Flag",'Best Position','Value (€)']]
                res["Value (€)"] = res["Value (€)"].apply(lambda v: str(float(v) / 1000000))
                res.index = np.arange(1, len(res) + 1)
                res.index.name='Classement'
                res=res.reset_index()
                create_table(res)

    if viz == "Comparaison des classements des meilleurs joueurs par pays":
            st.subheader("")
            st.subheader("Choisissez les pays dont vous souhaitez comparer les meilleurs joueurs puis cliquez sur le bouton 'Lancer la recherche'")
            list_pays=pd.DataFrame(df.groupby(['Nationality']).sum()['Value (€)'].sort_values(ascending=False)).index
            pays = st.multiselect("Sélectionnez le(s) pays de votre choix:", options=list_pays)
            if is_scan:
                res=df[df['Nationality'].isin(pays)].sort_values('Overall',ascending=False)[['Photo','Name','Age','Overall',"Club Logo","Flag",'Best Position','Value (€)']]
                res["Value (€)"] = res["Value (€)"].apply(lambda v: str(float(v) / 1000000))
                res.index = np.arange(1, len(res) + 1)
                res.index.name='Classement'
                res=res.reset_index()
                create_table(res)

    if viz == "Classement des joueurs avec les meilleurs potentiels d'évolution en 2022":
        st.subheader("Choisissez le nombre de joueurs avec les meilleurs potentiels d'évolution dans le classement puis cliquez sur le bouton 'Lancer la recherche'") 
        nb_joueurs = st.slider("Nombre de meilleurs espoirs : ", min_value=0, max_value=100,value=15)
        if is_scan:
            meilleurs_potentiels(nb_joueurs)

    if viz == "La répartition des meilleurs pieds des n meilleurs joueurs du monde":
        st.subheader("Choisissez le nombre de joueurs parmi les meilleurs du monde pour lesquels vous voulez voir la distribution des meilleurs pieds puis cliquez sur le bouton 'Lancer la recherche'") 
        nb_joueurs = st.slider("Nombre des meilleurs joueurs du monde sélectionnés : ", min_value=1, max_value=100,value=15)
        if is_scan:
            most1,most2,most3 = st.columns([1,1,1])
            with most2:            
                st.subheader('Parmi les ' +str(nb_joueurs) +' meilleurs joueurs du monde, voici la proportion des pieds forts de ces derniers')
                meilleur_pied(nb_joueurs)
            
    if viz == "Distribution des tailles et poids des n meilleurs joueurs du monde":
        st.subheader("Choisissez le nombre de joueurs parmi les meilleurs du monde pour lesquels vous voulez voir la distribution de la taille et du poids puis cliquez sur le bouton 'Lancer la recherche'") 
        nb_joueurs = st.slider("Nombre des meilleurs joueurs du monde sélectionnés : ", min_value=1, max_value=100,value=15)
        if is_scan:
            most1,most2,most3 = st.columns([1,3,1])
            with most2:
                st.subheader('Parmi les ' +str(nb_joueurs) +' meilleurs joueurs du monde, voici la distribution de leurs tailles et poids')
                distrib_poids_taille(nb_joueurs)
            
    if viz == "Classement des clubs ayant les effectifs les plus chers du monde":
        st.subheader("Choisissez le nombre de clubs que vous souhaitez voir dans le classement puis cliquez sur le bouton 'Lancer la recherche'") 
        nb_clubs = st.slider("Nombre de clubs sélectionnés : ", min_value=1, max_value=100,value=15)
        if is_scan:
            most1,most2,most3 = st.columns([1,5,1])
            with most2:
                st.subheader('Voici le classement des ' +str(nb_clubs) +' clubs ayant les effectifs les plus chers du monde')
                equipe_les_plus_cheres(nb_clubs)
            
    if viz == "Classement des clubs ayant les meilleurs espoirs du monde":
        st.subheader("Choisissez le nombre de clubs que vous souhaitez voir dans le classement puis cliquez sur le bouton 'Lancer la recherche'") 
        nb_clubs = st.slider("Nombre de clubs sélectionnés : ", min_value=1, max_value=100,value=15)
        if is_scan:
            most1,most2,most3 = st.columns([1,5,1])
            with most2:
                st.subheader('Voici le classement des ' +str(nb_clubs) +' clubs ayant les meilleurs espoirs')
                equipe_les_meilleurs_espoirs(nb_clubs)
                
    if viz == "Classement des clubs ayant les plus grands joueurs de taille en moyenne":
        st.subheader("Choisissez le nombre de clubs que vous souhaitez voir dans le classement puis cliquez sur le bouton 'Lancer la recherche'") 
        nb_clubs = st.slider("Nombre de clubs sélectionnés : ", min_value=1, max_value=100,value=15)
        if is_scan:
            most1,most2,most3 = st.columns([1,5,1])
            with most2:
                st.subheader('Voici le classement des ' +str(nb_clubs) +' clubs ayant les joueurs les plus grands de taille en moyenne')
                equipe_les_plus_grands(nb_clubs)
            
            
    if viz == "Classement des clubs ayant les joueurs les plus lourds en moyenne":
        st.subheader("Choisissez le nombre de clubs que vous souhaitez voir dans le classement puis cliquez sur le bouton 'Lancer la recherche'") 
        nb_clubs = st.slider("Nombre de clubs sélectionnés : ", min_value=1, max_value=100,value=15)
        if is_scan:
            most1,most2,most3 = st.columns([1,5,1])
            with most2:
                st.subheader('Voici le classement des ' +str(nb_clubs) +' clubs ayant les joueurs les plus lourds en moyenne')
                equipe_les_plus_lourds(nb_clubs)
            
            
    if is_scan:

        if viz == 'Classement du joueur le plus cher par pays en 2022':
             joueur_le_plus_cher_de_chaque_pays()
        elif viz =="Classement des 100 meilleurs joueurs du monde en 2022":
            x_meilleurs_joueurs_dumonde()
        elif viz == "Classement des 100 joueurs les plus chers du monde en 2022":
            x_plus_chers_joueurs_dumonde()
        elif viz == "Classement des pays ayant la plus grande valeur marchande":
            most1,most2,most3 = st.columns([1,6,1])
            with most2:
                viz_somme_des_valeurs_des_pays()             
        elif viz == "Situer les 75 meilleurs joueurs du monde sur une carte":
            most1,most2,most3 = st.columns([1,4,1])
            with most2:
                placer_sur_une_carte_les_joueurs()
            
    
        


@app.addapp(title='Radars')
def Radars():
    df=pd.read_csv('FIFA22_official_data.csv')
    df_fifa20=pd.read_csv('sofifa2020.csv')
    df["Name"]=df["Name"].apply(lambda name: unidecode(name))
    df=df[df['Overall']>76]
    df=df.sort_values('Overall',ascending=False)
    cols_to_keep=list(set(df.columns).intersection(df_fifa20.columns))
    index=0
    for i in ['Name', 'Age', 'Photo', 'Nationality',"Best Position",'Flag','Club','Club Logo','Jersey Number','Overall','Height','Weight']:
        cols_to_keep.insert(index,i)
        index+=1
    
    df=df[cols_to_keep]
    df['Club_cat']=df['Club'].astype('category').cat.codes
    df['Nationality_cat']=df['Nationality'].astype('category').cat.codes
    
    
    
    def value_to_float(x):
        if type(x) == float or type(x) == int:
            return x
        if 'K' in x:
            if len(x) > 1:
                return float(x.replace('K', '')) * 1000
            return 1000.0
        if 'M' in x:
            if len(x) > 1:
                return float(x.replace('M', '')) * 1000000
            return 1000000.0
        if 'B' in x:
            return float(x.replace('B', '')) * 1000000000
        return x
    
    df['Wage']=df['Wage'].str.replace('€', '').apply(value_to_float)
    df['Value']=df['Value'].str.replace('€', '').apply(value_to_float)
    
    df['Height']=df['Height'].str.replace('[dA-Za-z]', '').astype('int')
    df['Weight']=df['Weight'].str.replace('[dA-Za-z]', '').astype('int')
    df=df.rename(columns={"Value": "Value (€)", "Wage": "Wage (€)"})
    df['Value (€)']=df['Value (€)'].astype('int64')
    df['Wage (€)']=df['Wage (€)'].astype('int64')
    df=df[~df['Name'].str.contains(r'\d')]
    
    
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    
    color1=(0,152/255,158/255,0.8)
    color2=(180/255,215/255,141/255,0.8)
    color3=(255/255,230/255,122/255,0.8)
    color4=(246/255,172/255,26/255,0.8)
    color5=(216/255,34/255,13/255,0.8)
    colors=[color1,color2,color3,color4,color5]
    
    
    def radar_plot(player):
    
        player=player.reset_index()
    
        angles=np.linspace(0,2*np.pi,6,endpoint=False)
        angles=np.concatenate((angles, [0]))
    
        #Detect if GoalKeeper or not
        if player['GKDiving'].values[0]<30:
            attributes=player.loc[0,['Acceleration','Finishing','Agility','Dribbling','StandingTackle','Strength','BallControl']]
            labels=('ACCELERATION\n{:d}'.format(int(player.loc[0,'Acceleration'])),
                    'FINISHING\n{:d}\n'.format(int(player.loc[0,'Finishing'])),
                    'AGILITY\n{:d}\n'.format(int(player.loc[0,'Agility'])),
                    'DRIBBLING\n{:d}'.format(int(player.loc[0,'Dribbling'])),
                    '\nTACKLE\n{:d}'.format(int(player.loc[0,'StandingTackle'])),
                    '\nPHYSIC\n{:d}'.format(int(player.loc[0,'Strength'])),
                    'CONTROL\n{:d}'.format(int(player.loc[0,'BallControl'])))
        else:
            attributes=player.loc[0,['GKDiving','GKHandling','GKKicking','GKReflexes','Reactions','GKPositioning','GKDiving']]
            labels=('DIVING\n{:d}'.format(int(player.loc[0,'GKDiving'])),
                    'HANDLING\n{:d}\n'.format(int(player.loc[0,'GKHandling'])),
                    'KICKING\n{:d}\n'.format(int(player.loc[0,'GKKicking'])),
                    'REFLEXES\n{:d}'.format(int(player.loc[0,'GKReflexes'])),
                    '\nREACTIONS\n{:d}'.format(int(player.loc[0,'Reactions'])),
                    '\nPOSITION\n{:d}'.format(int(player.loc[0,'GKPositioning'])),
                    'DIVING\n{:d}'.format(int(player.loc[0,'GKDiving'])))
    
        fig=plt.figure(figsize=(3,3))
        rect0 = [0, 0, 1, 1]
        rect1 = [0.00005, 0.00005, 0.9999, 0.9999]
        rect2 = [0.1, 0.1, 0.8, 0.8]
        rect3 = [0.2, 0.2, 0.6, 0.6]
        rect4 = [0.3, 0.3, 0.4, 0.4]
        rect5 = [0.4, 0.4, 0.2, 0.2]
    
        rects=[rect1,rect2,rect3,rect4,rect5]
    
        for rect,col in zip(rects,colors):
            bk=plt.axes(rect,projection='polar')
            bk.fill(angles,[1,1,1,1,1,1,1],color=col)
            bk.set_ylim(0,1)
            bk.axis('off')
    
        rdp=plt.axes(rect0,projection='polar')
        rdp.plot(angles,attributes,color='#222222',linestyle="-",lw=2)
        rdp.fill(angles,attributes,color='#222222',alpha=0.3)
        rdp.set_thetagrids(angles/np.pi*180,labels,fontsize=13)
        rdp.set_title(player.Name[0],fontsize=20)
        rdp.patch.set_alpha(0)
        rdp.set_rticks([])
        rdp.set_ylim(0,100)
        #rdp.axis('off')
        plt.show()
    
    player_list = list(df["Name"].unique())
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    radar_plot(df[df['Name'].str.contains('Marquinhos')])
    radar_plot(df[df['Name'].str.contains('Neymar')])
    
    hide_streamlit_style = """
                <style>
                    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 500px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 500px;
            margin-left: -500px;
        }
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    
    st.title("&#9974; Bienvenue sur l'onglet Radars")
    st.subheader('')
    st.subheader('')
    st.subheader("Sur cet onglet, vous pourrez afficher les diagrammes de Kiviat de plusieurs joueurs.")
    st.subheader('')
    st.subheader("L'objectif est de comparer des joueurs selon leurs notes.")
    st.subheader('')
    
    
    players = st.multiselect("Joueur(s) sélectionné(s):", options=player_list, default=None)
    
    
    
    is_scan = st.button("LANCER LA RECHERCHE")
    
    
    
    if is_scan:
        if len(players)==1:
            st.title(
       "Voici le digramme du joueur sélectionné"
        )
        else : 
             st.title(
       "Voici les digrammes des joueurs sélectionnés"
        )   
        most1, most2,most3 = st.columns([1,1,1])
        with most2:
            for i in players:
    
                st.pyplot(radar_plot(df[df['Name'].str.contains(i)]),figsize=(6,6))
                st.subheader(
                  "----------------------------------------------------------------------------"
                )
    
            
        




#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
