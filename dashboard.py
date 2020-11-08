import pandas as pd
import streamlit as st

from dataclasses import dataclass
import dataclasses
import collections
import time

import requests
import urllib
import json as json

import textdistance as td
import pickle

API_BASE_URL = "https://apis.datos.gob.ar/georef/api/"


@st.cache
def get_data_arsat():
    arsat_pcfo_path = "./data/20201009_puntos_de_conexion_a_la_red_de-fibra_optica.csv"
    return pd.read_csv(arsat_pcfo_path, sep=";", encoding="latin-1")


@dataclass
class QueryConfig:
    nombre: str
    provincia: str
    departamento: str
    aplanar: bool = True
    campos: str = "estandar"
    exacto: bool = True


@dataclass
class Localidad:
    lat: float
    long: float

    def __str__(self):
        return f"LAT:{self.lat} LONG:{self:long}"

    def __repr__(self):
        return f"LAT:{self.lat} LONG:{self:long}"


def load_georeverse(df: pd.DataFrame):
    """
    Esta función carga desde el motor de elastic
    El problema que la cuota del que ofrecen está limitada,
    tendríamos que crear el entorno en
    """
    all_localidades = df.to_dict(orient="records")
    data = pd.DataFrame(columns=["lat", "long"])

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def parse_localidad(r):
        if len(r["localidades"]):
            # print(r["localidades"][0])
            return {
                "lat": r["localidades"][0]["centroide_lat"],
                "long": r["localidades"][0]["centroide_lon"],
            }
        else:
            return {"lat": None, "long": None}

    for localidades in chunks(all_localidades, 20):
        payload = {
            "localidades": [
                dataclasses.asdict(
                    QueryConfig(
                        nombre=s["Localidad"],
                        departamento=s["Departamento"],
                        provincia=s["Provincia"],
                    )
                )
                for s in localidades
            ]
        }

        ret = requests.post(
            "https://apis.datos.gob.ar/georef/api/localidades",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

        if not ret.ok:
            st.error(ret.text)
            return

        response_data = list(map(parse_localidad, ret.json()["resultados"]))
        data = data.append(pd.DataFrame.from_records(response_data))
        time.sleep(2)

    return data


@st.cache
def load_localidades():
    registros = pd.read_json("./data/localidades.ndjson", lines=True)
    registros = registros.drop(index=[0])
    registros = registros[["nombre", "geometria", "provincia"]]

    coordinates = registros.geometria.apply(pd.Series).coordinates
    registros["lon"] = coordinates.apply(pd.Series).apply(lambda r: r[0][0], axis=1)
    registros["lat"] = coordinates.apply(pd.Series).apply(lambda r: r[0][1], axis=1)
    registros = registros.drop(columns=["geometria"])
    del coordinates

    registros["provincia"] = registros.provincia.apply(pd.Series).nombre

    return registros


@st.cache
def load_lat_long_from_csv(df):
    """

    dataset: https://github.com/datosgobar/georef-ar-api/blob/master/config/georef.example.cfg
    """
    # all_localidades = df.to_dict(orient="records")
    data = pd.DataFrame(columns=["lat", "lon"])

    registros = load_localidades()

    localidades = df.Localidad.str.upper()
    provincias = df.Provincia

    # bar = st.progress(0.0)
    # step = 1 / localidades.shape[0]
    # progress = 0

    for l, p in zip(localidades, provincias):
        aux = registros[(registros.nombre == l)]
        # progress += step
        # bar.progress(progress)

        if aux.shape[0] == 1:
            data = data.append(aux[["lat", "lon"]])
        elif aux.shape[0] > 1:
            province_id_max = aux.provincia.apply(
                lambda prov: td.hamming(prov, p)
            ).argmin()
            # print(aux)
            # print(
            #     province_id_max,
            #     p,
            # )
            data = data.append(aux.iloc[[province_id_max]].loc[:, ["lat", "lon"]])
        elif aux.shape[0] == 0:
            data = data.append(pd.DataFrame.from_records([{"lat": None, "lon": None}]))

    return data.set_index(df.index)


st.sidebar.title("Tambien podemos hacer otras cosas")
st.sidebar.markdown("Podríamos elegir que solo muestre atributos de una provincia!")
# más abajo lo coloco... interesante!

st.title("Armando un Dashboard con streamlit")
with st.beta_expander("Mapas and Tablas"):

    st.header("Puntos de conexión a la Red Federal de Fibra Óptica")
    st.markdown(
        "La tabla muestra los puntos de acceso la Red Federal de Fibra Óptica que se encuentran en servicio, "
        + "junto a otros datos como la población de la localidad.\n\n"
        + "Información en el siguiente [link](https://datos.gob.ar/dataset/arsat-puntos-conexion-refefo/archivo/arsat_26a7adaa-cadd-43dc-b884-b77fec7aaa23)"
    )

    df = get_data_arsat()
    st.table(df.sample(10))

    st.header("Georeverse")
    st.markdown(
        "Intentamos recuperarlo desde [aquí](https://datosgobar.github.io/georef-ar-api/open-api), pero dada la cuota baja que tiene, nos bajamos el dataset con la geolocalización"
        + ". Vemos en ejemplo de los datos luegos de transformarlos y unirlos a nuestro dataframe"
    )

    lat_long_df = load_lat_long_from_csv(df)
    df = pd.concat([df, lat_long_df], axis=1)
    st.table(df[["Localidad", "Departamento", "Provincia", "lat", "lon"]].sample(5))

    st.header("Mapa")

    # más arriba coloqué una nota en el sidebar
    sidebar_selectbox = st.sidebar.selectbox(
        "Provincia", ["Todas", *df.Provincia.unique()]
    )

    df_show_in_map = df[df.lat.notna()]
    st.write(sidebar_selectbox)
    if sidebar_selectbox != "Todas":
        df_show_in_map = df_show_in_map[
            (df_show_in_map.Provincia == sidebar_selectbox)
            & (df_show_in_map.Provincia.notna())
        ]
    st.map(df_show_in_map)

    st.header("Exploración")
    st.markdown("Podríamos obtener algunas estadísticas")
    st.info("Esto marca cuantas conexiones por provincia")
    st.bar_chart(df.Provincia.value_counts())

    import plotly.graph_objects as go

    with st.beta_container():

        col1, col2, col3 = st.beta_columns(3)

        with col1:
            fig = go.Figure(
                go.Indicator(
                    mode="number",
                    value=df.Departamento.nunique(),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Departamentos"},
                )
            )

            st.plotly_chart(fig)

        with col3:
            fig = go.Figure(
                go.Indicator(
                    mode="number",
                    value=df.Localidad.nunique(),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Localidades"},
                )
            )

            st.plotly_chart(fig)


with st.beta_expander("Deployment Modelo"):
    st.header("Ejecutando nuestro modelo")
    st.markdown(
        "Vamos a subir el archivo de nuestro modelo serializado y ejecutarlo con data aquí."
        + "Parece una manera práctica de desplegar un modelo, aunque no la mejor"
    )

    model_serialized = open(
        "./data/breast_tree_model.pickle", "rb"
    )  # st.file_uploader("SUBÍ TU MODELO")
    if model_serialized is not None:
        model = pickle.load(model_serialized)
        st.write(model)

    feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
       'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
    st.bar_chart(pd.Series(model.feature_importances_, index=feature_names))
    
    col1, col2 = st.beta_columns(2)
    with col1:
        radius = st.number_input(
            "radius", min_value=0.0, max_value=30.0, key="input_radius", value=14.42
        )
        texture = st.number_input(
            "texture", min_value=0.0, max_value=50.0, key="input_texture", value=16.54
        )
        perimeter = st.number_input(
            "perimeter",
            min_value=0.0,
            max_value=200.0,
            key="input_perimeter",
            value=95.0,
        )
        area = st.number_input(
            "area", min_value=0.0, max_value=2800.0, key="input_area", value=300.0
        )
        smoothness = st.number_input(
            "smoothness",
            min_value=0.0,
            max_value=1.0,
            key="input_smoothness",
            value=0.09,
        )

    with col2:
        compactness = st.number_input(
            "compactness",
            min_value=0.0,
            max_value=1.0,
            key="input_compactness",
            value=0.12,
        )
        concavity = st.number_input(
            "concavity", min_value=0.0, max_value=1.0, key="input_concavity", value=0.1
        )
        concave_points = st.number_input(
            "concave_points",
            min_value=0.0,
            max_value=1.0,
            key="input_concave_points",
            value=0.05,
        )
        symmetry = st.number_input(
            "symmetry", min_value=0.0, max_value=1.0, key="input_symmetry", value=0.20
        )
        fractal_dimension = st.number_input(
            "fractal_dimension",
            min_value=0.0,
            max_value=1.0,
            key="input_fractal_dimension",
            value=0.05,
        )

    execute_model = st.button("Evaluar", key="model_button")

    output = st.empty()
    
    if execute_model:
        st.info("resultado de la evaluación")
        features = [
            radius,
            texture,
            perimeter,
            area,
            smoothness,
            compactness,
            concavity,
            concave_points,
            symmetry,
            fractal_dimension,
        ]

        def predict(model, features: list):
            cols = [*["proba_" + c for c in model.classes_], "target"]
            results = pd.DataFrame(columns=cols)
            probas = model.predict_proba(features)
            results.iloc[:, 0] = probas[:, 0]
            results.iloc[:, 1] = probas[:, 1]
            results.target = model.predict(features)
            st.table(results)
            return results

        print(predict(model, [features]))
