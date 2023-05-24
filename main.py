import pandas as pd
import configparser
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from predictor.LightGBMoptunamodel import LightGBM
from predictor.NeuralNetworkmodel import NeuralNetwork
from predictor.kNearestNeighbormodel import kNearestNeighbor


def read_config(path,name):
    config = configparser.ConfigParser()
    config.read(path)
    config_dict = dict(config[name])
    type_dict = {"int":int,"float":float,"str":str}
    for key,value in config_dict.items():
        type_, value = value.split(" ")
        config_dict[key] = type_dict[type_](value)
    return config_dict

def main():
    name = "lgbmsample1"
    setting_dict = read_config("config/sample.ini",name)


    train = pd.read_csv(setting_dict.pop("data_dir"),index_col="id")
    TEST = pd.read_csv("data/official_data/test.csv",index_col="id")

    x_train = train.drop("price_range",axis=1)
    y_train = train["price_range"]

    ppss = {
        "std": StandardScaler,
        "minmax": MinMaxScaler,
        "None": False
    }


    pps = ppss[setting_dict.pop("pps")]
    if pps:
        pps = pps()
        x_train = pd.DataFrame(pps.fit_transform(x_train),index=x_train.index,columns=x_train.columns)
        TEST = pd.DataFrame(pps.transform(TEST),index=TEST.index,columns=TEST.columns)

    models = {
        "lgbm": LightGBM,
        "nn":   NeuralNetwork,
        "knn":  kNearestNeighbor
    }


    model = models[setting_dict.pop("model_key")]
    model = model(*setting_dict.values())

    model.fit(x_train, y_train)

    model.dump()
    y_predict = model.predict(TEST)
    y_predict.to_csv(f"results/submit/{name}predict.csv",header=False)

if __name__ == '__main__':
    main()