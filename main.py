import pandas as pd
import os
import configparser
import argparse
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from predictor.LightGBMoptunamodel import LightGBM
from predictor.NeuralNetworkmodel import NeuralNetwork
from predictor.kNearestNeighbormodel import kNearestNeighbor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="sample.ini")
    parser.add_argument("--name", type=str, default="knnsample")
    args = parser.parse_args()
    return args

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
    config = parse_args()
    setting_dict = read_config(os.path.join("config",config.path),config.name)

    print("load config:",setting_dict)
    dir = setting_dict.pop("data_dir")
    train = pd.read_csv(f"{dir}train.csv",index_col="id")
    TEST = pd.read_csv(f"{dir}test.csv",index_col="id")

    x_train = train.drop("price_range",axis=1)
    y_train = train["price_range"]

    ppss = {
        "std": StandardScaler,
       "minmax": MinMaxScaler,
        "None": False,
    }

    pps_key = setting_dict.pop("pps")
    pps = ppss[pps_key]

    if pps:
        print("preprocessing:")
        pps = pps()
        x_train = pd.DataFrame(pps.fit_transform(x_train),index=x_train.index,columns=x_train.columns)
        TEST = pd.DataFrame(pps.transform(TEST),index=TEST.index,columns=TEST.columns)
        pd.concat([x_train,y_train],axis=1).to_csv(f"data/additive_data/{pps_key}train.csv")
        TEST.to_csv(f"data/additive_data/{pps_key}test.csv")


    if setting_dict.pop("load_pretraind")=="True":
        print("loading pretraind model")
        model = joblib.load(f"results/model/{setting_dict['model_name']}.model")
    else:
        print("training model")
        models = {
            "lgbm": LightGBM,
            "nn":   NeuralNetwork,
            "knn":  kNearestNeighbor
        }
        model = models[setting_dict.pop("model_key")]
        model = model(*setting_dict.values())
        model.fit(x_train,y_train)
        model.dump(config.name)

    print("predicting")
    y_predict = model.predict(TEST)
    y_predict.to_csv(f"results/submit/{config.name}predict.csv",header=False)
    print("done")

if __name__ == '__main__':
    main()