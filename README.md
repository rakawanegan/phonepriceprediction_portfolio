# Phone Price Prediction Bigenner Cup

solution to promotion Intermediate  
https://signate.jp/competitions/750


## Table of Contents

- [Competition Description](#competition-description)
- [Dataset](#dataset)
- [Solution Approach](#solution-approach)
- [Results](#results)
- [Reproduction](#reproduction)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)


## Competition Description

This competition is open to beginners and registered participants.  
If your F1 macro score exceeds 0.462885, you will be recognized as an authorized intermediate participant.


## Dataset

| 0  | id            | int | Index (used as an identifier)                        |  
|----|---------------|-----|------------------------------------------------------|  
| 1  | battery_power | int | Total energy a battery can store in one charge (mAh) |  
| 2  | blue          | int | Bluetooth availability (1 if available)              |  
| 3  | clock_speed   | float | Clock speed                                        |  
| 4  | dual_sim      | int | Dual SIM support availability (1 if available)       |  
| 5  | fc            | int | Front camera megapixels                              |  
| 6  | four_g        | int | 4G support availability (1 if available)             |  
| 7  | int_memory    | int | Internal memory (GB)                                 |  
| 8  | m_dep         | float | Mobile depth (cm)                                  |  
| 9  | mobile_wt     | int | Weight                                               |  
| 10 | n_cores       | int | Number of cores                                      |  
| 11 | pc            | int | Primary camera megapixels                            |  
| 12 | px_height     | int | Pixel resolution height                              |  
| 13 | px_width      | int | Pixel resolution width                               |  
| 14 | ram           | int | Random Access Memory (MB)                            |  
| 15 | sc_h          | int | Screen height of the mobile (cm)                     |  
| 16 | sc_w          | int | Screen width of the mobile (cm)                      |  
| 17 | talk_time     | int | Continuous talk time                                 |  
| 18 | three_g       | int | 3G support availability (1 if available)             |  
| 19 | touch_screen  | int | Touch screen availability                            |  
| 20 | wifi          | int | Wi-Fi availability (1 if available)                  |  
| 21 | price_range   | int | Price range:target 0 (low cost), 3 (very high cost)  |


## Solution Approach

### preprocess
- minmaxscaler(sk-learn)
- standardscaler(sk-learn)

### model
- LightGBM
- NeuralNetwork(Dence)
- KNearestNeighbor

Describe how others can use your code or model for their own predictions. Provide instructions on setting up the environment, installing dependencies, and running the code.
## Results

authorized intermediate participant.


## Usage

```
python3 run.py --path [.ini file path] --name [config name]
```
You have the flexibility to configure each model using the following format: [key] = [type] [value].  

### Common configurations:  
datadir(str): Directory path for the data  
model_key(str): Name of the model -->[ nn, knn, lgbm ]  
pps(str): Preprocessing method -->[ std, minmax, None ]  
load_pretrained(str): Set to either True or False -->[ True, False ]  
model_name(str): Name of the pre-trained model to load  

### Neural Network configuration:
input_shape(int): Shape of the input data  
L(int): Number of neurons in the first layer  
epoch(int): Maximum number of epochs  
batch(int): Batch size  
random_state(int): Random seed  

### KNN configuration:
k(int): Number of nearest points to consider  

### LGBM configuration:
None

Feel free to adjust these configurations based on your specific needs and model requirements.

## License

free.