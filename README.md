
# Description

In this project, after creating a linear regression model using tensorflow or pytorch among <a href="https://www.tensorflow.org/tutorials/keras/regression">the basic regression contents of the tensorflow tutorial</a>, the fuel efficiency prediction service is provided as a Rest API using Flask.

# Requirements
* CUDA Version: 11.8, if possible
```bash
pip install -r requirements.txt
```

# How to Build a Model

```bash
python build_tf.py
```
or
```bash
python build_pt.py
````

# How to Run
```bash
python App_tf.py
```
or
```bash
python App_pt.py
```

# Request
```bash
curl -X POST http://localhost:5000/predict -d '{"features": [4,140.0,86.0,2790.0,15.6,82,0,0,1]}'