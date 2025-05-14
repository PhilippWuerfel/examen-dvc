import json
import joblib

def read_json(file_name: str):
    with open(file_name) as file:
        return json.load(file)


def read_pk1_obj(file_name: str):
    return joblib.load(file_name)