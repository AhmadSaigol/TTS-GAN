
import json

def load_from_json(path_to_json):
    """
    Load a json file

    Parameters:
        path_to_json: path to json file
    
    Returns:
        dictionary with contents of the json
    """
    with open(path_to_json +".json", 'r') as f:
        dic = json.load(f)
    
    return dic


def save_to_json(dic, path_to_results):
    """
    Save a dictionary to a location

    Parameters:
        dic: dictionary to be saved
        path_to_results: path where to save the dictionary

    """
    with open(path_to_results+".json", "w") as fp:
        json.dump(dic, fp, indent=4)