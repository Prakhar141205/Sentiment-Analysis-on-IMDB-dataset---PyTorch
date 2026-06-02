import yaml
import os


def load_config_file(file_path):
    with open(file_path, 'r') as file:
        try:
            config_data = yaml.safe_load(file)
            print(config_data)
            return config_data
        except yaml.YAMLError as e:
            print(f"Error occured in loading config file: {e}")
            return None