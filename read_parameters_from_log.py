def find_value_from_log(key, log_data):
    """
    Finds the value of the given key in the log data

    Args:
        key (str): 

    Returns:
        str: value
    """
    start_index = log_data.index(key)
    length_of_key = len(key)
    if key == 'd_s_layers' or key == 'g_s_layers':
      temp = start_index + log_data[start_index:].index(",")
      temp = temp + log_data[temp:].index("'")
      temp = temp + log_data[temp:].index(",")
      end_index = temp 
    else:
      end_index = start_index + log_data[start_index:].index(',')
    value = log_data[start_index+length_of_key+1:end_index]
    return value
