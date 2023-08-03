import os, re
from bs4 import BeautifulSoup


def minor_preprocess(file):
    with open(file, 'rb') as f:
        lines = f.readlines()
        utf8_safe_lines = [line.decode('utf-8', 'ignore') for line in lines]
        xml_safe_lines = [re.sub(r'&#\d*;', '', line) for line in utf8_safe_lines]  # Get rid of problematic strings
        no_newlines = [line.replace('\n', ' ') for line in xml_safe_lines]
    f.close()

    return ''.join(no_newlines)


def compile_data(datapath='./dataset'):
    dataset = []
    for file in os.listdir(datapath):
        if file.endswith('.sgm'):
            preprocessed_data = minor_preprocess(datapath + '/' + file)
            records = [record + '</REUTERS>' for record in preprocessed_data.split('</REUTERS>') if
                       record]  # Retain all original formatting
            dataset.extend(records)
    return dataset


def compile_dictionary(data):
    data_dict = {
        'REUTERS TOPICS': '',  # Initialize each key with some value. Important for the try/except block.
        'LEWISSPLIT': '',  # train / test sample
        'TOPICS': 'none',  # Consider empty topics as a new category, 'none'. See next section.
        'TITLE': '',
        'BODY': '',
    }

    # Grab the Reuters Topics between the following tags
    start = data.find('<REUTERS TOPICS="') + len('<REUTERS TOPICS="')
    end = data.find('" LEWISSPLIT=')
    data_dict['REUTERS TOPICS'] = data[start:end]
    start = data.find('LEWISSPLIT="') + len('LEWISSPLIT="')
    end = data.find('" CGISPLIT=')
    data_dict['LEWISSPLIT'] = data[start:end]

    soup = BeautifulSoup(data, 'xml')

    # Use a try/except block to grab Topics, Title, and Body in case they are empty
    # If empty, the default value remains unchanged
    try:
        if soup.TOPICS.contents:
            data_dict['TOPICS'] = soup.TOPICS.contents[:]
            data_dict['TOPICS'] = [str(data).replace('<D>', '') for data in data_dict['TOPICS']]
            data_dict['TOPICS'] = [str(data).replace('</D>', '') for data in data_dict['TOPICS']]

        if soup.TITLE.contents:
            data_dict['TITLE'] = soup.TITLE.contents[0]

        if soup.BODY.contents:
            body = soup.BODY.contents[0]
            data_dict['BODY'] = soup.BODY.contents[0]
    except AttributeError:
        pass

    return data_dict
