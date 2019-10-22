#-*- coding:utf-8 -*-
import pandas as pd
import glob
import os
from facial_recognizers import config as cfg

def read_txtUsers(person_name):
    try:
        data = pd.read_csv('/home/marco/face_recognizer/app_face_recognition/users/users.csv',
                          na_values = ['no info', '.'])
        data.columns = ['Name', 'Gender', 'Position', 'Project', 'Advisor']
        
        gender = data.loc[data['Name'] == person_name, ['Gender']].values[0][0]
        position = data.loc[data['Name'] == person_name, ['Position']].values[0][0]
        project = data.loc[data['Name'] == person_name, ['Project']].values[0][0]
        advisor = data.loc[data['Name'] == person_name, ['Advisor']].values[0][0]
        return gender, position, project, advisor
    except Exception as e: 
        #print('Error at index {}: {!r}'.format(i, row))
        #print(e)
        return '--', '--', '--', '--'
'''
def get_noRepeatedFiles():
    files = [f for f in glob.glob(cfg._C.PATH_DATASET + "**/*.jpg", recursive=False)]

    dires = []
    image_list = []
    for f in files:
        dirname = os.path.basename(os.path.dirname(f))
        if dirname in dires:
            continue
        image_list.append(f)
        dires.append(dirname)
    return dires, image_list
    '''