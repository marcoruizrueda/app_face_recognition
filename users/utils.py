import pandas as pd

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
    
