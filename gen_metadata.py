import pandas as pd
import os

# BASE_FOLDER = './data/'
# path_icd = os.path.join(BASE_FOLDER, 'ICD_First_Admit_drop.csv') # remove the ICD codes that do not correspond to PheCodes
# path_pres = os.path.join(BASE_FOLDER, 'PRESCRIPTION_First_Admit.csv')
# path_cpt = os.path.join(BASE_FOLDER, 'CPT_First_Admit.csv')
# path_drg = os.path.join(BASE_FOLDER, 'DRGCODE_First_Admit.csv')
# path_lab = os.path.join(BASE_FOLDER, 'LAB_First_Admit.csv')
# path_note = os.path.join(BASE_FOLDER, 'NOTE_First_Admit.csv')

BASE_FOLDER = './data_full_admit/'
path_icd = os.path.join(BASE_FOLDER, 'ICD_Full_Admit_drop.csv') # remove the ICD codes that do not correspond to PheCodes
path_pres = os.path.join(BASE_FOLDER, 'Pres_Full_Admit.csv')
path_cpt = os.path.join(BASE_FOLDER, 'CPT_Full_Admit.csv')
path_drg = os.path.join(BASE_FOLDER, 'DRGCODE_Full_Admit.csv')
path_lab = os.path.join(BASE_FOLDER, 'Lab_Full_Admit.csv')
path_note = os.path.join(BASE_FOLDER, 'Note_Full_Admit.csv')
path_dict = {'icd': path_icd, 'pres': path_pres, 'cpt': path_cpt, 'drg': path_drg, 'lab': path_lab, 'note': path_note}
column_dict = {'icd': 'ICD9_CODE', 'pres': 'COMPOUND_ID', 'cpt': 'ICD9_CODE', 'drg': 'COMPOUND_ID', 'lab': 'LABEL',
               'note': 'TERM'}  # which column is defined as word for each modality

df = pd.DataFrame([path_dict, column_dict])
df['index'] = ['path', 'word_column']
df.set_index('index', inplace=True)
print(df.transpose())
df.transpose().to_csv(os.path.join(BASE_FOLDER, 'metadata.csv'), index_label='index')
