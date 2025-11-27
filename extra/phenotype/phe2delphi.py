"""
UK Biobank Phenotype Data Processor for Delphi Model

This script processes UK Biobank (UKB) phenotype data and converts it into a format 
suitable for the Delphi model. It transforms medical phenotype data into binary format 
for machine learning analysis of medical event sequences.

Key Components:
1. Data Mapping Setup:
   - Maps UKB field codes to ICD-10 medical codes using a mapping file
   - Creates a label dictionary for converting medical codes to numeric labels
   - Defines special fields like sex, birth year/month, death dates, and cancer information

2. Data Processing Pipeline:
   - Processes large UKB dataset in chunks (1000 rows at a time)
   - Demographics: Extracts sex (converts 0/1 to 1/2) and date of birth
   - Date Conversion: Converts all dates to "days since birth" for temporal modeling
   - Medical Events: Processes ICD-10 diagnosis codes and their occurrence dates
   - Cancer Data: Handles up to 21 cancer diagnoses per person with dates and types
   - Lifestyle Factors:
     * BMI categories (3=normal, 4=overweight >22, 5=obese >28)
     * Smoking status (6=never, 7=previous, 8=current)
     * Alcohol consumption (9=daily+, 10=occasional, 11=never)

3. Output Format:
   Creates three binary files with records: [patient_id, days_since_birth, medical_code_label]
   - real.bin - Complete dataset
   - real_train.bin - Training set (80% of patients)
   - real_val.bin - Validation set (20% of patients)

The final output represents medical events as timestamped sequences per patient, where 
each event is encoded as a numeric label representing different medical conditions, 
procedures, or lifestyle factors. This format enables temporal sequence modeling for 
predicting future medical events or outcomes.
"""

#%%
import pandas as pd
import tqdm
import numpy as np
import itertools


dir0 = "/mnt/d"
labels_file  = f"{dir0}/data/ukb/phe/delphi/labels.csv"
ukb_field_to_icd10_map_file = f"{dir0}/data/ukb/phe/delphi/icd10_codes_mod.tsv"
ubk_basket_tab_file = f"{dir0}/data/ukb/phe/rap/raw/pheno.tsv.gz" 
train_proportion = 0.8
output_prefix = 'real'


#%% 
# Read icd10 mapping file and defined index label link
icdict ={}
icdcodes = []
with open(ukb_field_to_icd10_map_file,'r') as f:
    for l in f:
        lvals=l.strip().split()
        icdict[lvals[0]]=lvals[5]
        icdcodes.append(lvals[5])

i = -1
label_dict = {}
with open(labels_file,'r') as f:
    for l in f:
        label_dict[l.strip().split(' ')[0]]=i
        i += 1

# hard coded sex and dob
icdict['p31'] = "sex"
icdict['p34'] = "YEAR"
icdict['p52'] = "MONTH"
icdict['p40000_i0'] = "Death" # ther is also p40000_i1

# cancer fields
for j in range(21):
    icdict['p40005_i'+str(j)] = "cancer_date_"+str(j)
    icdict['p40006_i'+str(j)] = "cancer_type_"+str(j)

icdict['p53_i0'] = "assessment_date"
icdict['p21001_i0']="BMI"
icdict['p1239_i0']="smoking"
icdict['p1558_i0']="alcohol"

len_icd = len(icdcodes)
icdcodes.extend(['Death','assessment_date']+['cancer_date_'+str(j) for j in range(21)])
# 🏮 Predefine the true date columns to parse safely
date_cols = ['Death', 'assessment_date'] + [f'cancer_date_{j}' for j in range(21)]


#%% 
# Read ukb basket file in chunks, select icd10 code occurance and dates, format for delphi
data_list = []
ukb_iterator = pd.read_csv(ubk_basket_tab_file, sep="\t", chunksize=1000, index_col=0)
first = next(ukb_iterator)                 # grab first chunk (DataFrame)
print(first.head(10))                  # head
print("cols:", first.columns.tolist()) # columns


#%% 
# 🏮 include the first chunk in processing (originally skipped)
for _, dd in tqdm.tqdm(enumerate(itertools.chain([first], ukb_iterator))):
    dd = dd.rename(columns=icdict)
    dd.dropna(subset=['sex'], inplace=True)
    dd['sex'] += 1
    dd = dd[[col for col in dd.columns if not col.startswith('p')]]

    # 🏮Robust DOB from YEAR/MONTH (blanks -> NaT, no exception)
    dd['dob'] =  pd.to_datetime(dd[['YEAR', 'MONTH']].assign(DAY=1), errors='coerce') 

    # 🏮Parse only actual date columns present; blanks -> NaT; then convert to days since dob 
    present_date_cols = [c for c in date_cols if c in dd.columns]
    dd[present_date_cols] = dd[present_date_cols].apply(pd.to_datetime, format="%Y-%m-%d", errors="coerce")
    dd[present_date_cols] = dd[present_date_cols].sub(dd['dob'], axis=0)  # I CHANGED HERE
    dd[present_date_cols] = dd[present_date_cols].apply(lambda x: x.dt.days)  # I CHANGED HERE

    # 🏮 Before casting to int, coerce any string-dates to days since dob on-the-fly.
    for col in icdcodes[:len_icd+1]:
        if col not in dd.columns:
            continue
        s = dd[col]

        # If dtype is object and looks like date strings, convert now.
        if s.dtype == object and s.dropna().astype(str).str.contains(r"\d{4}-\d{2}-\d{2}", regex=True).any():
            s_dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce") 
            s = (s_dt - dd['dob']).dt.days 

        X = s.dropna().reset_index().to_numpy()
        # 🏮 X has columns: [eid, value]; coerce value to int safely
        X[:, 1] = pd.to_numeric(X[:, 1], errors="coerce") 
        X = X[~np.isnan(X[:, 1])] 
        X = X.astype(int) 
        data_list.append(np.hstack((X, label_dict[col]*np.ones([X.shape[0],1], X.dtype))))

    X = dd['sex'].reset_index().to_numpy().astype(int)
    data_list.append(np.c_[X[:,0],np.zeros(X.shape[0]),X[:,1]].astype(int))
     
    for j in range(21):
        dd_cancer = dd[['cancer_date_'+str(j),'cancer_type_'+str(j)]].dropna().reset_index()
        if not dd_cancer.empty:
            dd_cancer['cancer'] = dd_cancer['cancer_type_'+str(j)].str.slice(0,3)
            dd_cancer['cancer_label'] = dd_cancer["cancer"].map(label_dict)
            data_list.append(dd_cancer[['eid','cancer_date_'+str(j),'cancer_label']].dropna().astype(int).to_numpy())
       
    dd_bmi = dd[['assessment_date','BMI']].dropna().reset_index()
    dd_bmi['bmi_status'] = np.where(dd_bmi['BMI']>28,5,np.where(dd_bmi.BMI>22,4,3))
    data_list.append(dd_bmi[['eid','assessment_date','bmi_status']].astype(int).to_numpy())
    
    dd_sm = dd[['assessment_date','smoking']].dropna().reset_index()
    dd_sm = dd_sm[dd_sm['smoking']!=-3]
    dd_sm['smoking_status'] = np.where(dd_sm['smoking']==1,8,np.where(dd_sm.smoking==2,7,6))
    data_list.append(dd_sm[['eid','assessment_date','smoking_status']].astype(int).to_numpy())
    
    dd_al = dd[['assessment_date','alcohol']].dropna().reset_index()
    dd_al = dd_al[dd_al['alcohol']!=-3]
    dd_al['alcohol_status'] = np.where(dd_al['alcohol']==1,11,np.where(dd_al.alcohol < 4,10,9))
    data_list.append(dd_al[['eid','assessment_date','alcohol_status']].astype(int).to_numpy())

  
#%% 
# reformat, split train and val and output to delphi format
data= np.vstack(data_list)
data = data[np.lexsort((data[:,1], data[:,2]==data[:,2].max(), data[:,0]))]
data = data[data[:,1]>=0]
data = pd.DataFrame(data).drop_duplicates([0,2]).values
data = data.astype(np.uint32)
data.tofile(output_prefix + '.bin')

ids = sorted(set(data[:,0]))  # I CHANGED HERE (fix .sort() returning None)
train_val_split = data[:,0] <= ids[int(len(ids)*train_proportion)]
data[train_val_split].tofile(output_prefix + '_train.bin')
data[~train_val_split].tofile(output_prefix + '_val.bin')
