
##### REQUIRES THE DATAFRAME FOLDER TO BE NAMED 'Cohorts', WHICH INCLUDES ALL PRECOMPUTED DATAFRAMES #####
import fiber
from fiber.cohort import Cohort
from fiber.condition import Patient, MRNs
from fiber.condition import Diagnosis
from fiber.condition import Measurement, Encounter, Drug
from fiber.storage import yaml as fiberyaml
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os
from functools import reduce


# In[4]:


Case_filtered_15_540 = pq.read_table('Cohorts/Case/Case_filtered_15_540.parquet').to_pandas()
Control_filtered_15_540 = pq.read_table('Cohorts/Control/Control_filtered_15_540.parquet').to_pandas()

#Cases_Blood_Pressure = pq.read_table("Cohorts/Case/Cases_Blood_Pressure.parquet").to_pandas()


# In[5]:


def df_to_cohort(df):
    mrns = list(df.index.values)
    condition = MRNs(mrns)
    return Cohort(condition)


# In[6]:


### BP Cases ###


# In[7]:


####CASE WEIGHT######


# In[10]:


cohort = df_to_cohort(Control_filtered_15_540)


# In[11]:


conditions = (Measurement("%Blood Pressure%"))


# In[ ]:



# In[ ]:


Cases_Blood_Pressure = []

Case_MRNs = list(Control_filtered_15_540.index.values)

for limit in range(300000, len(Case_MRNs), 50000):
    print("Begin of iteration: " +  str(limit))
    temp = Case_MRNs[limit:(limit+50000)]
    p_condition = MRNs(temp) #how to create cohort from dataframe
    cohort = Cohort(p_condition)

    enc = cohort.get(conditions)
    enc.to_parquet("Control_BP_" + str(limit) + ".parquet")

    print("End of iteration: " +  str(limit))


# In[10]:



# In[ ]:


