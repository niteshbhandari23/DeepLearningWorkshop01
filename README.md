
# Binary Classification with Neural Networks on the Census Income Dataset

### Name : NITESH BHANDARI K
### Reg.no : 212225240101 


 

This repository contains a PyTorch implementation of a binary classification model that predicts whether an individual earns more than $50,000 annually based on the Census Income Dataset.

## Dataset
The dataset used is the Census Income Dataset, which contains approximately 30,000 entries. 
[Dataset Link](https://drive.google.com/file/d/1ay5vOv2YiOwjKIWnXFT6sptW0gqew0oa/view?usp=sharing)


## Overview
This repository contains a complete PyTorch implementation of a binary classification model designed to predict whether an individual earns more than $50,000 annually. The project prepares the Census Income Dataset (separating categorical and continuous columns), builds a custom `TabularModel` Neural Network with an embedding layer and batch normalization, trains for 300 epochs using the Adam optimizer, and finally evaluates the test accuracy.

##  Code
Here is the complete Python code used for the project, spanning data preparation, the custom `TabularModel` architecture, the training loop, model evaluation, and inference on a custom user input.
21
```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gdown
import os
import warnings
warnings.filterwarnings('ignore')

# 1. Download the dataset from Google Drive for Google Colab
file_id = '1ay5vOv2YiOwjKIWnXFT6sptW0gqew0oa'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'income.csv'

if not os.path.exists(output):
    print("Downloading dataset...")
    gdown.download(url, output, quiet=False)

df = pd.read_csv(output)

# 2. Identify categorical, continuous, and label columns
target_col = df.columns[-1]

if df[target_col].dtype == 'object':
    df[target_col] = LabelEncoder().fit_transform(df[target_col])

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cont_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

if target_col in cont_cols:
    cont_cols.remove(target_col)
if target_col in cat_cols:
    cat_cols.remove(target_col)

# 3. Convert categorical columns to category type and get their codes
for cat in cat_cols:
    df[cat] = df[cat].astype('category')

cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)
conts = np.stack([df[col].values for col in cont_cols], 1)

cats = torch.tensor(cats, dtype=torch.int64)
conts = torch.tensor(conts, dtype=torch.float)
labels = torch.tensor(df[target_col].values, dtype=torch.long)

# 4. Split the dataset into training (25,000) and testing (5,000) sets
b = 25000 # Training samples
t = 5000  # Testing samples

cat_train = cats[:b]
cat_test = cats[b:b+t]
cont_train = conts[:b]
cont_test = conts[b:b+t]
y_train = labels[:b]
y_test = labels[b:b+t]

# 5. Define TabularModel class
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont
        
        layerlist = []
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        
        self.layers = nn.Sequential(*layerlist)
        
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        
        x = self.layers(x)
        return x

torch.manual_seed(42)
model = TabularModel(emb_szs, conts.shape[1], 2, [50], p=0.4)

# 6. Define criterion and optimizer, and Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(cat_train, cont_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 30 == 0:
        print(f'Epoch {i:3} | Loss: {loss.item():.4f}')

# 7. Evaluate the model
model.eval()
with torch.no_grad():
    y_val = model(cat_test, cont_test)
    loss = criterion(y_val, y_test)
    print(f'\nTest Loss: {loss.item():.4f}')
    
preds = torch.max(y_val, 1)[1]
correct = (preds == y_test).sum()
accuracy = 100 * correct / len(y_test)
print(f'Test Accuracy: {accuracy.item():.2f}%')

# 8. Function to predict new data
def predict_income(model, new_data_dict, df, cat_cols, cont_cols):
    model.eval()
    
    cat_seq = []
    for col in cat_cols:
        val = new_data_dict.get(col, df[col].mode()[0])
        cats = df[col].cat.categories
        if val in cats:
            cat_seq.append(cats.get_loc(val))
        else:
            cat_seq.append(0)
            
    cont_seq = []
    for col in cont_cols:
        val = new_data_dict.get(col, df[col].median())
        cont_seq.append(val)
        
    cat_tensor = torch.tensor([cat_seq], dtype=torch.int64)
    cont_tensor = torch.tensor([cont_seq], dtype=torch.float)
    
    with torch.no_grad():
        z = model(cat_tensor, cont_tensor)
        pred = z.argmax(dim=1).item()
        
    return ">50K" if pred == 1 else "<=50K"

sample_input = {
    'marital-status': 'Married-civ-spouse',
    'education': 'Bachelors',
    'hours-per-week': 45.0,
    'age': 35,
    'sex': 'Male'
}

prediction = predict_income(model, sample_input, df, cat_cols, cont_cols)
print(f"\nPredicted Income for the given input: {prediction}")
```

## Execution Output
When running the model for 300 epochs using the Adam Optimizer (`lr=0.001`) and `CrossEntropyLoss`, the expected execution output is similar to the following:
```
Downloading dataset...
Dataset shape: (32561, 15)
```


<img width="1131" height="254" alt="image" src="https://github.com/user-attachments/assets/de482c95-6d36-40f6-801d-d4f42a42ca22" />

### Epoch
```
Epoch  30 | Loss: 0.4351
Epoch  60 | Loss: 0.3812
Epoch  90 | Loss: 0.3541
Epoch 120 | Loss: 0.3318
Epoch 150 | Loss: 0.3155
Epoch 180 | Loss: 0.3012
Epoch 210 | Loss: 0.2988
Epoch 240 | Loss: 0.2941
Epoch 270 | Loss: 0.2922
Epoch 300 | Loss: 0.2915
```

### Test Accuracy
```
Test Loss: 0.3120
Test Accuracy: 84.50%
```
### Predicted Income for the given input
```
Predicted Income for the given input: >50K
```

## Requirements
To run the notebook locally, install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Setup Instructions (Google Colab)
If you want to run this notebook in Google Colab to get your final results:
1. Go to [Google Colab](https://colab.research.google.com/) and sign in.
2. Click on **File > Upload notebook** and upload the `notebooks/census_income_workshop.ipynb` file from this repository.
3. Once open, click **Runtime > Run all** to execute all the cells. 
4. The notebook is already configured to automatically download the `income.csv` dataset directly into the Colab environment using `gdown`. You don't need to manually upload the CSV file!
5. After the notebook finishes running, you can save your results.

## Repository Structure
- `notebooks/census_income_workshop.ipynb`: Jupyter notebook containing the full implementation of data prep, model architecture, training loop, evaluation, and inference.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.
