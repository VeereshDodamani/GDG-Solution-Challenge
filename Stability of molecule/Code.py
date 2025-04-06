!pip install deepchem
!pip install rdkit

import deepchem as dc
from deepchem.data import NumpyDataset
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def generate_random_smiles(num_samples=1000):
    smiles_list = []
    stability_values = []
    for _ in range(num_samples):
        mol = Chem.MolFromSmiles("C" + "C" * np.random.randint(1, 10))
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
            stability = Descriptors.MolWt(mol) + np.random.normal(0, 10)
            stability_values.append(stability)
    return smiles_list, stability_values

smiles, stability = generate_random_smiles(num_samples=1000)
data = pd.DataFrame({"SMILES": smiles, "Stability": stability})
data.to_csv("molecule_stability_dataset.csv", index=False)
print("Dataset created and saved to 'molecule_stability_dataset.csv'.")

import deepchem as dc
from deepchem.feat import CircularFingerprint

dataset_file = "molecule_stability_dataset.csv"
tasks = ["Stability"]
featurizer = CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=tasks, feature_field="SMILES", featurizer=featurizer)
dataset = loader.create_dataset(dataset_file)

splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

from deepchem.models import MultitaskRegressor

model = MultitaskRegressor(
    n_tasks=1,
    n_features=1024,
    layer_sizes=[512, 256, 128],
    dropouts=0.2,
    learning_rate=0.001,
    model_dir="stability_model"
)

model.fit(train_dataset, nb_epoch=50)

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
train_scores = model.evaluate(train_dataset, [metric])
valid_scores = model.evaluate(valid_dataset, [metric])
test_scores = model.evaluate(test_dataset, [metric])

print(f"Train R^2 Score: {train_scores}")
print(f"Validation R^2 Score: {valid_scores}")
print(f"Test R^2 Score: {test_scores}")


new_smiles = "CCO"

new_mol = Chem.MolFromSmiles(new_smiles)
new_features = featurizer.featurize([new_mol])
new_dataset = dc.data.NumpyDataset(X=new_features)

predicted_stability = model.predict(new_dataset)
print(f"Predicted Stability for {new_smiles}: {predicted_stability}")
