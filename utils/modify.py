import numpy as np
import pandas as pd
from CrabNet_eg.kingcrab import CrabNet
from CrabNet_eg.model import Model
from CrabNet_eg.get_compute_device import get_compute_device

compute_device = get_compute_device()

class Modifier:
    def __init__(self):
        self.model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                      model_name='predict_new_data', verbose=True, classification=False)
        self.model.load_network(f'./models/trained_models/fold_2.pth')

    def modify(self, formula: list, band_gap: list):
        df = pd.DataFrame({'formula': formula, 'band gap': band_gap})
        df['target'] = 0
        self.model.load_data(df)
        pred = self.model.predict(self.model.data_loader)[1]
        pred = np.nan_to_num(pred, copy=True, nan=0.0)
        df['target'] = pred
        df = df.rename(columns={'target': 'hse'})
        return df

