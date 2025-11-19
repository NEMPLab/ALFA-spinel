import pandas as pd
import numpy as np
from CrabNet.kingcrab import CrabNet
from CrabNet.model import Model
from CrabNet.get_compute_device import get_compute_device
from pymatgen.core import Composition

compute_device = get_compute_device()
model = Model(CrabNet(compute_device=compute_device).to(compute_device),
              model_name="", verbose=False, classification=False)


def get_new_spinels(spinels_df):
    A = np.array(['Ag', 'Al', 'Ba', 'Be', 'Ca', 'Co', 'Cs', 'Cu', 'Fe', 'Ga', 'Ge', 'La', 'Mg', 'Mn', 'Mo', 'Ni'
                     , 'Rb', 'V', 'W', 'Zn'])
    B = np.array(['Ag', 'Al', 'Bi', 'Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'In', 'La', 'Lu',
                  'Mg', 'Mn', 'Mo', 'Na', 'Nd', 'Ni', 'Pd', 'Rb', 'Ti', 'V', 'W', 'Zn'])
    C = np.array(['O', 'S', 'Se'])
    formulee = spinels_df['formula'].values
    result = []
    for a in A:
        for b in B:
            for c in C:
                if a != b:
                    formula = a + b + str(2) + c + str(4)
                    if Composition(formula).reduced_formula not in list(formulee):
                        result.append({"formula": formula, "A": a, "B": b, "C": c})
    return pd.DataFrame(result)


def screen_Ef(spinels_df):
    model.load_network('./models/trained_models/eform.pth')
    new_data = spinels_df.copy()
    new_data['target'] = 0
    model.load_data(new_data)
    pred = model.predict(model.data_loader)[1]
    pred = np.nan_to_num(pred, copy=True, nan=0.0)
    new_data['Ef'] = pred
    result = new_data[new_data['Ef'] <= 0].reset_index(drop=True)
    print('origin length:%s' % len(new_data))
    print('after Ef screen:%s' % len(result))
    return result


def screen_Eh(spinels_df):
    model.load_network('./models/trained_models/ehull.pth')
    model.load_data(spinels_df)
    pred = model.predict(model.data_loader)[1]
    pred = np.nan_to_num(pred, copy=True, nan=0.0)
    spinels_df['Eh'] = pred
    result = spinels_df[spinels_df['Eh'] <= 0.025].reset_index(drop=True)
    print('origin length:%s' % len(spinels_df))
    print('after Eh screen:%s' % len(result))
    return result


def screen_non_metal(spinels_df):
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name="", verbose=False, classification=True)
    model.load_network('./models/trained_models/is_metal.pth')
    spinels_df['classification_target'] = 0
    model.load_data(spinels_df)
    pred = model.predict(model.data_loader)[1]
    pred = np.nan_to_num(pred, copy=True, nan=0.0)
    spinels_df['is_metal'] = pred
    result = spinels_df[spinels_df['is_metal'] == 0].reset_index(drop=True)
    print('origin length:%s' % len(spinels_df))
    print('after non metal screen:%s' % len(result))
    return result


if __name__ == '__main__':
    spinels = pd.read_csv('spinels.csv', index_col=0)
    new_spinels = get_new_spinels(spinels)
    new_spinels = screen_Ef(new_spinels)
    new_spinels = screen_Eh(new_spinels)
    new_spinels = screen_non_metal(new_spinels)
    new_spinels = new_spinels.drop(columns=['classification_target', 'target', 'count'])
    print(new_spinels)
    new_spinels.to_csv('new_spinels.csv')