import pandas as pd
import numpy as np
from pymatgen.core import Composition
from sklearn.preprocessing import MinMaxScaler
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.site.fingerprint import VoronoiFingerprint
feature_calculators = MultipleFeaturizer([cf.AtomicOrbitals()])

def get_radii_properties():
    df = pd.read_csv('./data/Ionic_Radii.txt',
                 sep='\s+',
                 header=None,
                 names=["ions", "radii", "Sources"],
                 skiprows=1)
    # 分割第一列
    df['Element'] = df['ions'].str[:2]
    df['Charge'] = df['ions'].str[2:4]
    df['Coordination'] = df['ions'].str[5:].astype(np.int)

    def remove_char(x):
        if '_' in x:
            return str(x).split('_')[0]
        elif '+' in x:
            return int(str(x).split('+')[0])
        elif '-' in x:
            return 0 - int(str(x).split('-')[0])
        else:
            return x


    df['Element'] = df['Element'].map(lambda x: remove_char(x))
    df['Charge'] = df['Charge'].map(lambda x: remove_char(x))
        # 删除原始的第一列
    df.drop('ions', axis=1, inplace=True)
    return df

def get_normalized_element_properties(is_normalize):
    def normalized(df):
        # 获取数字部分列索引
        numeric_columns = df.select_dtypes(include=np.number).columns
    
        # 创建MinMaxScaler对象
        scaler = MinMaxScaler()

        # 拟合并转换数字部分数据
        numeric_data = df[numeric_columns]
        normalized_data = scaler.fit_transform(numeric_data)
        normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)

        # 合并归一化后的数字部分和非数字部分
        non_numeric_columns = df.columns.difference(numeric_columns)
        merged_df = pd.concat([normalized_df, df[non_numeric_columns]], axis=1)
        return merged_df
    ele_df1 = pd.read_csv('./data/elements.csv')
    ele_df2 = pd.read_csv('./data/elements2.csv')
    if is_normalize:
        ele_df1 = normalized(ele_df1)
        ele_df2 = normalized(ele_df2)
    return ele_df1,ele_df2



class Featurizor:
    def __init__(self,is_base=True,is_normalize=False):
        self.radi_df = get_radii_properties()
        self.ele_df1,self.ele_df2 = get_normalized_element_properties(is_normalize)
        self.is_base = is_base
        
    def get_base_features(self,data:pd.DataFrame):
        ele_df1 = self.ele_df1
        ele_df2 = self.ele_df2
        result = []
        for i,row in data.iterrows():
            tmp = {}
            a = row['A']
            b = row['B']
            c = row['C']
            # 密度
            tmp[r'$D_a$'] = ele_df1[ele_df1['symbol']==a]['Density'].values[0]
            tmp[r'$D_b$'] =  ele_df1[ele_df1['symbol']==b]['Density'].values[0]
            tmp[r'$D_c$'] =  ele_df1[ele_df1['symbol']==c]['Density'].values[0]
            tmp[r'$Ce_a$'] = ele_df2[ele_df2['symbol']==a]['Cohesive energy (eV)'].values[0]
            tmp[r'$Ce_b$'] =  ele_df2[ele_df2['symbol']==b]['Cohesive energy (eV)'].values[0]
            tmp[r'$Ce_c$'] =  ele_df2[ele_df2['symbol']==c]['Cohesive energy (eV)'].values[0]
            tmp[r'$Dp_a$'] = ele_df1[ele_df1['symbol']==a]['dipole_polarizability'].values[0]
            tmp[r'$Dp_b$'] =  ele_df1[ele_df1['symbol']==b]['dipole_polarizability'].values[0]
            tmp[r'$Dp_c$'] =  ele_df1[ele_df1['symbol']==c]['dipole_polarizability'].values[0]
            tmp[r'$Fi_a$'] = ele_df1[ele_df1['symbol']==a]['FirstIonization'].values[0]
            tmp[r'$Fi_b$'] =  ele_df1[ele_df1['symbol']==b]['FirstIonization'].values[0]
            tmp[r'$Fi_c$'] =  ele_df1[ele_df1['symbol']==c]['FirstIonization'].values[0]
            tmp[r'$Ve_a$'] = ele_df1[ele_df1['symbol']==a]['number_of_valence_electrons'].values[0]
            tmp[r'$Ve_b$'] =  ele_df1[ele_df1['symbol']==b]['number_of_valence_electrons'].values[0]
            tmp[r'$Ve_c$'] =  ele_df1[ele_df1['symbol']==c]['number_of_valence_electrons'].values[0]
            tmp[r'$N_a$'] = ele_df1[ele_df1['symbol']==a]['number'].values[0]
            tmp[r'$N_b$'] =  ele_df1[ele_df1['symbol']==b]['number'].values[0]
            tmp[r'$N_c$'] =  ele_df1[ele_df1['symbol']==c]['number'].values[0]
            tmp[r'$P_a$'] = ele_df1[ele_df1['symbol']==a]['Period'].values[0]
            tmp[r'$P_b$'] =  ele_df1[ele_df1['symbol']==b]['Period'].values[0]
            tmp[r'$P_c$'] =  ele_df1[ele_df1['symbol']==c]['Period'].values[0]
            tmp[r'$F_a$'] = ele_df2[ele_df2['symbol']==a]['Family number'].values[0]
            tmp[r'$F_b$'] =  ele_df2[ele_df2['symbol']==b]['Family number'].values[0]
            tmp[r'$F_c$'] =  ele_df2[ele_df2['symbol']==c]['Family number'].values[0]
            tmp[r'$G_a$'] = ele_df2[ele_df2['symbol']==a]['Group number'].values[0]
            tmp[r'$G_b$'] =  ele_df2[ele_df2['symbol']==b]['Group number'].values[0]
            tmp[r'$G_c$'] =  ele_df2[ele_df2['symbol']==c]['Group number'].values[0]
            tmp[r'$En_a$'] = ele_df1[ele_df1['symbol']==a]['Electronegativity'].values[0]
            tmp[r'$En_b$'] =  ele_df1[ele_df1['symbol']==b]['Electronegativity'].values[0]
            tmp[r'$En_c$'] =  ele_df1[ele_df1['symbol']==c]['Electronegativity'].values[0]
            tmp[r'$Ns_a$'] = ele_df2[ele_df2['symbol']==a]['Number of s electrons'].values[0] 
            tmp[r'$Ns_b$'] =  ele_df2[ele_df2['symbol']==b]['Number of s electrons'].values[0] 
            tmp[r'$Ns_c$'] =  ele_df2[ele_df2['symbol']==c]['Number of s electrons'].values[0]
            tmp[r'$Np_a$'] = ele_df2[ele_df2['symbol']==a]['Number of p electrons'].values[0] 
            tmp[r'$Np_b$'] =  ele_df2[ele_df2['symbol']==b]['Number of p electrons'].values[0] 
            tmp[r'$Np_c$'] =  ele_df2[ele_df2['symbol']==c]['Number of p electrons'].values[0] 
            tmp[r'$Nd_a$'] = ele_df2[ele_df2['symbol']==a]['Number of d electrons'].values[0]
            tmp[r'$Nd_b$'] =  ele_df2[ele_df2['symbol']==b]['Number of d electrons'].values[0]
            tmp[r'$Nd_c$'] =  ele_df2[ele_df2['symbol']==c]['Number of d electrons'].values[0]
            tmp[r"$Mt_a$"] = ele_df2[ele_df2['symbol']==a]['Melting point (K)'].values[0]
            tmp[r"$Mt_b$"] =  ele_df2[ele_df2['symbol']==b]['Melting point (K)'].values[0]
            tmp[r"$Mt_c$"] =  ele_df2[ele_df2['symbol']==c]['Melting point (K)'].values[0]
            tmp[r"$Bt_a$"] = ele_df2[ele_df2['symbol']==a]['Boiling point (K)'].values[0]
            tmp[r"$Bt_b$"] =  ele_df2[ele_df2['symbol']==b]['Boiling point (K)'].values[0]
            tmp[r"$Bt_c$"] =  ele_df2[ele_df2['symbol']==c]['Boiling point (K)'].values[0]
            tmp[r"$W_a$"] = ele_df2[ele_df2['symbol']==a]['Atomic weight'].values[0]
            tmp[r"$W_b$"] =  ele_df2[ele_df2['symbol']==b]['Atomic weight'].values[0]
            tmp[r"$W_c$"] =  ele_df2[ele_df2['symbol']==c]['Atomic weight'].values[0]
            tmp[r"$Mn_a$"] = ele_df2[ele_df2['symbol']==a]['Mendeleev number'].values[0]
            tmp[r"$Mn_b$"] =  ele_df2[ele_df2['symbol']==b]['Mendeleev number'].values[0]
            tmp[r"$Mn_c$"] =  ele_df2[ele_df2['symbol']==c]['Mendeleev number'].values[0]
            tmp[r"$Ar_a$"] = ele_df2[ele_df2['symbol']==a]['Atomic radius (Å)'].values[0]
            tmp[r"$Ar_b$"] =  ele_df2[ele_df2['symbol']==b]['Atomic radius (Å)'].values[0]
            tmp[r"$Ar_c$"] =  ele_df2[ele_df2['symbol']==c]['Atomic radius (Å)'].values[0]
            tmp[r"$Cr_a$"] = ele_df2[ele_df2['symbol']==a]['Covalent radius (Å)'].values[0]
            tmp[r"$Cr_b$"] =  ele_df2[ele_df2['symbol']==b]['Covalent radius (Å)'].values[0]
            tmp[r"$Cr_c$"] =  ele_df2[ele_df2['symbol']==c]['Covalent radius (Å)'].values[0]
            tmp[r"$GEn_a$"] = ele_df2[ele_df2['symbol']==a]['Gordy EN'].values[0]
            tmp[r"$GEn_b$"] =  ele_df2[ele_df2['symbol']==b]['Gordy EN'].values[0]
            tmp[r"$GEn_c$"] =  ele_df2[ele_df2['symbol']==c]['Gordy EN'].values[0]
            tmp[r"$AEn_a$"] = ele_df2[ele_df2['symbol']==a]['Allen EN'].values[0]
            tmp[r"$AEn_b$"] =  ele_df2[ele_df2['symbol']==b]['Allen EN'].values[0]
            tmp[r"$AEn_c$"] =  ele_df2[ele_df2['symbol']==c]['Allen EN'].values[0]
            result.append(tmp)
        return pd.DataFrame(result)
    def get_combined_features(self,data: pd.DataFrame):
        columns = data.columns
        A_features = []
        B_features = []
        C_features = []
        for column in columns:
            if str(column).__contains__('_a'):
                A_features.append(column)
            elif str(column).__contains__('_b'):
                B_features.append(column)
            elif str(column).__contains__('_c'):
                C_features.append(column)
        for a_feature, b_feature, c_feature in zip(A_features, B_features, C_features):

            # 提取相同部分并转换为集合
            common_chars = set(a_feature) & set(b_feature) & set(c_feature)

            # 去除"_"和空格
            common_chars = {c for c in common_chars if c != '_' and c != ' ' and c != '$'}

            # 保持字符顺序和大小写不变
            result = [c for c in a_feature if c in common_chars]

            # 将结果转换为字符串
            feature_name = ''.join(result)

            data[r'$%s^\mathrm{d1}$'% feature_name] = np.abs(data[a_feature] - data[b_feature])
            data[r'$%s^\mathrm{d2}$'% feature_name] = np.abs(data[a_feature] + data[b_feature]- data[c_feature])
            data[r'$%s^\mathrm{d3}$'% feature_name] = np.abs(data[a_feature] - data[c_feature])
            data[r'$%s^\mathrm{d4}$'% feature_name] = np.abs(data[b_feature] - data[c_feature])

            data[r'$%s^\mathrm{wd1}$'% feature_name] = np.abs(data[a_feature] - 2*data[b_feature])
            data[r'$%s^\mathrm{wd2}$'% feature_name] = np.abs((data[a_feature] + 2*data[b_feature])- 4*data[c_feature])
            data[r'$%s^\mathrm{wd3}$'% feature_name] = np.abs(data[a_feature] - 4*data[c_feature])
            data[r'$%s^\mathrm{wd4}$'% feature_name] = np.abs(2*data[b_feature] - 4*data[c_feature])

            data[r'$%s^\mathrm{s1}$'% feature_name] = data[a_feature] + data[b_feature]
            data[r'$%s^\mathrm{s2}$'% feature_name] = data[a_feature] + data[c_feature]
            data[r'$%s^\mathrm{s3}$'% feature_name] = data[b_feature] + data[c_feature]

            data[r'$%s^\mathrm{ws1}$'% feature_name] = data[a_feature] + 2*data[b_feature]
            data[r'$%s^\mathrm{ws2}$'% feature_name] = data[a_feature] + 4*data[c_feature]
            data[r'$%s^\mathrm{ws3}$'% feature_name] = 2*data[b_feature] + 4*data[c_feature]

            data[r'$%s^\mathrm{avg}$'% feature_name] = pd.concat([data[a_feature],data[b_feature],data[c_feature]],axis=1).mean(axis=1)
            data[r'$%s^\mathrm{std}$'% feature_name] = pd.concat([data[a_feature],data[b_feature],data[c_feature]],axis=1).std(axis=1)
            data[r'$%s^\mathrm{max}$'% feature_name] = pd.concat([data[a_feature],data[b_feature],data[c_feature]],axis=1).max(axis=1)
            data[r'$%s^\mathrm{min}$'% feature_name] = pd.concat([data[a_feature],data[b_feature],data[c_feature]],axis=1).min(axis=1)

        return data
    
    def featurize(self,data,is_new_data=False,add_factor=False,is_structure=True):
        spinels = data.copy()
        if is_structure:
            structure = spinels['structure'] 
        radi_df = self.radi_df
        A = []
        B = []
        C = []
        for i,row in spinels.iterrows():
            record = {}
            comp = Composition(row['formula'])
            for k, v in comp.items():
                if v == 1: A.append(k.symbol)
                if v == 2: B.append(k.symbol)
                if v == 4: C.append(k.symbol)
        spinels['A'] = A
        spinels['B'] = B
        spinels['C'] = C
        spinels = spinels[~spinels['C'].isin(['Te','Cl','F'])].reset_index(drop=True)
        spinels[r'$y$'] = spinels[r'$y$'].map(lambda x: 1 if x else 0)
        spinels = spinels[['formula',r'$y$']]
        result=[]
        to_drop_ids = []
        c_radii = {'O':1.4,'S':1.84,'Se':1.98,'Te':2.21}
        for i,row in spinels.iterrows():
            record = {}
            comp = Composition(row['formula'])
            for k, v in comp.items():
                if v == 1: record['A'] = k.symbol
                if v == 2: record['B'] = k.symbol
                if v == 4: record['C'] = k.symbol
            A_df = radi_df[(radi_df['Element']==record['A'])&(radi_df['Coordination']==4)]
            B_df = radi_df[(radi_df['Element']==record['B'])&(radi_df['Coordination']==6)]
            # 去除配位环境不对的材料
            if len(A_df) == 0 or len(B_df) == 0:
                to_drop_ids.append(i)
                continue
            record[r'$V_c$'] = -2
            record[r'$R_c$'] = c_radii[record['C']]
            # oxi A 2 B 3
            if 2 in A_df['Charge'].values and len(radi_df[(radi_df['Element']==record['B'])&(radi_df['Charge']==3)])>0:
                record[r'$V_a$'] = 2
                record[r'$R_a$'] = A_df[A_df['Charge']==2]['radii'].values[-1]
                record[r'$V_b$'] = 3
                record[r'$R_b$'] = B_df[B_df['Charge']==3]['radii'].values[-1]
                result.append(record)
                continue
            # oxi A 4 B 2
            elif 4 in A_df['Charge'].values and len(radi_df[(radi_df['Element']==record['B'])&(radi_df['Charge']==2)])>0:
                record[r'$V_a$'] = 4
                record[r'$R_a$'] = A_df[A_df['Charge']==4]['radii'].values[-1]
                record[r'$V_b$'] = 2
                record[r'$R_b$'] = B_df[B_df['Charge']==2]['radii'].values[-1]
                result.append(record)
                continue
            else:#存在非整数解
                combinations = []
                # 遍历列表A和列表B的所有组合
                for a in A_df['Charge'].values:
                    for b1 in B_df['Charge'].values:
                        for b2 in B_df['Charge'].values:
                        # 检查是否满足等式
                            if a + b1 + b2 == 8:
                                combination = (a, b1, b2)
                                combinations.append(combination)
                if len(combinations) > 0 and len(combinations) < 4:
                    combination = combinations[0]
                    record[r'$V_a$'] = combination[0]
                    record[r'$R_a$'] = A_df[A_df['Charge']==combination[0]]['radii'].values[-1]
                    record[r'$V_b$'] = (combination[1]+combination[2])/2
                    record[r'$R_b$'] = (B_df[B_df['Charge']==combination[1]]['radii'].values[-1]+
                                     B_df[B_df['Charge']==combination[2]]['radii'].values[-1])/2
                    result.append(record)
                    continue
                elif len(combinations) == 4:
                    combination = combinations[1]
                    record[r'$V_a$'] = combination[0]
                    record[r'$R_a$'] = A_df[A_df['Charge']==combination[0]]['radii'].values[-1]
                    record[r'$V_b$'] = (combination[1]+combination[2])/2
                    record[r'$R_b$'] = (B_df[B_df['Charge']==combination[1]]['radii'].values[-1]+
                                     B_df[B_df['Charge']==combination[2]]['radii'].values[-1])/2
                    result.append(record)
                    continue
                else:
                    to_drop_ids.append(i)
        spinels_new = spinels.drop(index=to_drop_ids).reset_index(drop=True)
        tmp = pd.DataFrame(result)
        spinels = pd.concat([spinels_new,tmp],axis=1).reset_index(drop=True)
        formula = spinels['formula']
        base_features = self.get_base_features(spinels)
        spinels_features = pd.concat([spinels.iloc[:,5:],base_features],axis=1)
        if self.is_base:
            features = spinels_features
            features[r'$(En)diff_{ab}$'] = np.abs(features[r'$En_a$'] - features[r'$En_b$'])
            features[r'$(En)diff_{cb}$'] = features[r'$En_c$'] - features[r'$En_b$']
            features[r'$(En)diff_{ca}$'] = features[r'$En_c$'] - features[r'$En_a$']
            

        else:
            features = self.get_combined_features(spinels_features)
        tf = (np.sqrt(3) * (spinels[r'$R_b$'] + spinels[r'$R_c$'])) / (2 * (spinels[r'$R_a$'] + spinels[r'$R_c$']))  # 容忍因子
        # features = get_combined_features(spinels_features)
        features[r'$tf$'] = tf
        # 四面体因子
        features[r'$t$'] = spinels[r'$R_a$']/spinels[r'$R_c$']
        # 八面体因子
        features[r'$o$'] = spinels[r'$R_b$']/spinels[r'$R_c$']
        
        data = pd.concat([spinels[['formula',r'$y$']],features],axis=1)
        
        if add_factor:
            data['BF'] = 1/(data['$Nd_b$']+data['$Ns_a$'])+np.log(data['$R_b$'])
            data = data.drop(columns=['$Nd_b$','$Ns_a$','$R_b$'])
        if 'composition' not in data.columns:
            data['composition'] = data['formula'].map(Composition)
        data = feature_calculators.featurize_dataframe(data, col_id='composition')
        data = data.drop(columns=['composition'])
        cha_dict = {'s':1,'p':2,'d':3,'f':4}
        data['HOMO_character'] = data['HOMO_character'].map(cha_dict)
        data['LUMO_character'] = data['LUMO_character'].map(cha_dict)
        from pymatgen.core import Element
        data['HOMO_element'] = data['HOMO_element'].map(lambda x: Element(x).number)
        data['LUMO_element'] = data['LUMO_element'].map(lambda x: Element(x).number)
        if is_structure:
            from matminer.featurizers.structure import SiteStatsFingerprint
            from matminer.featurizers.site.fingerprint import VoronoiFingerprint
            data['structure'] = structure
            voronoi = VoronoiFingerprint()
            sf = SiteStatsFingerprint(voronoi)

            data= sf.featurize_dataframe(data,'structure',ignore_errors=True)
            data = data.drop(columns=['structure'])
            data = data.fillna(0)
        data['formula'] = formula
        if is_new_data:
            return data
        else:
                # 删除某一列值全相同的列
            data = data.drop(data.columns[data.nunique() == 1], axis=1)

            # 计算相关系数矩阵
            corr_matrix = data.corr().abs()

            # 创建一个布尔型矩阵，标记要删除的特征
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.8)]

            # 删除特征
            data = data.drop(to_drop, axis=1)
            return data
        
    
    