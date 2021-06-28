import pandas as pd
import json


with open("./data/features_extract.json") as fd:
    feat_data = json.load(fd)  
feat_data = pd.DataFrame.from_dict(feat_data, orient = 'index').reset_index()
feat_data["index"] = feat_data["index"].str.split("/").str[5]
feat_data = feat_data.rename(columns = {'index':'muni_id'})
feat_data.head()


features = {}
for col, row in feat_data.iterrows():
    cur_feats = [row.glimpse0, row.glimpse1, row.glimpse2, row.glimpse3]
    cur_feats = [float(item) for sublist in cur_feats for item in sublist]
    features[row.muni_id] = cur_feats
    
    
with open("./data/features.json", "w") as outfile: 
    json.dump(features, outfile)