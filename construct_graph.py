import geopandas as gpd
import pandas as pd
import geograph
import json


with open("./data/kfold2_features.json") as fd:
    feat_data = json.load(fd)  
    
with open("../pooling/data/migration_data.json") as l:
    labels = json.load(l)
    
with open("../archive/CAOE/data/census_data.json") as c:
    census = json.load(c)
    
keep_ids = list(feat_data.keys())
    
gdf = gpd.read_file("../pooling/data/MEX/ipumns_shp.shp")
gdf = gdf.dropna(subset = ["geometry"])
gdf = gdf[gdf['shapeID'].isin(keep_ids)]
print("Number of polygons: ", len(gdf), "  |  Number of ID's: ", len(keep_ids))
shapeIDs = gdf['shapeID'].to_list()
gdf["shapeID"] = [str(i) for i in range(0, len(gdf))]
gdf.head()


graph = {}
for shapeID in gdf['shapeID'].to_list():  
    
    muni_id = shapeIDs[int(shapeID)]
    
    try:
        
        g = geograph.GeoGraph(str(shapeID),
                              gdf, 
                              degrees = 1, 
                              load_data = False, 
                              boxes = False)    

        node_attrs = {'x': feat_data[muni_id] + census[muni_id],
                'label': labels[muni_id],
                'neighbors': g.degree_dict[1]
               }

        graph[shapeID] = node_attrs
        
    except:
        
        print("No features for ", muni_id)
    
    
with open("./data/kfold2_graph.json", "w") as outfile: 
    json.dump(graph, outfile)
    
    