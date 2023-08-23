#!/usr/bin/env python
# coding: utf-8

# In[2]:


import osmnx as ox
import geopandas as gpd
#MatPlotLib
import matplotlib.pyplot as plt
#Numpy and Pandas
import numpy as np, pandas as pd
#Special Modules
from scipy import stats
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point

from pandana.loaders import osm

from shapely import geometry

import pandana  as pdna
import pandas as pd


# In[3]:


location_point =(25.269, 55.3031) 
Latitude1 = 55.2728
Longitude1 = 25.2394
Latitude2 = 55.3331
Longitude2 = 25.2872
bbox= (Longitude2, Latitude2, Longitude1, Latitude1)


# In[4]:


df= ox.features_from_point(location_point, dist=3000,   tags={'building':True} )


# In[5]:


G = ox.graph_from_point(location_point, dist=2000, dist_type="bbox", network_type="all")
nodes, edges = ox.graph_to_gdfs(G)

#Edge Prep
length= np.array(edges['length'])
edges=gpd.GeoDataFrame(G.edges())
edges.rename(columns= {0:'from'},inplace = True)
edges.rename(columns= {1:'to'},inplace = True)

edges['distance']=length
edges.to_csv(r'C:\Users\noura\edges_3.csv')

#Nodes Prep
nodes= ox.graph_to_gdfs(G, edges=False, nodes= True)
nodes_1= nodes.drop(columns=['highway','street_count','geometry'])
nodes_1.rename(columns= {'osmid':'id'},inplace = True)
nodes_1.to_csv(r'C:\Users\noura\nodes_3.csv')


# In[7]:


#Network Prep
nodes = pd.read_csv('nodes_3.csv', index_col=0)
edges = pd.read_csv('edges_3.csv', index_col=[0,1])
network = pdna.Network(node_x= nodes['x'], node_y=nodes['y'], edge_from= edges['from'], edge_to=edges['to'], edge_weights=edges[['distance']])


# In[8]:


#Fetch pois
buildings=  ox.features_from_point(location_point, dist=3000,   tags={'building':True} )
res= buildings.loc[:,['building','geometry']]


# In[9]:


#Fetch unique values
res['building'].unique()


# In[10]:


#1-Commercial/Retail
comm= res.query("building == 'commercial'")

pois_d= comm.geometry.representative_point()
pois_d_x=pois_d.get_coordinates()['x']
pois_d_y=pois_d.get_coordinates()['y']

network.set_pois(category = 'retail',maxdist = 800,maxitems = 10,x_col = pois_d_x,  y_col = pois_d_y)


# In[11]:


results_1 = network.nearest_pois(distance = 800,category = 'retail',num_pois =10,include_poi_ids = True)


# In[12]:


# keyword arguments to pass for the matplotlib figure
bbox_aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
fig_kwargs = {'facecolor':'w', 'figsize':(10, 10 * bbox_aspect_ratio),'frameon':True, 'dpi':1000}

# keyword arguments to pass for scatter plots
plot_kwargs = {'s':8, 
               'alpha':0.6, 
               'cmap':'viridis_r', 
               'edgecolor':'none'}

# network aggregation plots are the same as regular scatter plots, but without a reversed colormap
agg_plot_kwargs = plot_kwargs.copy()
agg_plot_kwargs['cmap'] = 'viridis'

# keyword arguments to pass for hex bin plots
hex_plot_kwargs = {'gridsize':60,
                   'alpha':0.9, 
                   'cmap':'viridis_r', 
                   'edgecolor':'none'}

# keyword arguments to pass to make the colorbar
cbar_kwargs = {}

# keyword arguments to pass to basemap
bmap_kwargs = {}

# color to make the background of the axis
bgcolor = 'k'
n = 1

bmap, fig, ax = network.plot(results_1[1],  plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs, 
                              cbar_kwargs=cbar_kwargs)


# In[ ]:




