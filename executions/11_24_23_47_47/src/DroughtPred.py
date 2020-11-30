
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:15:34 2019

@author: deborahkhider
@description: Climate indices (drought) for DARPA World's modeler program
"""
import xarray as xr
import pandas as pd
import numpy as np
import uuid
from datetime import date
import datetime
import os
#import glob
#from calendar import monthrange
import sys
import ast
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import imageio
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import json
import matplotlib.cm as cm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#%% Open files. Needs to contain the calculated SPI index

def openDataset(dataset_name, bounding_box, globe):
    """Open ECMWF datasets and return precipitation and temperature

    Args:
        dataset_name (str): The name of the ECMWF file
        bounding_box (list): lat/lon to cut to appropriate size
        globe (bool): If considering the full spatial coverage

    Returns:
        da_precip (Xarra DataArray): A dataArray of precipitation
        da_temp (Xarra DataArray): A dataArray of temperature
    """

    # Loop over all datasets in the various folders to get the data
    data = xr.open_dataset(dataset_name)
    if globe == True:
        da_spi = data.spi
    else:
        p_ = data.sel(latitude=slice(bounding_box[2], bounding_box[3]),\
                  longitude=slice(bounding_box[0],bounding_box[1]))
        da_spi = p_.spi

    return da_spi


#%% Functions below allows to make predictions up to 4 months in advance

# Assemble predictors/predictands

def assemble_predictors (da_spi, forecast, start_date):
    '''

    Parameters
    ----------
    da_spi : xarray dataarray
        A dataarray of SPI values
    forecast : bool
        Whether to use forecasting mode (True) or Hindcasting mode (False)
    start_date : str
        When to start the hindcasting. Default is '2020-01-01'. Ignored for forecasting mode.

    Raises
    ------
    ValueError
        Hindcasting has yet to be implemented

    Returns
    -------
    X : Numpy array
        Array of SPI values for the last three months to initiate the forecast
    start_date : datetime
        The start date for the hindcasting (as selected) or forecasting (end of available data)
    '''

    if forecast == True:
        #grab the last of the data
       spi = np.stack([da_spi.values[-3:,]])
       spi[np.isnan(spi)] = 0 #remove nans
       X = spi.astype(np.float32) #Make sure we have floats
       start_date = da_spi['time'][-1].values
    else:
        raise ValueError("Hindcasting hasn't been implemented")

    return X, start_date

#Use this class to load the predictions
class SPIPredictors(Dataset):
    def __init__(self,predictors):
        self.predictors = predictors
    def __len__(self):
        return self.predictors.shape[0]
    def __getitem__(self, idx):
        return self.predictors[idx]

#CNN
class CNN(nn.Module):
    def __init__(self, num_input_time_steps=1, print_feature_dimension=False):
        """
        inputs
        -------
            num_input_time_steps        (int) : the number of input time
                                                steps in the predictor
            print_feature_dimension    (bool) : whether or not to print
                                                out the dimension of the features
                                                extracted from the conv layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_input_time_steps, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 42, 5)
        self.fc1 = nn.Linear(42*3*10, 13*20)
        self.print_layer = Print()
        self.print_feature_dimension = print_feature_dimension

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if self.print_feature_dimension:
          x = self.print_layer(x)
        x = x.view(-1, 42*3*10)
        x = self.fc1(x)
        return x

class Print(nn.Module):
    """
    This class prints out the size of the features
    """
    def forward(self, x):
        print(x.size())
        return x

def makePredictions(lead_time,predictloader,size=[13,20]):
    '''make predictions on the value for SPI at various lead time

    Parameters
    ----------
    lead_time : int
        How many months in advance
    predictloader : utils.data.dataloader.DataLoader
        Pytorch Dataloader from Class SPIPredictors
    size : list, optional
        List of spatial grid size. The default is [13,20].

    Returns
    -------
    predictions : numpy array
        Array of SPI predictions per lead months

    '''
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device=torch.device('cpu')
    net = CNN(3,False)
    lead_times = np.arange(1,lead_time+1).tolist()
    lead_times=[str(item) for item in lead_times] # transform to string

    #intantiate a matrix of zeros
    predictions = np.zeros((lead_time,size[0],size[1]))

    for idx, t in enumerate(lead_times):
        net.load_state_dict(torch.load('threelayerCNN_0400-01-01_2100-12-31_{}.pt'.format(t),map_location=device))
        net.eval()
        net.to(device)

        try:
            del predictions_t
        except:
            pass

        for i, data in enumerate(predictloader):
            batch_predictors = data
            batch_predictors = batch_predictors.to(device)
            batch_predictions = net(batch_predictors).squeeze()
            batch_predictions = batch_predictions.detach().cpu().numpy()
            batch_predictions=batch_predictions.reshape(size[0],size[1])
            try:
                predictions_t = np.stack((predictions_t, batch_predictions))
            except NameError:
                predictions_t = batch_predictions
        predictions[idx,:,:] = predictions_t

    return predictions
#%% Return a netcdf using MINT conventions

def to_netcdfMint(start_date,lead_time,predictions,da_spi):

    #Adding the global attributes
    t_start = pd.to_datetime(start_date)+pd.DateOffset(months=1)
    t_end = pd.to_datetime(start_date)+pd.DateOffset(months=lead_time)

    # Get the time vector
    time=[t_start]
    while time[-1]<t_end:
        time.append(time[-1]+pd.DateOffset(months=1))
    time=np.array(time,dtype='datetime64')

    #Get lat/lon coords
    if 'latitude' in da_spi.coords:
        latitude = da_spi['latitude'].values
        longitude = da_spi['longitude'].values
    elif 'lat' in da_spi.coords:
        latitude = da_spi['lat'].values
        longitude = da_spi['lon'].values
    elif 'Y' in da_spi.coords:
        latitude = da_spi['Y'].values
        longitude = da_spi['X'].values

    #Create a data array
    da_for = xr.DataArray(predictions, coords=[time,latitude,longitude],dims=['time','latitude','longitude'])
    # To ds
    ds = da_for.to_dataset(name='spi')
    ds.attrs['conventions'] = 'MINT-1.0, Galois cube'
    long_name = 'Standardized Precipitation Index'
    ds.attrs['title'] = long_name
    ds.attrs['description'] = long_name + 'forecasted using a CNN with three convolutional layers and 1 fully connected layer. The CNN was trained on 1700 years of data from the CESM control run and tested on ECMWF ERA5 data for the period the period 1981-2017. Accuracy as assessed by the Pearson correlation coefficient varies between 0.8 for projection at 1 month to 0.4 for projection at 4 month.'
    ds.attrs['naming_authority'] = "MINT Workflow"
    ds.attrs['id'] = str(uuid.uuid4())
    ds.attrs['date_created'] = str(date.today())
    ds.attrs['date_modified']= str(date.today())
    ds.attrs['creator_name'] = 'Deborah Khider'
    ds.attrs['creator_email'] = 'khider@usc.edu'
    ds.attrs['institution'] = 'USC Information Sciences Institute'
    ds.attrs['project'] = 'MINT'
    ds.attrs['time_coverage_start'] = str(ds.time.values[0])
    ds.attrs['time_coverage_end'] = str(ds.time.values[-1])
    ds.attrs['time_coverage_resolution'] = 'monthly'
    ds.attrs['dependent_vars'] = 'SPI'
    ds.attrs['tags'] = ['Drought', 'Climate', 'SPI']
    ds.attrs['geospatial_lat_min'] = float(ds.latitude.min())
    ds.attrs['geospatial_lat_max'] = float(ds.latitude.max())
    ds.attrs['geospatial_lon_min'] = float(ds.longitude.min())
    ds.attrs['geospatial_lon_max'] = float(ds.longitude.max())
    ds.attrs['independent_vars'] = ['latitude', 'longitude', 'time']
    # Adding var attributes
    ds.spi.attrs['title'] = 'Standardized Precipitation Index'
    ds.spi.attrs['standard_name'] = 'atmosphere_water__standardized_precipitation_wetness_index'
    ds.spi.attrs['long_name'] = 'Standardized Precipitation Index'
    ds.spi.attrs['units'] = 'unitless'
    ds.spi.attrs['valid_min'] = float(ds.spi.min())
    ds.spi.attrs['valid_max'] = float(ds.spi.max())
    ds.spi.attrs['valid_range'] = list((ds.spi.attrs['valid_min'], ds.spi.attrs['valid_max']))
    ds.spi.attrs['missing_value'] = np.nan

    #Write it out to file
    if os.path.isdir('./results') is False:
        os.makedirs('./results')
    ds.to_netcdf(path = './results/results.nc')

    return ds

#%% Visualization

def visualizeDroughtIndex(ds):
    """ Visualization of drought index

    Args:
        ds (xarray dataset): the dataset containing the index
        dir_out (str): the output directory for the visualization
        dynamic_name (bool): Whether to generate a unique name dynamically using parameter settings. Otherwise returns, results.mp4
    """
    proj=ccrs.PlateCarree()
    idx = np.size(ds['time'])
    count = list(np.arange(0,idx,1))
    varname=list(ds.data_vars.keys())[0]
    filenames=[]

    #Make a directory for results/figures if it doesn't exit
    if os.path.isdir('./results') is False:
        os.makedirs('./results')

    if os.path.isdir('./figures') is False:
        os.makedirs('./figures')

    # Get the levels for the countours
    if varname == 'spi' or varname == 'spei':
        levels = np.arange(-4,4.2,0.2)
    else:
        levels = np.linspace(float(ds[varname].min()),float(ds[varname].max()),40)
    for i in count:
        fig,ax = plt.subplots(figsize=[15,10])
        ax = plt.axes(projection=proj)
        if xr.apply_ufunc(np.isnan,ds[varname].isel(time=i)).all()==False:
            if varname == 'spi' or varname == 'spei':
                ds[varname].isel(time=i).plot.contourf(ax=ax,levels = levels,
                  transform=ccrs.PlateCarree(), cmap=cm.BrBG, cbar_kwargs={'orientation':'horizontal'})
            else:
                ds[varname].isel(time=i).plot.contourf(ax=ax,levels = levels,
                  transform=ccrs.PlateCarree(), cmap=cm.viridis, cbar_kwargs={'orientation':'horizontal'})
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.RIVERS)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = False
        gl.ylines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 12, 'color': 'gray'}
        gl.ylabel_style = {'size': 12, 'color': 'gray'}
        #save a jepg

        filename = './figures/'+varname+'_t'+str(i)+'.jpeg'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)

    #create a gif

    writer = imageio.get_writer('./results/results.mp4', fps=1)

    for filename in filenames:
        writer.append_data(imageio.imread(filename))
    writer.close()

#%% Main
if __name__ == "__main__":
    #Open JSON file and get settings
    with open(sys.argv[1]) as json_file:
        config = json.load(json_file)
    dataset_name=config['data']['dataset_name']
    fig = ast.literal_eval(config['output']['fig'])
    lead_time = int(config['time']['lead_time'])
    forecast = ast.literal_eval(config['time']['forecast'])
    hindcast_start = config['time']['hindcast_start']
    globe = ast.literal_eval(config['spatial']['global'])
    if globe == False:
        bounding_box = ast.literal_eval(config['spatial']['bounding_box'])
    elif globe == True:
        bounding_box = []
    else:
        raise ValueError("globe option should be set as 'True' or 'False'")
    #Make sure everything is as it should be
    if lead_time>4:
        raise ValueError('The model currently supports lead times between 1 and 4')
    #Open the data
    da_spi = openDataset(dataset_name,bounding_box,globe)
    #Assemble the predictors
    X, start_date = assemble_predictors(da_spi, forecast, hindcast_start)
    #Make predictions
    predict_dataset = SPIPredictors(X)
    predictloader = DataLoader(predict_dataset)
    size = [X.shape[-2],X.shape[-1]]
    predictions = makePredictions(lead_time,predictloader,size)
    # Save
    ds = to_netcdfMint(start_date,lead_time,predictions,da_spi)
    #Figure
    if fig == True:
        visualizeDroughtIndex(ds)
