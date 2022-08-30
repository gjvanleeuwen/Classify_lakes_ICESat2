import numpy as np
import os

import cartopy.crs as ccrs

import matplotlib.pyplot as plt

def scatter_plot_map(lon, lat, thin=1):
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.coastlines()
    # ax.gridlines()
    print(len(lon))
    ax.scatter(lon[0::thin], lat[0::thin], transform=ccrs.PlateCarree())
    print(len(lon[0::thin]))
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')


def plot_photon_height(IS2_atl03_mds, distance_photon, index=None, coords=None, beams=None, outpath=None):


    # -- create scatter plot of photon data for all beams
    if not beams:
        ax = {}
        f1, ((ax['gt1l'], ax['gt1r'])) = plt.subplots(num=1, nrows=1, ncols=2, sharex=True, sharey=True, figsize=(20, 20))  #(ax['gt2l'], ax['gt2r']), (ax['gt3l'], ax['gt3r']))

    else:
        ax = {beam_name : 0 for beam_name in beams}
        f1, ax = plt.subplots(num=1, nrows=len(beams), sharex=True, sharey=True)

    for gtx, ax1 in ax.items():
        # -- data for beam gtx
        val = IS2_atl03_mds[gtx]
        distance_from_start_photon_beam = distance_photon[gtx]
        # -- signal classification confidence
        # ice_sig_conf = val['heights']['signal_conf_ph'][:, 3]
        if index:
            ax1.plot(distance_from_start_photon_beam[index], val['heights']['h_ph'][index], ',', c='0.5')
        elif coords:
            print('not implemented yet')
        else:
            ax1.plot(distance_from_start_photon_beam, val['heights']['h_ph'][index], ',', c='0.5')

        ax1.set_title(gtx)
        # -- adjust ticks
        ax1.get_xaxis().set_tick_params(which='both', direction='in')
        ax1.get_yaxis().set_tick_params(which='both', direction='in')
    # -- add x and y labels
    for gtx in ['gt3l', 'gt3r']:
        ax[gtx].set_xlabel('distance')
    for gtx in ['gt1l', 'gt2l', 'gt3l']:
        ax[gtx].set_ylabel('Elevation above WGS84 Ellipsoid [m]')
    # -- show the plot
    if outpath:
        plt.savefig(outpath)

def plot_photon_height_single_beam(IS2_atl03_mds, gtx, distance_photon, index=None, coords=None, beams=None, outpath=None):
    # -- create scatter plot of photon data for all beams
    plt.ioff()
    f1, ax1 = plt.subplots(figsize=(20, 20))  #(ax['gt2l'], ax['gt2r']), (ax['gt3l'], ax['gt3r']))

    # -- data for beam gtx
    val = IS2_atl03_mds[gtx]
    distance_from_start_photon_beam = distance_photon[gtx]
    # -- signal classification confidence
    # ice_sig_conf = val['heights']['signal_conf_ph'][:, 3]
    if index:
        ax1.plot(distance_from_start_photon_beam[index] - min(distance_from_start_photon_beam), val['heights']['h_ph'][index], ',', c='0.5')
        try:
            min_lon, max_lon = np.round(min(val['heights']['lon_ph'][index]), 2), np.round(max(val['heights']['lon_ph'][index]),2)
            min_lat, max_lat = np.round(min(val['heights']['lat_ph'][index]),2), np.round(max(val['heights']['lat_ph'][index]),2)
        except:
            min_lon, min_lat, max_lat, max_lon = 0,0,0,0
    elif coords:
        print('not implemented yet')
    else:
        ax1.plot(distance_from_start_photon_beam - min(distance_from_start_photon_beam), val['heights']['h_ph'], ',', c='0.5')
        try:
            min_lon, max_lon = min(val['heights']['lon_ph'][index]), max(val['heights']['lon_ph'][index])
            min_lat, max_lat = min(val['heights']['lat_ph'][index]), max(val['heights']['lat_ph'][index])
        except:
            min_lon, min_lat, max_lat, max_lon = 0, 0, 0, 0

    ax1.set_title(gtx + "- for lon {}-{} lat {}-{}".format(min_lon, max_lon, min_lat, max_lat))
    # -- adjust ticks
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    # -- add x and y labels
    ax1.set_xlabel('distance')
    ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
    # -- show the plot
    if outpath:
        plt.savefig(outpath)


def plot_lake(dist_along, height_ph, dem_ph, example_lakes, lake_ID, outpath):
    f1, ax1 = plt.subplots(figsize=(20, 20))
    ax1.plot(dist_along - min(dist_along), height_ph,
             ',', c='0.5')
    ax1.plot(dist_along - min(dist_along), dem_ph,
             ',', c='0.5')

    min_lon, max_lon = example_lakes['lon'][lake_ID][0], example_lakes['lon'][lake_ID][1]
    min_lat, max_lat = example_lakes['lat'][lake_ID][0], example_lakes['lat'][lake_ID][1]

    ax1.set_title(example_lakes['beam'][lake_ID] + "- for lon {}/{} lat {}/{}".format(min_lon, max_lon, min_lat, max_lat))
    # -- adjust ticks
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    # -- add x and y labels
    ax1.set_xlabel('distance')
    ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
    # -- show the plot
    plt.ioff()
    plt.savefig(outpath)


def plot_classified_photons(data_df, clusters, ph_start, eps, outpath, add_line=False, line_data=None):
    plt.ioff()
    f1, ax1 = plt.subplots(figsize=(20, 20))
    ax1.scatter(data_df['distance'], data_df['height'], c=clusters, cmap='Set2', marker=',',
                s=0.5)
    if add_line:
        ax1.plot(line_data.iloc[:,0], line_data.iloc[:,1])

    ax1.set_title('classification gt1l' + "- for photons {} and EPS {}".format(ph_start, eps))
    ax1.get_xaxis().set_tick_params(which='both', direction='in')
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.set_xlabel('distance')
    ax1.set_ylabel('Elevation above WGS84 Ellipsoid [m]')
    plt.savefig(outpath)




