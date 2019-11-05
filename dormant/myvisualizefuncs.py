"""
    Functions used for visualizing data
    Author: Dougie Squire
    Date created: 09/03/2018
    Python Version: 3.5
"""

__all__ = ['plot_field']
    
# Local packages -----
from mygeneralfuncs import *
    
def plot_field(field, coastlines=True, animatein=None, Nframes=20, saveas=None,
               minC=None, maxC=None, plot_lat_labels=True, plot_lon_labels=True, flip_abscissa=False):
    '''
        File name: plot_field.py
        
        Description: Embeds a contour plot of the provide field in the Jupyter window. For 3D arrays,
            with animatein='coordinate', will embed an animation showing Nframes slices spanning the full
            length of the animatein coordinate. In these cases, the code places longitude along the ordinate 
            if such a coodinate exists
        Limitations: If a 3D array is provided with animatein=None, plots the first slice in time, or in the first
                listed dimension if not time dimension exists
                To save animations, ffmpeg must be installed
        
        Author: Dougie Squire
        Date created: 13/02/2018
        Python Version: 3.5
        
        Args:
            field (:class:`xarray.DataArray`): A 2D or 3D array
            coastlines (:class:`boolean`): Turns on/off coastline plotting
            animatein (:class:`string`): Dimension to animate along for 3D arrays. Set to 'None' for no animation
            Nframes (:class:`int`): Number of frames in animation
            minC (:class:`xarray.DataArray`): Minimum contour level
            maxC (:class:`xarray.DataArray`): Maximum contour level
            plot_lat_labels (:class:`xarray.DataArray`): Plot latitude labels
            plot_lon_labels (:class:`xarray.DataArray`): Plot longitude labels
    '''
    
    # Python packages -----
    from mpl_toolkits.basemap import Basemap
    from matplotlib import animation
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import xarray as xr
    import warnings
    import time
    import os
    
    # Widgets -----
    from ipywidgets import FloatProgress
    from IPython.display import display
    from IPython.display import HTML
    
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 12}

    mpl.rc('font', **font)

    field = np.squeeze(make_lon_positive(field))

    if animatein == None:

        # If 3D array given without animate command, plot first slice of time/other dimension -----
        if field.ndim == 3:
            try: 
                field = np.squeeze(field.isel(time = [0]))
                dimz = 'time'
            except: 
                dimz = field.dims[0]
                field = np.squeeze(field.loc[{dimz : field[dimz][0]}])
            warnings.warn("Specified 3D array with static plot settings. Plotting 1st slice of dimension: " + dimz)

        # Initialise plot variables -----
        field_plot = np.copy(field)
        dimx_name = field.dims[1]
        dimx = field[dimx_name]
        dimy_name = field.dims[0]
        dimy = field[dimy_name]      

        # Set up plotting -----
        fig = plt.figure(figsize=(10, 5), dpi= 150, facecolor='w', edgecolor='k')
        ax = plt.axes() 
        
        # Plot coastlines -----
        if coastlines:
            m = Basemap(resolution='c',projection='cyl',lon_0=180, 
                        llcrnrlon=0, llcrnrlat=-90, 
                        urcrnrlon=360, urcrnrlat=90)
            parallels = np.linspace(-90.,90., 5)
            meridians = np.linspace(0.,360., 5)

            m.drawcoastlines(linewidth=1,color='w')
            m.drawparallels(parallels,labels=[plot_lat_labels,False,False,False]) # labels = [left,right,top,bottom]
            m.drawmeridians(meridians,labels=[False,False,False,plot_lon_labels])

        # Plot contours -----
        if maxC!=None and minC!=None and minC<maxC:
            field_plot[field_plot<minC] = minC
            field_plot[field_plot>maxC] = maxC
            cont = plt.contourf(dimx,dimy,field_plot,vmin=minC,vmax=maxC);
        else:
            cont = plt.contourf(dimx,dimy,field_plot)
        if flip_abscissa:
                plt.gca().invert_yaxis()
        plt.title(dimz + ': ' + str(field[dimz].values))
        plt.xlabel(dimx_name, labelpad=20)
        plt.ylabel(dimy_name, labelpad=20)
        cbar = plt.colorbar(cont, fraction=0.03, pad=0.03)
        cbar.set_label(field.name, rotation=90, labelpad=10)
    else:
        # Check that Nframes is sensible -----
        if (len(field[animatein])-1)/Nframes < 1.0:
            Nframes = len(field[animatein])-1   
            warnings.warn("Nframes changed to length of dimension: " 
                          + animatein + " (ie " + str(len(field[animatein])-1) +
                          ")")
            
        # Instantiate progress bar -----
        f = FloatProgress(min=0, max=Nframes, description='animating...') 
        display(f)
        
        # Initialise plot variables -----
        nonanim_dims = [x for x in field.dims if animatein not in x]
        # Try to put longitude as x-coordinate -----
        try:
            ilon = nonanim_dims.index("lon")
            dimx_name = nonanim_dims[ilon]
            dimx = field[dimx_name]
            dimy_name = nonanim_dims[abs(ilon-1)]
            dimy = field[dimy_name]
        except: 
            dimx_name = nonanim_dims[1]
            dimx = field[dimx_name]
            dimy_name = nonanim_dims[0]
            dimy = field[dimy_name]  

        # Set up plotting -----
        fig = plt.figure(figsize=(10, 5), dpi= 150, facecolor='w', edgecolor='k')
        ax = plt.axes() 
        plt.ion()

        # Initialize colorbar range and ticks -----
        field_plot = np.copy(np.squeeze(field.loc[{animatein : field[animatein]}]))
        if maxC!=None and minC!=None and minC<maxC:
            clim_low = minC
            clim_upp = maxC
        else:
            clim_low = np.nanmin(field_plot)
            clim_upp = np.nanmax(field_plot)

        # Animation function -----
        def animate(i):

            # Signal to increment the progress bar -----
            f.value += 1 

            # Increment video over entire range of specified dimension -----
            step = np.floor(len(field[animatein])/Nframes)
            n = int(1 + i*step)

            # Plot coastlines -----
            plt.clf()
            if coastlines:
                m = Basemap(resolution='c',projection='cyl',lon_0=180, 
                            llcrnrlon=0, llcrnrlat=-90, 
                            urcrnrlon=360, urcrnrlat=90)
                parallels = np.linspace(-90.,90., 5)
                meridians = np.linspace(0.,360., 5)

                m.drawcoastlines(linewidth=1,color='w')
                m.drawparallels(parallels,labels=[plot_lat_labels,False,False,False]) # labels = [left,right,top,bottom]
                m.drawmeridians(meridians,labels=[False,False,False,plot_lon_labels])

            # Plot contours -----
            field_plot = np.copy(np.squeeze(field.loc[{animatein : field[animatein][n]}]))
            if maxC!=None and minC!=None and minC<maxC:
                field_plot[field_plot<minC] = minC
                field_plot[field_plot>maxC] = maxC
                cont = plt.contourf(dimx,dimy,field_plot,vmin=minC,vmax=maxC);
            else:
                cont = plt.contourf(dimx,dimy,field_plot)
            if flip_abscissa:
                plt.gca().invert_yaxis()
            plt.title(animatein + ': ' + str(field[animatein][n].values))
            plt.xlabel(dimx_name, labelpad=20)
            plt.ylabel(dimy_name, labelpad=20)
            cbar = plt.colorbar(cont, fraction=0.03, pad=0.03)
            cbar.set_label(field.name, rotation=90, labelpad=10)
            plt.clim(clim_low, clim_upp)
            plt.draw()

            return cont  
        
        # Buffer and imbed the animation -----
        anim = animation.FuncAnimation(fig, animate, frames=Nframes)
        
        # Change embed limit so can deal with up to ~ 300 frames -----
        plt.rcParams['animation.embed_limit'] = 200
        
        # Save the video if required -----
        Writer = animation.writers['ffmpeg'] # Set up formatting for the movie files
        writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)
        if saveas != None:
            anim.save(saveas, writer=writer)
        
        try:
            return HTML(anim.to_jshtml()) #HTML(anim.to_html5_video())
        finally:
            plt.close()
            try:
                os.remove('None0000000.png')
            except OSError:
                pass