# https://gist.github.com/ArcturusB/613eaba080a50385fa29e2eff8fe203f
# https://github.com/ArcturusB/papy
import numpy as np
def aggressive_autoscale(axes, axis, margin=0.1):
    ''' Autoscale an axis taking into account the limit along the other axis.
    
    Example use case: set the x-limits, then autoscale the y-axis using only
    the y-data within the x-limits. (Matplotlib's behaviour would be to use the
    full y-data.)
    
    Parameters
    ==========
    axes : matplotlib axes object
    axis : 'x' or 'y'
        The axis to autoscale
    margin : float
        The margin to add around the data limit, as a fraction of the data
        amplitude.
        
    Adapted from https://stackoverflow.com/a/35094823/4352108
    '''
    if axis not in ('x', 'y'):
        raise ValueError('invalid axis: '.format(axis))

    # determine axes data limits
    data_limits = []
    for line in axes.get_lines():
        # determine line data limits
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        xmin, xmax = line.axes.get_xlim()
        ymin, ymax = line.axes.get_ylim()
        xdata_displayed = xdata[(ymin < ydata) & (ydata < ymax)]
        ydata_displayed = ydata[(xmin < xdata) & (xdata < xmax)]
        xmin = np.nanmin(xdata_displayed)
        xmax = np.nanmax(xdata_displayed)
        ymin = np.nanmin(ydata_displayed)
        ymax = np.nanmax(ydata_displayed)
        line_limits = (xmin, xmax), (ymin, ymax)
        data_limits.append(line_limits)
    data_limits = np.array(data_limits)
    xmin, ymin = np.min(data_limits, axis=(0, 2))
    xmax, ymax = np.max(data_limits, axis=(0, 2))

    # apply margin
    x_margin = (xmax - xmin) * margin / 2
    y_margin = (ymax - ymin) * margin / 2
    xmin -= x_margin
    xmax += x_margin
    ymin -= y_margin
    ymax += y_margin

    # scale axes
    if axis == 'x':
        axes.set_xlim(xmin, xmax)
    elif axis == 'y':
        axes.set_ylim(ymin, ymax)