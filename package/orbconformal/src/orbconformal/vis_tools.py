import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def vis_surfaces(z_multi,
                initial_opacity = .4,
                high_opacity = .9,
                low_opacity = .2):
    """
    Create a plotly visualization of all surfaces (includes a dropdown to
    strong highlight a single curve)

    Parameters
    ----------
    z_multi : numpy.ndarray
        numpy array (r, n, d) of 2d multivariate functions (each row is a
        single funciton)
    initial_opacity : float
        initial opacity for all multivariate functions in the figure
    high_opacity : float
        opacity for observation when the user selects that specific observation
    low_opacity : float
        opacity for the rest of the observations when user selects a specific
        observation

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Figure visualizing all the multivariate functions with interactive
        selector to select a single function to emphasis.
    """
    n_surfaces = z_multi.shape[0]
    fig = go.Figure(data =
                    [go.Surface(z = z_multi[i],
                                surfacecolor = i/n_surfaces *\
                                                np.ones(shape = z_multi.shape[1:]),
                               showscale = False, opacity = initial_opacity,
                               colorscale = "mygbm",
                               cmin = 0, cmax = 1) for i in np.arange(z_multi.shape[0])
                    ])

    fig.update_layout(updatemenus=[
            dict(
                type = "dropdown",
                direction = "down",
                buttons=list([
                    dict(
                        args=["opacity", [low_opacity]*(i) +\
                                         [high_opacity] +\
                                         [low_opacity]*(n_surfaces-i-1)],
                        label=str(i),
                        method="restyle"
                    ) for i in np.arange(z_multi.shape[0])

                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ])

    fig.update_layout(
        annotations=[
            dict(text="Highlight: ", x=0, xref="paper", y=1.06, yref="paper",
                                 align="left", showarrow=False)])

    #fig.show()
    return fig

def vis_sequence_surface(z_multi):
    """
    Visual animation of a sequence of surfaces

    Parameters
    ----------
    z_multi : numpy.ndarray
        numpy array (r, n, d) of 2d multivariate functions (each row is a
        single funciton)

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Figure visualizing the sequence of surfaces with a play button to
        move through all of them.
    """
    # animation for convergence: https://plotly.com/python/animations/


    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300,
                      # "easing": "cubic-in-out"
                      },
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }


    # original data?
    fig_dict["data"] = [go.Surface(z = z_multi[0],
                                    surfacecolor = np.ones(shape = z_multi.shape[1:]),
                                   showscale = False)]
    # frames part:
    for i in np.arange(z_multi.shape[0], dtype = int):
        frame = {"data": [go.Surface(z = z_multi[i],
                                    surfacecolor = np.ones(shape = z_multi.shape[1:]),
                                   showscale = False)
                        ], "name": str(i)}
        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [i],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": str(i),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)





    fig_dict["layout"]["sliders"] = [sliders_dict]


    fig = go.Figure(fig_dict)

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[0,5],),
                         yaxis = dict(nticks=4, range=[0,14],),
                         zaxis = dict(nticks=4, range=[6,12],),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))


    fig.show()


def get_plane_ben(M,v, axis_str, x_range, y_range, z_range):
    """
    Internal function to get plan related to a starting point and angular
    vector of a plane

    Arguments
    ---------
    M : numpy array
        3d vector of direction of plane
    v : numpy array
        3d starting point for the plane vector to define from
    axis_str : string
        string of either ("x", "y" or "xy") for the type of plane in 3d
    x_range : numpy array
        vector with the minimum and maximum range of x values
    y_range : numpy array
        vector with the minimum and maximum range of y values
    x_range : numpy array
        vector with the minimum and maximum range of z values

    Returns:
    --------
    vectors of X, Y and Z that define the plane (each should contain 4 points)
    """
    x0, y0, _= M
    a, b, _= v

    if axis_str == "x":
        Y = y_range
        X = x0*np.ones(2)
        Z = np.array(list(z_range)+list(z_range)).reshape((2,2))
    elif axis_str == "y":
        X = x_range
        Y = y0*np.ones(2)
        Z = np.array([z_range[0]]*2+[z_range[1]]*2).reshape((2,2))
    elif axis_str == "xy":
        X = x_range
        Y = y0+b*(X-x0)/a
        Z = np.array([z_range[0]]*2+[z_range[1]]*2).reshape((2,2))
    else:
        pass
    return X, Y, Z

def range_ben(x):
    """
    internal function to caculate range of a array (min and max value)

    Arguments
    ---------
    x : numpy array
        object to take min and max of

    Returns
    -------
    numpy array with min and max value (aka length 2)
    """
    return np.array([x.min(), x.max()])


def vis_slice_x(xx, yy, zs):
    """
    visualize slices of surfaces and slices relative to x

    Arguments
    ---------
    xx : numpy array
        vector (n, ) of x values that define the surfaces
    yy : numpy array
        vector (m, ) of y values that define the surfaces
    zs : list
        list of numpy arrays each (n,m) relative to x and y values

    Returns
    -------
    fig : plotly object of 2 subplots
    """
    fig = make_subplots(
         rows=1, cols=2,
         horizontal_spacing=0.1,
         column_widths=[0.6, 0.4],
         specs=[[{"type": "scene"}, {"type": "xy"}]])

    counter_surface = 0
    for index in range(len(zs)):
        fig.add_trace(go.Surface(x=xx,
                                  y=yy,
                                  z=zs[index],
                                  colorscale="Viridis",
                                  showscale=False,
                                  opacity = .3), row=1, col=1)
        counter_surface += 1

    # --------
    # x slices
    # --------

    counter_xxx_slices = 0

    xxx_inner_storage = list()

    for xxx_idx, xxx in enumerate(xx):# slice based on y
        My = [xxx,0,0]
        M = My
        vy = [0,1,0]
        v = vy


        X,Y,Z = get_plane_ben(M, v, "x", range_ben(xx),
                               range_ben(yy), range_ben(np.array(zs)))

        fig.add_trace(go.Surface(x=X.copy(), y=Y.copy(), z=Z.copy(),
                                 colorscale= [[0, "rgb(254, 254, 254)"],
                                              [1, "rgb(254, 254, 254)"]],
                                 showscale=False,
                                 opacity =0.65,
                                 visible=False), row=1, col=1,
                                 )

        # per y associated splices of surfaces
        for index in range(len(zs)):
            xxx_inner_storage.append(zs[index][:,xxx_idx])
            ival = len(xxx_inner_storage)
            fig.add_trace(go.Scatter(x= yy,
                                     y = xxx_inner_storage[ival-1].copy(), mode="lines",
                                    visible=False), row=1, col=2)    # 2d lines

            counter_xxx_slices += 1

    # initializing y slice shown (col 1 and col 2)
    fig.data[counter_surface].visible = True
    for index in range(len(zs)):
        fig.data[counter_surface+1+index].visible = True




    # ---------------
    # create slider x
    # ---------------

    steps_xxx = []
    block_size = len(zs)+1
    for xxx_idx, xxx in enumerate(xx):
        step = dict(
            method="update",
            args=[{"visible": [True]*counter_surface +\
                               [False] * (len(fig.data)-counter_surface)},
                  #^ turth is for surfaces to always be shown, second is originally setting rest as FALSE
                  {"title": "X slice: " + str(xxx)}],  # layout attribute
        )

        for index in range(len(zs)+1):
            step["args"][0]["visible"][counter_surface +\
                                       block_size*xxx_idx + index] = True  # Toggle index'th group to "visible"

        steps_xxx.append(step)



    sliders = [dict(
        active=0,
        currentvalue={"prefix": "X Slice: "},
        steps=steps_xxx
    )]



    fig.update_layout(
        sliders=sliders
    )

    fig['layout']['sliders'][0]['currentvalue']['prefix']='X value: '
    for xxx_index, xxx in enumerate(xx):
        fig['layout']['sliders'][0]['steps'][xxx_index]['label'] = str(xxx)

    fig.update_layout(# figure info:
                      width=900, height=700,
                      title_text="Slicing a surface by a plane (X)",
                      title_x=0.5,
                      # 3d info:
                      scene= {"camera": {"eye": {"x": 1.65*1.3, "z":0.75*1.3}},
                             "xaxis": dict(
                             backgroundcolor="rgba(0, 0, 0,0)",
                             gridcolor="rgba(0, 0, 0,.2)",
                             showbackground=True,
                             zerolinecolor="rgba(0, 0, 0,.5)"),
                             "yaxis": dict(
                             backgroundcolor="rgba(0, 0, 0,0)",
                             gridcolor="rgba(0, 0, 0,.2)",
                             showbackground=True,
                             zerolinecolor="rgba(0, 0, 0,.5)"),
                             "zaxis":dict(
                             backgroundcolor="rgba(0, 0, 0,0)",
                             gridcolor="rgba(0, 0, 0,.2)",
                             showbackground=True,
                             zerolinecolor="rgba(0, 0, 0,.5)")},

                      # 2d info:
                      yaxis = {"domain":  [0, 0.85],
                              "gridcolor":"rgba(0, 0, 0,.2)",
                              "zerolinecolor":"rgba(0, 0, 0,.3)"},
                      plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(
                             gridcolor="rgba(0, 0, 0,.2)",
                             zerolinecolor="rgba(0, 0, 0,.3)")
                  )

    fig.update_yaxes(range = range_ben(np.array(zs)),
                     title = "z value", row=1, col=2)
    fig.update_xaxes(title = "y value", row=1, col=2)

    fig.update(layout_showlegend=False)
    return(fig)


def vis_slice_y(xx,yy,zs, keep_2d_legend = False):
    """
    visualize slices of surfaces and slices relative to y

    Arguments
    ---------
    xx : numpy array
        vector (n, ) of x values that define the surfaces
    yy : numpy array
        vector (m, ) of y values that define the surfaces
    zs : list
        list of numpy arrays each (n,m) relative to x and y values
    keep_2d_legend : boolean
        default is false

    Returns
    -------
    fig : plotly object of 2 subplots
    """
    # y

    assert xx.shape[0] == zs[0].shape[1], \
        "expect the number of *columns* of zs elements to the same length of xx"
    assert yy.shape[0] == zs[0].shape[0], \
        "expect the number of *rows* of zs elements to the same length of xx"


    fig = make_subplots(
         rows=1, cols=2,
         horizontal_spacing=0.1,
         column_widths=[0.6, 0.4],
         specs=[[{"type": "scene"}, {"type": "xy"}]])

    counter_surface = 0
    for index in range(len(zs)):
        fig.add_trace(go.Surface(x=xx,
                                  y=yy,
                                  z=zs[index],
                                  colorscale="Viridis",
                                  showscale=False,
                                  opacity = .3), row=1, col=1)
        counter_surface += 1

    # --------
    # y slices
    # --------

    counter_yyy_slices = 0

    yyy_inner_storage = list()

    for yyy_idx, yyy in enumerate(yy):# slice based on y
        My = [0,yyy,0]
        M = My
        vy = [1,0,0]
        v = vy

        X,Y,Z =  get_plane_ben(M, v, "y", range_ben(xx),
                               range_ben(yy), range_ben(np.array(zs)))

        fig.add_trace(go.Surface(x=X.copy(), y=Y.copy(), z=Z.copy(),
                                 colorscale= [[0, "rgb(254, 254, 254)"],
                                              [1, "rgb(254, 254, 254)"]],
                                 showscale=keep_2d_legend,
                                 opacity =0.65,
                                 visible=False), row=1, col=1,
                                 )

        # per y associated splices of surfaces
        for index in range(len(zs)):
            yyy_inner_storage.append(zs[index][yyy_idx,:])
            ival = len(yyy_inner_storage)
            fig.add_trace(go.Scatter(x= xx,
                                     y = yyy_inner_storage[ival-1].copy(), mode="lines",
                                    visible=False), row=1, col=2)    # 2d lines

            counter_yyy_slices += 1

    # initializing y slice shown (col 1 and col 2)
    fig.data[counter_surface].visible = True
    for index in range(len(zs)):
        fig.data[counter_surface+1+index].visible = True


    # ---------------
    # create slider y
    # ---------------
    steps_yyy = []
    block_size = len(zs)+1
    for yyy_idx, yyy in enumerate(yy):
        step = dict(
            method="update",
            args=[{"visible": [True]*counter_surface +\
                               [False] * (len(fig.data)-counter_surface)},
                  #^ turth is for surfaces to always be shown, second is originally setting rest as FALSE
                  {"title": "Y slice: " + str(yyy)}],  # layout attribute
        )

        for index in range(len(zs)+1):
            step["args"][0]["visible"][counter_surface +\
                                       block_size*yyy_idx + index] = True  # Toggle index'th group to "visible"

        steps_yyy.append(step)


    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Y Slice: "},
        steps=steps_yyy
    )]


    fig.update_layout(
        sliders=sliders
    )

    fig['layout']['sliders'][0]['currentvalue']['prefix']='Y value: '
    for yyy_index, yyy in enumerate(yy):
        fig['layout']['sliders'][0]['steps'][yyy_index]['label'] = str(yyy)


    fig.update_layout(# figure info:
                      width=900, height=700,
                      title_text="Slicing a surface by a plane (Y)",
                      title_x=0.5,
                      # 3d info:
                      scene= {"camera": {"eye": {"x": 1.65*1.3, "z":0.75*1.3}},
                             "xaxis": dict(
                             backgroundcolor="rgba(0, 0, 0,0)",
                             gridcolor="rgba(0, 0, 0,.2)",
                             showbackground=True,
                             zerolinecolor="rgba(0, 0, 0,.5)"),
                             "yaxis": dict(
                             backgroundcolor="rgba(0, 0, 0,0)",
                             gridcolor="rgba(0, 0, 0,.2)",
                             showbackground=True,
                             zerolinecolor="rgba(0, 0, 0,.5)"),
                             "zaxis":dict(
                             backgroundcolor="rgba(0, 0, 0,0)",
                             gridcolor="rgba(0, 0, 0,.2)",
                             showbackground=True,
                             zerolinecolor="rgba(0, 0, 0,.5)")},

                      # 2d info:
                      yaxis = {"domain":  [0, 0.85],
                              "gridcolor":"rgba(0, 0, 0,.2)",
                              "zerolinecolor":"rgba(0, 0, 0,.3)"},
                      plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(
                             gridcolor="rgba(0, 0, 0,.2)",
                             zerolinecolor="rgba(0, 0, 0,.3)")
                  )

    fig.update_yaxes(range = range_ben(np.array(zs)),
                     title = "z value", row=1, col=2)
    fig.update_xaxes(title = "x value", row=1, col=2)

    fig.update(layout_showlegend=keep_2d_legend)

    return(fig)
