import numpy as np
import plotly.graph_objects as go

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
