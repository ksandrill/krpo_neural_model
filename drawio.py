import os

import plotly
import plotly.graph_objs as go


def draw_graph(value_array: list[float], x_name: str, y_name: str, smth: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i + 1 for i in range(len(value_array))], y=value_array))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title=smth,
                      xaxis_title=x_name,
                      yaxis_title=y_name,
                      margin=dict(l=0, r=0, t=30, b=0))

    fig.show()
    return fig


def save_param_to_html(fig: go.Figure, path_to_save: str, param_name: str):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plotly.io.write_html(fig=fig, file=os.path.join(path_to_save, param_name + ".html"))


def draw_model_real(model_out: list[float], real_out: list[float], name: str, x_axis_name: str, y_axis_name: str,
                    real_color: str, model_color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(len(model_out))], y=model_out, name="model " + name,
                             line=dict(color=model_color)))
    fig.add_trace(go.Scatter(x=[i for i in range(len(model_out))], y=real_out, name="real " + name,
                             line=dict(color=real_color)))
    fig.update_traces(showlegend=True)
    fig.update_layout(legend_orientation="h", title=name, xaxis_title=x_axis_name, yaxis_title=y_axis_name)
    fig.show()
    return fig