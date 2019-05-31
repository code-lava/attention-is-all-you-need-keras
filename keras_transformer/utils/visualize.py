from scipy.stats import mode
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools

def plot_attention_map(x_words, y_words, weights, average_weights=True, show_max_only=False):

    if average_weights:
        mean_weights = np.mean(weights, axis=-1)
        weights = [mean_weights]

    traces = []
    for weight_count, weight in  enumerate(weights):
        if show_max_only:
            weight = (weight == weight.max(axis=-1)[:, None]).astype(int)

        color_centroid = np.around(weights, decimals=4)
        color_centroid =  mode(color_centroid, axis=None)[0][0]
        print(color_centroid)
        traces.append(go.Heatmap(z=weight,
                                 zmin= color_centroid - np.var(weight)/100,
                                 zmax= color_centroid + np.var(weight)/100,
                                 x=list(x for x in range(len(x_words))),
                                 y=list(y for y in range(len(y_words))),
                                 showscale=True,
                                 colorbar=dict(x=0.45 if len(weights) > 1 and weight_count%2 == 0 else 1.0,
                                               # y=0.45 if weight_count%2 == 0 else 1, len= 0.45
                                               )
                                 ))
    if len(weights) == 1:
        layout = {}
        layout.update({'yaxis': {'ticktext': y_words,
                                 'tickvals': list(y for y in range(len(y_words))),
                                 'tickmode': 'array', 'autorange': 'reversed'}})
        layout.update({'xaxis': {'ticktext': x_words,
                                 'tickvals': list(x for x in range(len(x_words))),
                                 'tickmode': 'array',
                                 'tickangle': -90}})
        fig = go.Figure(traces, layout=layout)
    else:
        fig = tools.make_subplots(rows=(len(weights)+1)//2, cols=2, shared_yaxes=False, shared_xaxes=False, print_grid=False)

        layout = {'height': (18*len(y_words))*(len(weights)//2 + 1)}
        for trace_count, trace in enumerate(traces):
            fig.append_trace(trace, (trace_count//2)+1, (trace_count%2)+1)

            layout.update({'yaxis' + str(trace_count+1): {'ticktext': y_words,
                                                        'tickvals': list(y for y in range(len(y_words))),
                                                        'tickmode': 'array', 'autorange': 'reversed'}})
            layout.update({'xaxis' + str(trace_count+1): {'ticktext': x_words,
                                                        'tickvals': list(x for x in range(len(x_words))) ,
                                                        'tickmode':'array'}})

        fig['layout'].update(layout)
    py.plot(fig, image="svg")

# Place the blue cube on top of the green cube