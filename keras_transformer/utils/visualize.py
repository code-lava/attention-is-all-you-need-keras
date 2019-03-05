import plotly.offline as py
import plotly.graph_objs as go

def plot_attention_map(x_words, y_words, weights):
    trace = go.Heatmap(z=weights,
                       x=x_words,
                       y=y_words)
    data = [trace]
    layout = go.Layout(
        yaxis=dict(ticks='', autorange='reversed',  tickmode='auto', nticks=len(y_words)+1),
        xaxis=dict(ticks='', tickangle=90, nticks=len(x_words)+1, tickmode='auto', type='category')
    )
    py.plot(go.Figure(data=data, layout=layout))