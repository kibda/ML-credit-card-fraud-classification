import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load the data
df = pd.read_csv('data/card_transdata.csv')


# Initialize the app
app = Dash(__name__)
app.title = "Credit Card Fraud Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Credit Card Fraud Detection Dashboard", style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns if col != 'fraud'],
        value='distance_from_home',
        style={'width': '60%', 'margin': 'auto'}
    ),

    dcc.Graph(id='feature-distribution'),

    dcc.Graph(id='fraud-pie'),

    dcc.Graph(id='correlation-heatmap')
])

# Callbacks
@app.callback(
    Output('feature-distribution', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_distribution(feature):
    fig = px.histogram(df, x=feature, color=df['fraud'].map({0: 'Non-Fraud', 1: 'Fraud'}),
                       barmode='overlay', nbins=50,
                       title=f'Distribution of {feature} by Fraud Status')
    return fig

@app.callback(
    Output('fraud-pie', 'figure'),
    Input('feature-dropdown', 'value')  # Dummy input just to trigger update
)
def update_pie(_):
    counts = df['fraud'].value_counts()
    fig = px.pie(names=['Non-Fraud', 'Fraud'], values=counts, title='Fraud vs Non-Fraud Distribution')
    return fig

@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('feature-dropdown', 'value')  # Dummy input just to trigger update
)
def update_heatmap(_):
    corr = df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(title="Feature Correlation Heatmap", height=600)
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)



