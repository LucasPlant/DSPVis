"""
DSP Visualization Application
A Dash-based web application for demonstrating DSP concepts.
"""

import dash
from dash import Dash, html, dcc, page_container

# Initialize the Dash app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
    ]
)

# Main layout with page container
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    page_container
], style={
    'fontFamily': 'Inter, sans-serif',
    'backgroundColor': '#0f0f23',
    'minHeight': '100vh',
})

if __name__ == '__main__':
    app.run(debug=True, port=8050)
