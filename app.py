"""
DSP Visualization Application
A Dash-based web application for demonstrating DSP concepts.
"""

import dash
from dash import Dash, html, dcc, page_container

LOCAL = False

# Initialize the Dash app with multi-page support
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
    ],
    assets_folder='assets'
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

# Export the server for gunicorn/production deployment
server = app.server

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 7860))
    if LOCAL:
        app.run(debug=True, host='127.0.0.1', port=port)
    else:
        app.run(debug=False, host='0.0.0.0', port=port)