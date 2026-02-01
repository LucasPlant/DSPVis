"""
Home Page - Landing page with demo selection buttons
"""

import dash
from dash import html, dcc

dash.register_page(__name__, path='/', name='Home')

# List of available demos
demos = [
    {
        'name': 'Sampling & Windowing',
        'path': '/sampling-windowing',
        'description': 'Explore sampling, windowing functions, and their effects on the DFT'
    },
    # Add more demos here as they are created
]

def create_demo_button(demo):
    """Create a styled button for a demo."""
    return html.A(
        html.Div([
            html.H3(demo['name'], style={
                'margin': '0 0 10px 0',
                'color': '#ffffff',
                'fontSize': '1.5rem',
                'fontWeight': '600',
            }),
            html.P(demo['description'], style={
                'margin': '0',
                'color': '#a0a0b0',
                'fontSize': '0.95rem',
            })
        ], style={
            'backgroundColor': '#1a1a2e',
            'border': '2px solid #2a2a4e',
            'borderRadius': '12px',
            'padding': '30px 40px',
            'cursor': 'pointer',
            'transition': 'all 0.3s ease',
            'minWidth': '300px',
            'maxWidth': '500px',
            'textAlign': 'center',
        }, className='demo-button'),
        href=demo['path'],
        style={'textDecoration': 'none'}
    )

layout = html.Div([
    # Header
    html.Div([
        html.H1('DSP Visualization Lab', style={
            'color': '#ffffff',
            'fontSize': '3rem',
            'fontWeight': '700',
            'marginBottom': '10px',
            'textAlign': 'center',
        }),
        html.P('Interactive demonstrations of Digital Signal Processing concepts', style={
            'color': '#8080a0',
            'fontSize': '1.2rem',
            'textAlign': 'center',
            'marginBottom': '60px',
        })
    ], style={'paddingTop': '80px'}),
    
    # Demo buttons container
    html.Div([
        create_demo_button(demo) for demo in demos
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'gap': '20px',
    })
], style={
    'minHeight': '100vh',
    'backgroundColor': '#0f0f23',
    'padding': '20px',
})
