"""
Sampling & Windowing Demo
Visualize sampling, windowing functions, and their effects on the DFT/DTFT
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ALL, ctx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass

dash.register_page(__name__, path='/sampling-windowing', name='Sampling & Windowing')


# ===================================================================
# Constants and Parameters
# ===================================================================

@dataclass
class SliderParams:
    """The parameters for a slider/input pair."""
    min: float
    max: float
    default: float
    step: float
    unit: str

# Parameters
FREQ_PARAMS = SliderParams(0.1, 10, 1, 0.1, '(Hz)')
AMPLITUDE_PARAMS = SliderParams(-10, 10, 1.0, 0.1, '')
PHASE_PARAMS = SliderParams(-180, 180, 0, 1, '(deg)')
TIME_HORIZON_PARAMS = SliderParams(0.1, 20, 2, 0.1, '(s)')
SAMPLE_RATE_PARAMS = SliderParams(0.1, 100, 10, 0.1, '(Hz)')
WINDOW_LENGTH_PARAMS = SliderParams(1, 500, 10, 1, '(samples)')

MAX_SINUSOIDS = 4
DTFT_ZERO_PAD_FACTOR = 100
OVERSAMPLE_FACTOR = 50
OVERTIME_FACTOR = 50
MAX_DISPLAY_POINTS = 5000

# Styling constants
CARD_STYLE = {
    'backgroundColor': '#1a1a2e',
    'borderRadius': '12px',
    'padding': '20px',
    'marginBottom': '20px',
    'border': '1px solid #2a2a4e',
}

LABEL_STYLE = {
    'color': '#ffffff',
    'fontSize': '0.9rem',
    'marginBottom': '5px',
    'display': 'block',
}

INPUT_STYLE = {
    'backgroundColor': '#0f0f23',
    'border': '1px solid #3a3a5e',
    'borderRadius': '6px',
    'color': '#ffffff',
    'padding': '8px 12px',
    'width': '100%',
    'fontSize': '0.9rem',
}

SLIDER_STYLE = {
    'marginTop': '10px',
}

# Window function options
WINDOW_OPTIONS = [
    "Rectangular",
    "Triangular",
    "Hann",
    "Hamming",
    "Blackman",
    # "Kaiser"
]

# ===================================================================
# UI and Layout
# ===================================================================

def create_input_with_slider(id_name, label, params: SliderParams):
    """Create an input box paired with a slider.
    
    Args:
        id_name: Simple string ID for the input/slider pair
        label: Display label
        params: SliderParams with min, max, default, step, unit
    """
    return html.Div([
        html.Label(f'{label} {params.unit}', style=LABEL_STYLE),
        html.Div([
            dcc.Input(
                id=f'input-{id_name}',
                type='number',
                value=params.default,
                min=params.min,
                max=params.max,
                step=params.step,
                style={**INPUT_STYLE, 'width': '80px', 'marginRight': '15px'}
            ),
            html.Div([
                dcc.Slider(
                    id=f'slider-{id_name}',
                    min=params.min,
                    max=params.max,
                    step=params.step,
                    value=params.default,
                    marks=None,
                    tooltip={'placement': 'bottom', 'always_visible': False},
                )
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={'marginBottom': '15px'})

def create_sinusoid_input_with_slider(index, param_name, label, params: SliderParams):
    """Create an input/slider pair for a sinusoid parameter.
    
    Uses pattern-matching IDs: {'type': 'sinusoid-input', 'param': param_name, 'index': index}

    TODO Determine if this actually needs to be its own function vs using create_input_with_slider
    could reduce code length greatly by reusing that function with slight modifications.
    """
    return html.Div([
        html.Label(f'{label} {params.unit}', style=LABEL_STYLE),
        html.Div([
            dcc.Input(
                id={'type': 'sinusoid-input', 'param': param_name, 'index': index},
                type='number',
                value=params.default,
                min=params.min,
                max=params.max,
                step=params.step,
                style={**INPUT_STYLE, 'width': '80px', 'marginRight': '15px'}
            ),
            html.Div([
                dcc.Slider(
                    id={'type': 'sinusoid-slider', 'param': param_name, 'index': index},
                    min=params.min,
                    max=params.max,
                    step=params.step,
                    value=params.default,
                    marks=None,
                    tooltip={'placement': 'bottom', 'always_visible': False},
                )
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={'marginBottom': '15px'})


def create_sinusoid_controls(index):
    """Create controls for a single sinusoid component."""
    return html.Div([
        html.Div([
            html.Span(f'Sinusoid {index + 1}', style={
                'color': '#ffffff',
                'fontWeight': '600',
                'fontSize': '1rem',
            })
        ], style={'marginBottom': '15px'}),
        create_sinusoid_input_with_slider(index, 'freq', 'Frequency', FREQ_PARAMS),
        create_sinusoid_input_with_slider(index, 'amp', 'Amplitude', AMPLITUDE_PARAMS),
        create_sinusoid_input_with_slider(index, 'phase', 'Phase', PHASE_PARAMS),
    ], style={
        'backgroundColor': '#252540',
        'borderRadius': '8px',
        'padding': '15px',
        'marginBottom': '10px',
    }, id={'type': 'sinusoid-control', 'index': index})

def create_left_column() -> html.Div:
    """Create the left column with controls."""
    return html.Div([
        html.Div([
            # Signal Parameters Card
            html.Div([
                html.H3('Signal Parameters', style={
                    'color': '#ffffff',
                        'marginTop': '0',
                        'marginBottom': '20px',
                        'fontSize': '1.2rem',
                    }),
                    
                    # Number of sinusoids
                    html.Div([
                        html.Label('Number of Sinusoids', style=LABEL_STYLE),
                        dcc.Dropdown(
                            id='num-sinusoids',
                            options=[{'label': str(i), 'value': i} for i in range(1, 5)],
                            value=1,
                            style={'backgroundColor': '#0f0f23'},
                            className='dark-dropdown'
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Sinusoid controls container
                    html.Div(id='sinusoid-controls-container'),
                    
                    html.Hr(style={'borderColor': '#3a3a5e', 'margin': '20px 0'}),
                    
                    # Time horizon
                    create_input_with_slider('time-horizon', 'Display Time Horizon', TIME_HORIZON_PARAMS),
                    
                    # Sampling rate
                    create_input_with_slider('sample-rate', 'Sampling Rate', SAMPLE_RATE_PARAMS),
                    
                ], style=CARD_STYLE),
                
                # Window Function Card
                html.Div([
                    html.H3('Window Function', style={
                        'color': '#ffffff',
                        'marginTop': '0',
                        'marginBottom': '20px',
                        'fontSize': '1.2rem',
                    }),
                    
                    html.Div([
                        html.Label('Window Type', style=LABEL_STYLE),
                        dcc.Dropdown(
                            id='window-type',
                            options=WINDOW_OPTIONS,
                            value='rectangular',
                            className='dark-dropdown'
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    create_input_with_slider('window-length', 'Window Length', WINDOW_LENGTH_PARAMS),
                    
                ], style=CARD_STYLE),
        ], style={
            'overflowY': 'auto',
            'paddingRight': '10px',
            'height': '100%'
        })
    ], style={
        'width': '350px', 
        'flexShrink': '0',
        'height': '100%'
    })

def create_right_column() -> html.Div:
    return html.Div([
        html.Div([
            # Continuous/Discrete Signal Plots
            html.Div([
                html.H3('Signal Analysis', style={
                    'color': '#ffffff',
                    'marginTop': '0',
                    'marginBottom': '15px',
                    'fontSize': '1.2rem',
                }),
                dcc.Graph(id='signal-plots', style={'height': '500px'}),
            ], style=CARD_STYLE),
            
            # Window Function Plots
            html.Div([
                html.H3('Window Function Analysis', style={
                    'color': '#ffffff',
                    'marginTop': '0',
                    'marginBottom': '15px',
                    'fontSize': '1.2rem',
                }),
                dcc.Graph(id='window-plots', style={'height': '400px'}),
            ], style=CARD_STYLE),
            
            # Windowed Signal Plots
            html.Div([
                html.H3('Windowed Signal Analysis', style={
                    'color': '#ffffff',
                    'marginTop': '0',
                    'marginBottom': '15px',
                    'fontSize': '1.2rem',
                }),
                dcc.Graph(id='windowed-signal-plots', style={'height': '500px'}),
            ], style=CARD_STYLE),
        ], style={
            'overflowY': 'auto',
            'paddingRight': '10px',
            'height': '100%'
        })
    ], style={
        'flex': '1', 
        'marginLeft': '0',
        'height': '100%'
    })


layout = html.Div([
    # Header container
    html.Div([
        # Back button
        html.Div([
            dcc.Link('← Back to Home', href='/', style={
                'color': '#6080ff',
                'textDecoration': 'none',
                'fontSize': '1rem',
            })
        ], style={'marginBottom': '20px'}),
        
        # Title
        html.H1('Sampling & Windowing Visualization', style={
            'color': '#ffffff',
            'fontSize': '2rem',
            'fontWeight': '700',
            'marginBottom': '30px',
        }),
    ], style={'padding': '30px', 'paddingBottom': '0'}),
    
    # Main content - two columns
    html.Div([
        create_left_column(),
        create_right_column()
    ], style={
        'display': 'flex', 
        'alignItems': 'flex-start',
        'height': 'calc(100vh - 180px)',
        'gap': '20px',
        'padding': '0 30px 30px 30px',
        'overflow': 'hidden'
    }),
    
    # Store for intermediate data
    dcc.Store(id='signal-data-store'),
], style={
    'backgroundColor': '#0f0f23',
    'height': '100vh',
    'display': 'flex',
    'flexDirection': 'column',
    'minHeight': '100vh',
    'overflow': 'hidden'
})


# ===================================================================
# Helper and Math Functions
# ===================================================================
def create_plotly_layout():
    """Create common plotly layout settings."""
    return {
        'paper_bgcolor': '#1a1a2e',
        'plot_bgcolor': '#0f0f23',
        'font': {'color': '#ffffff'},
        'xaxis': {
            'gridcolor': '#2a2a4e',
            'zerolinecolor': '#3a3a5e',
        },
        'yaxis': {
            'gridcolor': '#2a2a4e',
            'zerolinecolor': '#3a3a5e',
        },
        'margin': {'l': 60, 'r': 30, 't': 40, 'b': 40},
    }

def get_window_function(window_type, length):
    """Generate window function of specified type and length."""
    if window_type == 'Rectangular':
        main_lobe_width = 4 * np.pi / length
        return np.ones(length), main_lobe_width
    elif window_type == 'Triangular':
        main_lobe_width = 8 * np.pi / length
        return np.bartlett(length), main_lobe_width
    elif window_type == 'Hann':
        main_lobe_width = 8 * np.pi / length
        return np.hanning(length), main_lobe_width
    elif window_type == 'Hamming':
        main_lobe_width = 8 * np.pi / length
        return np.hamming(length), main_lobe_width
    elif window_type == 'Blackman':
        main_lobe_width = 12 * np.pi / length
        return np.blackman(length), main_lobe_width
    # elif window_type == 'Kaiser':
    #     return np.kaiser(length, beta=5)
    else:
        return np.zeros(length), length

def get_default_params():
    """Return dictionary with all default parameter values."""
    return {
        'sinusoids': [{'freq': FREQ_PARAMS.default, 'amp': AMPLITUDE_PARAMS.default, 'phase': PHASE_PARAMS.default}],
        'time_horizon': TIME_HORIZON_PARAMS.default,
        'sample_rate': SAMPLE_RATE_PARAMS.default,
        'window_type': 'Rectangular',
        'window_length': int(WINDOW_LENGTH_PARAMS.default),
    }

def parse_sinusoid_inputs(num_sinusoids):
    """
    Parse sinusoid inputs from ctx.inputs using pattern-matched IDs.
    
    Returns a list of sinusoid dicts with freq, amp, phase.
    TODO: This method is kinda funky, see if there's a cleaner way to do this.
    """
    sinusoids = []
    
    # Build sinusoids from ctx.inputs which contains all current input values
    # Keys are like: '{"index":0,"param":"freq","type":"sinusoid-input"}.value'
    for i in range(num_sinusoids):
        freq = None
        amp = None
        phase = None
        
        # Search through ctx.inputs for matching sinusoid params
        for key, value in ctx.inputs.items():
            if 'sinusoid-input' in key and f'"index":{i}' in key:
                if '"param":"freq"' in key:
                    freq = value
                elif '"param":"amp"' in key:
                    amp = value
                elif '"param":"phase"' in key:
                    phase = value
        
        sinusoids.append({
            'freq': freq if freq is not None else FREQ_PARAMS.default,
            'amp': amp if amp is not None else AMPLITUDE_PARAMS.default,
            'phase': phase if phase is not None else PHASE_PARAMS.default,
        })
    
    return sinusoids if sinusoids else [{'freq': FREQ_PARAMS.default, 'amp': AMPLITUDE_PARAMS.default, 'phase': PHASE_PARAMS.default}]


# ===================================================================
# Callbacks
# ===================================================================

# Callback to update sinusoid controls based on number selected
@callback(
    Output('sinusoid-controls-container', 'children'),
    Input('num-sinusoids', 'value')
)
def update_sinusoid_controls(num_sinusoids):
    if num_sinusoids is None:
        num_sinusoids = 1
    return [create_sinusoid_controls(i) for i in range(num_sinusoids)]


# Callback to sync sinusoid input and slider values
@callback(
    Output({'type': 'sinusoid-slider', 'param': ALL, 'index': ALL}, 'value'),
    Output({'type': 'sinusoid-input', 'param': ALL, 'index': ALL}, 'value'),
    Input({'type': 'sinusoid-slider', 'param': ALL, 'index': ALL}, 'value'),
    Input({'type': 'sinusoid-input', 'param': ALL, 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def sync_sinusoid_input_slider(slider_values, input_values):
    triggered_id = ctx.triggered_id
    
    if triggered_id is None:
        return slider_values, input_values
    
    if triggered_id['type'] == 'sinusoid-slider':
        return slider_values, slider_values
    else:
        return input_values, input_values


# Callbacks to sync global parameter inputs with their sliders
@callback(
    Output('slider-time-horizon', 'value'),
    Output('input-time-horizon', 'value'),
    Input('slider-time-horizon', 'value'),
    Input('input-time-horizon', 'value'),
    prevent_initial_call=True
)
def sync_time_horizon(slider_val, input_val):
    if ctx.triggered_id == 'slider-time-horizon':
        return slider_val, slider_val
    return input_val, input_val


@callback(
    Output('slider-sample-rate', 'value'),
    Output('input-sample-rate', 'value'),
    Input('slider-sample-rate', 'value'),
    Input('input-sample-rate', 'value'),
    prevent_initial_call=True
)
def sync_sample_rate(slider_val, input_val):
    if ctx.triggered_id == 'slider-sample-rate':
        return slider_val, slider_val
    return input_val, input_val


@callback(
    Output('slider-window-length', 'value'),
    Output('input-window-length', 'value'),
    Input('slider-window-length', 'value'),
    Input('input-window-length', 'value'),
    prevent_initial_call=True
)
def sync_window_length(slider_val, input_val):
    if ctx.triggered_id == 'slider-window-length':
        return slider_val, slider_val
    return input_val, input_val


# Central callback to parse all inputs and store in dcc.Store
@callback(
    Output('signal-data-store', 'data'),
    Input('num-sinusoids', 'value'),
    Input('window-type', 'value'),
    Input('input-time-horizon', 'value'),
    Input('input-sample-rate', 'value'),
    Input('input-window-length', 'value'),
    Input({'type': 'sinusoid-input', 'param': ALL, 'index': ALL}, 'value'),
)
def update_data_store(num_sinusoids, window_type, time_horizon, sample_rate, window_length, _sinusoid_values):
    """Parse all inputs and store as a clean dictionary."""
    defaults = get_default_params()
    
    # Handle None/missing values with defaults
    num_sinusoids = num_sinusoids if num_sinusoids is not None else 1
    
    params = {
        'sinusoids': parse_sinusoid_inputs(num_sinusoids),
        'time_horizon': time_horizon if time_horizon is not None else defaults['time_horizon'],
        'sample_rate': sample_rate if sample_rate is not None else defaults['sample_rate'],
        'window_type': window_type if window_type is not None else defaults['window_type'],
        'window_length': int(window_length) if window_length is not None else defaults['window_length'],
    }
    
    return params


# Main callback for signal plots
@callback(
    Output('signal-plots', 'figure'),
    Input('signal-data-store', 'data'),
)
def update_signal_plots(params):
    if params is None:
        params = get_default_params()
    
    # Extract parameters from store
    sinusoids = params['sinusoids']
    time_horizon = params['time_horizon']
    sample_rate = params['sample_rate']
    
    # Oversampling factor for continuous approximation
    continuous_sample_rate = sample_rate * OVERSAMPLE_FACTOR

    # Extend signal for for infinite duration approximation
    time_horizon_extended = time_horizon * OVERTIME_FACTOR
    
    # Generate time vectors
    t_continuous = np.arange(0, time_horizon, 1/continuous_sample_rate)
    t_discrete = np.arange(0, time_horizon, 1/sample_rate)
    t_continuous_extended = np.arange(0, time_horizon_extended, 1/continuous_sample_rate)
    t_discrete_extended = np.arange(0, time_horizon_extended, 1/sample_rate)
    
    # Generate composite signal
    signal_continuous = np.zeros_like(t_continuous)
    signal_discrete = np.zeros_like(t_discrete)
    signal_continuous_extended = np.zeros_like(t_continuous_extended)
    signal_discrete_extended = np.zeros_like(t_discrete_extended)
    
    for s in sinusoids:
        freq = s['freq']
        amp = s['amp']
        phase_rad = np.deg2rad(s['phase'])

        signal_continuous += amp * np.sin(2 * np.pi * freq * t_continuous + phase_rad)
        signal_discrete += amp * np.sin(2 * np.pi * freq * t_discrete + phase_rad)
        signal_continuous_extended += amp * np.sin(2 * np.pi * freq * t_continuous_extended + phase_rad)
        signal_discrete_extended += amp * np.sin(2 * np.pi * freq * t_discrete_extended + phase_rad)
    
    # TODO may need to zero pad lets see
    # Compute frequency spectra
    # For "continuous" signal - use high resolution and extended
    fourier_transform = np.fft.fft(signal_continuous_extended, len(signal_continuous_extended))
    fourier_transform_freqs = np.fft.fftfreq(len(signal_continuous_extended), 1/continuous_sample_rate)
    
    # For discrete signal
    discrete_time_fourier_transform = np.fft.fft(signal_discrete_extended, len(signal_discrete_extended))
    # discrete_time_fourier_transform_freqs = np.fft.fftfreq(len(signal_discrete_extended)) # -pi to pi normalized
    discrete_time_fourier_transform_freqs = np.linspace(-np.pi, np.pi, len(discrete_time_fourier_transform))
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Continuous Signal (Time Domain)',
            'Continuous Time Fourier Transform (Frequency Domain Approx)',
            'Sampled Signal (Time Domain)',
            'Discrete Time Fourier Transform (Frequency Domain Approx)',
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Continuous time domain signal
    step = max(1, len(t_continuous) // MAX_DISPLAY_POINTS)
    fig.add_trace(
        go.Scatter(
            x=t_continuous[::step],
            y=signal_continuous[::step],
            mode='lines',
            name='X(t)',
            line={'color': '#00ff88', 'width': 1.5}
        ),
        row=1, col=1
    )
    
    # Continuous frequency domain (magnitude, positive frequencies only)
    # (CTFTs)
    pos_mask_cont = fourier_transform_freqs >= 0
    max_freq_present = max([s['freq'] for s in sinusoids])
    pos_mask_cont &= fourier_transform_freqs <= max_freq_present * 2 # Limit to nyquist
    mag_continuous = np.abs(fourier_transform[pos_mask_cont])
    
    fig.add_trace(
        go.Scatter(
            x=fourier_transform_freqs[pos_mask_cont],
            y=mag_continuous,
            mode='lines',
            name='|X(f)|',
            line={'color': '#00ff88', 'width': 1.5}
        ),
        row=1, col=2
    )
    
    # Discrete time domain - show as stem plot
    fig.add_trace(
        go.Scatter(
            x=t_discrete,
            y=signal_discrete,
            mode='markers',
            name='X[n]',
            marker={'color': '#ff6088', 'size': 6}
        ),
        row=2, col=1
    )
    
    # Add stem lines
    for i, (t, y) in enumerate(zip(t_discrete, signal_discrete)):
        if i < 500:  # Limit for performance
            fig.add_trace(
                go.Scatter(
                    x=[t, t],
                    y=[0, y],
                    mode='lines',
                    line={'color': '#ff6088', 'width': 1},
                    showlegend=False
                ),
                row=2, col=1
            )
    
    fig.add_trace(
        go.Scatter(
            x=discrete_time_fourier_transform_freqs,
            y=np.abs(discrete_time_fourier_transform),
            mode='lines',
            name='|X[k]|',
            line={'color': '#ff6088', 'width': 1.5}
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        **create_plotly_layout(),
        showlegend=False,
        height=500,
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_xaxes(title_text='Frequency (Hz)', row=1, col=2)
    fig.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=2)
    
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Magnitude', row=1, col=2)
    fig.update_yaxes(title_text='Amplitude', row=2, col=1)
    fig.update_yaxes(title_text='Magnitude', row=2, col=2)
    
    return fig


# Callback for window function plots
@callback(
    Output('window-plots', 'figure'),
    Input('signal-data-store', 'data'),
)
def update_window_plots(params):
    if params is None:
        params = get_default_params()
    
    window_type = params['window_type']
    window_length = params['window_length']
    
    # Generate window
    window, _ = get_window_function(window_type, window_length)
    n = np.arange(window_length)
    
    # Compute DTFT of window
    n_fft = len(window) * DTFT_ZERO_PAD_FACTOR
    window_spectrum = np.fft.fft(window, n=n_fft)
    freqs = np.fft.fftfreq(n_fft)
    
    # Shift for centered view
    window_spectrum = np.fft.fftshift(window_spectrum)
    freqs = np.fft.fftshift(freqs)
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'{window_type.capitalize()} Window (Time Domain)',
            f'{window_type.capitalize()} Window (Frequency Domain - dB)'
        ),
        horizontal_spacing=0.12
    )
    
    # Time domain - stem plot
    fig.add_trace(
        go.Scatter(
            x=n,
            y=window,
            mode='markers+lines',
            name='w[n]',
            line={'color': '#ffaa00', 'width': 1},
            marker={'color': '#ffaa00', 'size': 5}
        ),
        row=1, col=1
    )
    
    # Frequency domain (dB scale)
    magnitude_db = 20 * np.log10(np.abs(window_spectrum) / np.max(np.abs(window_spectrum)) + 1e-10)
    
    fig.add_trace(
        go.Scatter(
            x=freqs * window_length,  # Normalize to show in terms of bins
            y=magnitude_db,
            mode='lines',
            name='|W(f)| dB',
            line={'color': '#ffaa00', 'width': 1.5}
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        **create_plotly_layout(),
        showlegend=False,
        height=400,
    )
    
    fig.update_xaxes(title_text='Sample (n)', row=1, col=1)
    fig.update_xaxes(title_text='Normalized Frequency (bins)', row=1, col=2, range=[-window_length/2, window_length/2])
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Magnitude (dB)', row=1, col=2, range=[-100, 5])
    
    return fig


# Callback for windowed signal plots
@callback(
    Output('windowed-signal-plots', 'figure'),
    Input('signal-data-store', 'data'),
)
def update_windowed_signal_plots(params):
    if params is None:
        params = get_default_params()
    
    # Extract all parameters from store
    sinusoids = params['sinusoids']
    time_horizon = params['time_horizon']
    sample_rate = params['sample_rate']
    window_type = params['window_type']
    window_length = params['window_length']
    
    # Generate signals
    t_discrete = np.arange(0, time_horizon, 1/sample_rate)
    # Oversampled version for "continuous" windowed signal visualization
    t_continuous = np.arange(0, time_horizon, 1/(sample_rate * OVERSAMPLE_FACTOR))
    # allocate signals
    signal_discrete = np.zeros_like(t_discrete)
    signal_continuous = np.zeros_like(t_continuous)
    
    # Calculate signals
    for s in sinusoids:
        freq = s['freq']
        amp = s['amp']
        phase_rad = np.deg2rad(s['phase'])
    
        signal_continuous += amp * np.sin(2 * np.pi * freq * t_continuous + phase_rad)
        signal_discrete += amp * np.sin(2 * np.pi * freq * t_discrete + phase_rad)
    
    # Apply window to beginning of signal
    window_discrete, _ = get_window_function(window_type, min(window_length, len(signal_discrete)))
    window_continuous, main_lobe_width_continuous = get_window_function(window_type, min(window_length * OVERSAMPLE_FACTOR, len(signal_continuous)))
    
    # Window signals - maintain full length, zero outside window
    windowed_signal_discrete = np.zeros_like(signal_discrete)
    windowed_signal_discrete[:len(window_discrete)] = signal_discrete[:len(window_discrete)] * window_discrete
    windowed_signal_continuous = np.zeros_like(signal_continuous)
    windowed_signal_continuous[:len(window_continuous)] = signal_continuous[:len(window_continuous)] * window_continuous
    
    # Compute DFT of windowed signal
    dtft = np.fft.fft(windowed_signal_discrete)
    # freqs_discrete_fourier_transform = np.fft.fftfreq(n_fft_dft, 1/sample_rate)
    freqs_dtft = np.linspace(-np.pi, np.pi, len(dtft))
    
    # Compute Fourier Transform of windowed signal (approx fourier transform)
    fourier_transform = np.fft.fft(windowed_signal_continuous, len(windowed_signal_continuous) * 100) # TODO fix this
    freqs_fourier_transform = np.fft.fftfreq(len(fourier_transform), 1/(sample_rate * OVERSAMPLE_FACTOR))
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Windowed Signal (Continuous Approx)',
            'Fourier Transform of Windowed Signal (approx)',
            'Windowed Signal (Discrete)',
            'DTFT of Windowed Signal',
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Continuous windowed signal (oversampled approximation)
    fig.add_trace(
        go.Scatter(
            x=t_continuous,
            y=windowed_signal_continuous,
            mode='lines',
            name='Windowed (cont)',
            line={'color': '#00ccff', 'width': 1.5}
        ),
        row=1, col=1
    )
    
    # Discrete windowed signal
    fig.add_trace(
        go.Scatter(
            x=t_discrete,
            y=windowed_signal_discrete,
            mode='markers',
            name='Windowed (disc)',
            marker={'color': '#ff66aa', 'size': 6}
        ),
        row=2, col=1
    )
    
    # Add stem lines for discrete
    for i, (t, y) in enumerate(zip(t_discrete, windowed_signal_discrete)):
        fig.add_trace(
            go.Scatter(
                x=[t, t],
                y=[0, y],
                mode='lines',
                line={'color': '#ff66aa', 'width': 1},
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Fourier Transform of windowed continuous signal

    pos_mask_cont = freqs_fourier_transform >= 0
    max_freq_present = max([s['freq'] for s in sinusoids])
    pos_mask_cont &= freqs_fourier_transform <= max_freq_present + (main_lobe_width_continuous * 50 * sample_rate) # Show multiple lobes

    fig.add_trace(
        go.Scatter(
            x=freqs_fourier_transform[pos_mask_cont],
            y=np.abs(fourier_transform[pos_mask_cont]),
            mode='lines',
            name='Fourier Transform',
            line={'color': '#00ccff', 'width': 1.5}
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=freqs_dtft,
            y=np.abs(dtft),
            mode='lines+markers',
            name='DFT',
            line={'color': '#ff66aa', 'width': 1},
            marker={'color': '#ff66aa', 'size': 4}
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        **create_plotly_layout(),
        showlegend=False,
        height=500,
    )
    
    fig.update_xaxes(title_text='Time (ms)', row=1, col=1)
    fig.update_xaxes(title_text='Time (ms)', row=1, col=2)
    fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=1)
    fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=2)
    
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Amplitude', row=1, col=2)
    fig.update_yaxes(title_text='Magnitude', row=2, col=1)
    fig.update_yaxes(title_text='Magnitude', row=2, col=2)
    
    return fig
