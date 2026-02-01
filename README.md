# DSPVis - Digital Signal Processing Visualization

A web-based application for visualizing and exploring digital signal processing (DSP) concepts interactively. Built with Dash and Plotly, DSPVis provides an intuitive interface for understanding complex DSP operations through real-time visualization.

## Features

- **Interactive Visualizations**: Real-time DSP concept demonstrations
- **Sampling & Windowing**: Explore sampling theory and 

## Live Demo

Try it on [Hugging Face Spaces](https://huggingface.co/spaces/LucasPlant/DSPVis)

## Local Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LucasPlant/DSPVis.git
cd DSPVis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:8050`

## Requirements

- `dash>=2.14.0` - Web framework for building interactive dashboards
- `plotly>=5.18.0` - Interactive visualization library
- `numpy>=1.24.0` - Numerical computing library

## Project Structure

```
DSPVis/
├── app.py                    # Main application entry point
├── pages/
│   ├── home.py              # Home page
│   └── sampling_windowing.py # Sampling & windowing demonstrations
├── assets/
│   └── style.css            # Custom CSS styles
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Usage

Navigate through different DSP concepts using the sidebar menu. Each page offers interactive controls to explore various signal processing operations and their effects.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Lucas Plant** - [GitHub](https://github.com/LucasPlant)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the application.
