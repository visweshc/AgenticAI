# Train Project

A Python training project for learning and development.

## Setup Instructions

### Prerequisites
- Python 3.12 or higher
- `uv` package manager installed

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd c:/vishwa/Train
   ```

2. **Activate the virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   Or on Windows Command Prompt:
   ```cmd
   .\.venv\Scripts\activate.bat
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

### Running the Project

- Run the main script: `python main.py`
- Run the hello world example: `python hello_world.py`

### Project Structure

```
Train/
├── .venv/              # Virtual environment
├── .python-version     # Python version specification
├── pyproject.toml      # Project configuration
├── uv.lock            # Dependency lock file
├── README.md          # This file
├── main.py            # Main application script
└── hello_world.py     # Example script
```

## Development

### Adding Dependencies

To add new dependencies, use:
```bash
uv add package_name
```

### Running Tests

If you add tests, you can run them with:
```bash
uv run pytest
```

## License

Add your license information here.