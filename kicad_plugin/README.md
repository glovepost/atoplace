# AtoPlace KiCad Plugin

This plugin integrates AtoPlace's AI-powered PCB placement optimization directly into KiCad's PCB Editor.

## Features

- **Optimize Placement**: Run force-directed placement with automatic legalization
- **Validate Placement**: Check for pre-route issues and DRC violations
- **Generate Report**: Create detailed placement quality reports

## Installation

### 1. Install AtoPlace Package

First, ensure the atoplace package is installed in KiCad's Python environment:

```bash
# macOS
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install -e /path/to/atoplace

# Linux
python3 -m pip install -e /path/to/atoplace

# Windows
"C:\Program Files\KiCad\bin\python.exe" -m pip install -e C:\path\to\atoplace
```

### 2. Install the Plugin

#### Option A: Symlink (Recommended for Development)

```bash
# macOS
ln -s /path/to/atoplace/kicad_plugin ~/Library/Application\ Support/kicad/8.0/scripting/plugins/atoplace

# Linux
ln -s /path/to/atoplace/kicad_plugin ~/.local/share/kicad/8.0/scripting/plugins/atoplace

# Windows (run as Administrator)
mklink /D "%APPDATA%\kicad\8.0\scripting\plugins\atoplace" "C:\path\to\atoplace\kicad_plugin"
```

#### Option B: Copy

Copy the entire `kicad_plugin` directory to your KiCad plugins folder and rename it to `atoplace`.

### 3. Refresh Plugins

In KiCad's PCB Editor:
1. Go to **Tools → External Plugins → Refresh Plugins**
2. The AtoPlace actions should now appear in **Tools → External Plugins**

## Usage

### Optimize Placement

1. Open a PCB in KiCad's PCB Editor
2. Go to **Tools → External Plugins → AtoPlace: Optimize Placement**
3. The plugin will:
   - Run force-directed refinement to optimize component positions
   - Apply legalization (grid snapping, row alignment, overlap removal)
   - Save changes to the board file

### Validate Placement

1. Go to **Tools → External Plugins → AtoPlace: Validate Placement**
2. The plugin will run pre-route validation and DRC checks
3. Results are displayed in a summary dialog

### Generate Report

1. Go to **Tools → External Plugins → AtoPlace: Generate Report**
2. Choose where to save the markdown report
3. The report includes:
   - Pre-route validation results
   - DRC violations
   - Confidence scoring
   - Detected functional modules

## Plugin Paths

KiCad searches for plugins in these directories:

- **macOS**: `~/Library/Application Support/kicad/8.0/scripting/plugins/`
- **Linux**: `~/.local/share/kicad/8.0/scripting/plugins/`
- **Windows**: `%APPDATA%\kicad\8.0\scripting\plugins\`

To find your exact plugin directory, run in KiCad's Python console:
```python
import pcbnew
print(pcbnew.PLUGIN_DIRECTORIES_SEARCH)
```

## Troubleshooting

### "AtoPlace package not found"

Ensure atoplace is installed in KiCad's Python environment, not your system Python:

```bash
# Check which Python KiCad uses
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -c "import sys; print(sys.executable)"

# Install atoplace there
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install -e /path/to/atoplace
```

### Plugin not appearing in menu

1. Check that the plugin directory is correctly named and placed
2. Refresh plugins: **Tools → External Plugins → Refresh Plugins**
3. Check KiCad's scripting console for error messages

### "Board has not been saved"

The plugin requires the board to be saved to disk before running. Save your board first with **File → Save**.

## Development

For development, symlink the plugin directory so changes take effect immediately after refreshing plugins.

To debug, use KiCad's scripting console (**Tools → Scripting Console**) to test imports:

```python
import sys
sys.path.insert(0, "/path/to/atoplace")
from atoplace.placement.force_directed import ForceDirectedRefiner
print("Import successful!")
```
