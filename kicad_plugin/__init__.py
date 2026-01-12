"""
AtoPlace KiCad Action Plugin

This plugin integrates AtoPlace's AI-powered PCB placement optimization
directly into KiCad's PCB Editor.

Installation:
    Symlink or copy this directory to KiCad's plugin search path:
    - macOS: ~/Library/Application Support/kicad/scripting/plugins/atoplace
    - Linux: ~/.local/share/kicad/scripting/plugins/atoplace
    - Windows: %APPDATA%/kicad/scripting/plugins/atoplace

Usage:
    Tools -> External Plugins -> AtoPlace: Optimize Placement
"""

# Register action plugins when this package is loaded by KiCad
try:
    from .atoplace_action import (
        AtoPlacePlaceAction,
        AtoPlaceValidateAction,
        AtoPlaceReportAction,
    )

    # Register all plugins
    AtoPlacePlaceAction().register()
    AtoPlaceValidateAction().register()
    AtoPlaceReportAction().register()

except Exception as e:
    # Log error but don't crash KiCad if plugin fails to load
    import sys
    print(f"[AtoPlace] Failed to register plugins: {e}", file=sys.stderr)
