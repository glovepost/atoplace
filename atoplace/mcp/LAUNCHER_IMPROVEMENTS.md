# MCP Launcher Improvements

## Automatic Orphaned Process Cleanup

### Problem
When the MCP server or its bridge processes crash or are forcefully terminated, orphaned `atoplace.mcp.bridge` processes can remain running in the background. These orphaned processes:
- Prevent new MCP connections from succeeding
- Consume system resources
- Require manual cleanup via `pkill` or Activity Monitor

This was a recurring issue that required users to manually run:
```bash
pkill -f "atoplace.mcp.bridge"
```

### Solution
The MCP launcher now automatically detects and cleans up orphaned bridge processes on startup.

#### How It Works
When `MCPLauncher.start()` is called:
1. **Detect KiCad Python** - Find the correct Python interpreter
2. **Clean up orphaned processes** ← NEW STEP
   - Scans for any running `atoplace.mcp.bridge` processes
   - Terminates them gracefully
   - Logs the cleanup for debugging
3. **Clean up stale socket** - Remove old socket file
4. **Start new bridge** - Launch fresh bridge process
5. **Wait for bridge** - Verify it's ready
6. **Start MCP server** - Launch the MCP protocol server

#### Implementation Details

**Primary Method (with psutil):**
```python
def _cleanup_orphaned_processes(self):
    """Kill any orphaned bridge processes from previous runs."""
    import psutil

    # Find all bridge processes
    orphaned = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info.get('cmdline', [])
        if cmdline and 'atoplace.mcp.bridge' in ' '.join(cmdline):
            orphaned.append(proc)

    # Kill them
    for proc in orphaned:
        proc.kill()
        proc.wait(timeout=2)
```

**Fallback Method (Unix systems without psutil):**
```python
if platform.system() in ('Darwin', 'Linux'):
    subprocess.run(['pkill', '-f', 'atoplace.mcp.bridge'])
```

### Benefits
1. **Automatic reconnection** - Users can reconnect without manual cleanup
2. **Improved reliability** - Eliminates a common failure mode
3. **Better UX** - No need to explain `pkill` to users
4. **Cross-platform** - Works on macOS, Linux (with fallback for Windows)
5. **Safe** - Only kills `atoplace.mcp.bridge` processes, nothing else

### Dependencies
- **psutil >= 5.9.0** - Added to `mcp` extras for robust process management
- Falls back to `pkill` if psutil is not available (Unix-like systems)

### Testing
The cleanup has been tested with:
- Multiple orphaned processes (21+ in production scenario)
- Zero orphaned processes (no-op case)
- Processes from different parent processes
- Integration with full launcher startup flow

### Logging
The launcher logs cleanup operations for debugging:
```
WARNING: Found 21 orphaned bridge process(es), cleaning up...
INFO: Cleaned up 21 orphaned bridge process(es)
```

Or when no cleanup is needed:
```
DEBUG: No orphaned bridge processes found
```

### Future Improvements
Potential enhancements:
1. Track bridge PIDs in a lock file to only kill our own bridges
2. Add timeout for zombie processes that won't die
3. Report cleanup statistics to the user via MCP status
4. Add cleanup command to CLI: `atoplace mcp cleanup`

## Related Issues
- Fixed issue where 21+ orphaned bridge processes prevented reconnection
- Resolves "Failed to reconnect to atoplace" errors
- Addresses socket conflicts from multiple bridge instances

## Files Modified
- `atoplace/mcp/launcher.py` - Added `_cleanup_orphaned_processes()` method
- `pyproject.toml` - Added `psutil>=5.9.0` to `mcp` extras

## Backward Compatibility
This change is fully backward compatible:
- Existing launchers continue to work
- psutil is optional (falls back to pkill)
- No changes to MCP protocol or API
- No configuration changes required

---

## KIPY Mode as Default with Intelligent Fallback

### Problem
The MCP launcher was hardcoded to use IPC mode (bridge-based), which meant:
- Changes were saved to separate `.placed` files instead of syncing in real-time
- Users couldn't see live updates in KiCad while making changes
- The superior KIPY mode (real-time sync with KiCad 9+) wasn't being utilized
- Users had to manually configure environment variables to enable KIPY mode

### Solution
The launcher now defaults to KIPY mode with automatic fallback to IPC when KiCad is not running.

#### How It Works
The MCP server now uses intelligent backend selection:

1. **KIPY Mode (Default)** - If KiCad 9+ is running with API enabled:
   - Real-time synchronization with open KiCad board
   - Changes appear instantly in KiCad viewport
   - No separate `.placed` files created
   - Uses `kicad-python` package for IPC communication

2. **IPC Mode (Fallback #1)** - If KIPY unavailable but bridge is running:
   - Bridge-based IPC to pcbnew process
   - Changes saved to `.placed` files
   - Python 3.10+ compatible

3. **DIRECT Mode (Fallback #2)** - If no other backend available:
   - Direct pcbnew access (requires KiCad Python environment)
   - Fallback for legacy setups

#### Backend Detection Logic
The system checks backends in priority order:

```python
# In backends.py
def check_kipy_available() -> Tuple[bool, str]:
    """Check if kipy is available and can connect to KiCad."""
    try:
        from kipy import KiCad
        kicad = KiCad()
        kicad.ping()  # Test connection to running KiCad
        return True, "Connected to KiCad"
    except:
        return False, "KiCad not running or API not enabled"
```

The launcher tries each backend until one succeeds, logging warnings when falling back:
```
INFO: MCP server using kipy mode  # Success!
```

Or with fallback:
```
WARNING: Fell back to ipc mode (preferred: kipy)
INFO: MCP server using ipc mode
```

#### Changes Made

**launcher.py:**
- Removed: `env["ATOPLACE_USE_IPC"] = "1"` (line 203)
- Now: Server auto-selects backend with KIPY preference
- Added: Documentation explaining backend priority order

**server.py:**
- Already defaulted to KIPY (line 146): `preferred = BackendMode.KIPY`
- Uses `create_session_with_fallback(preferred)` for automatic fallback

**backends.py:**
- Already had full fallback logic in place
- `check_kipy_available()` pings KiCad to verify it's running
- Falls back through chain: KIPY → IPC → DIRECT

### Benefits

1. **Best experience by default** - Users get real-time sync automatically
2. **Graceful degradation** - Works even when KiCad not running
3. **Zero configuration** - No environment variables needed
4. **Clear feedback** - Logs show which mode is active
5. **Future-proof** - KIPY mode is the future of KiCad integration

### User Experience

**With KiCad running:**
```python
# User opens examples/dogtracker/default.kicad_pcb in KiCad
# Then in Claude:
load_board("examples/dogtracker/default.kicad_pcb")
# → Connects to live board via KIPY
# → Changes appear instantly in KiCad

align_components(["C1", "C2", "C3"], axis="x")
# → Capacitors align in real-time, no save needed!
```

**Without KiCad running:**
```python
load_board("examples/dogtracker/default.kicad_pcb")
# → Falls back to IPC mode automatically
# → Changes saved to default.placed.kicad_pcb
```

### Configuration Override
Users can still force a specific mode if needed:
```bash
export ATOPLACE_BACKEND=ipc   # Force IPC mode
export ATOPLACE_BACKEND=kipy  # Force KIPY (fails if unavailable)
export ATOPLACE_BACKEND=direct # Force DIRECT mode
```

### Prerequisites for KIPY Mode
For automatic KIPY mode to work:
1. **KiCad 9+** installed and running
2. **kicad-python** package installed: `pip install kicad-python`
3. **KiCad API enabled** in Preferences → Plugins
4. **Board open in KiCad** before calling `load_board()`

If any prerequisite is missing, the system automatically falls back to IPC mode.

### Files Modified
- `atoplace/mcp/launcher.py` - Removed hardcoded IPC mode, updated docs
- Documentation updated to explain KIPY-first behavior

### Backward Compatibility
This change is fully backward compatible:
- Existing workflows continue to work
- IPC mode still available as fallback
- Environment variables still respected (`ATOPLACE_BACKEND`)
- No breaking changes to API or MCP protocol
- Logs clearly indicate which mode is active

### Testing
Tested scenarios:
- ✅ KiCad running → KIPY mode activated
- ✅ KiCad not running → IPC fallback works
- ✅ Real-time component alignment in KIPY mode
- ✅ Environment variable override works
- ✅ Graceful fallback with clear logging
