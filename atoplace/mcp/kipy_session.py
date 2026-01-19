"""
KiPy Session - Live KiCad Integration via Official IPC API.

This module provides a Session-compatible interface that communicates
directly with a running KiCad 9+ instance using the official KiCad
IPC API (kicad-python/kipy).

Key features:
- Real-time component manipulation (instant visual updates)
- Integration with KiCad's native undo/redo system
- No save/reload cycles required
- Transaction support for grouping operations

Requirements:
- KiCad 9.0+ running with API enabled (Preferences > Plugins)
- kicad-python package (pip install kicad-python)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..board.abstraction import Board

logger = logging.getLogger(__name__)


class KiCadNotRunningError(Exception):
    """KiCad is not running or IPC API is not available."""

    pass


class KiCadConnectionLostError(Exception):
    """Connection to KiCad was lost during operation."""

    pass


class KiPySession:
    """
    Session implementation using kipy for live KiCad communication.

    Provides the same interface as Session/IPCSession, enabling seamless
    switching between backends. All component modifications are immediately
    reflected in KiCad's UI without requiring file save/reload.

    Usage:
        session = KiPySession()
        session.connect()
        session.load()  # Load currently open board from KiCad
        session.update_component("C6", x=10.0, y=15.0, rotation=90.0)
        # Component moves instantly in KiCad!
    """

    MAX_RECONNECT_ATTEMPTS = 3

    def __init__(self):
        """Initialize KiPySession."""
        self._kicad = None
        self._kipy_board = None
        self.board: Optional[Board] = None
        self.source_path: Optional[Path] = None
        self._connected = False
        self._dirty = False
        self._dirty_refs: set = set()
        self._current_commit = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_loaded(self) -> bool:
        """Check if a board is loaded."""
        return self.board is not None and self._connected

    @property
    def is_dirty(self) -> bool:
        """Check if board has unsaved changes."""
        return self._dirty

    @property
    def dirty_components(self) -> List[str]:
        """Get list of modified component references."""
        return list(self._dirty_refs)

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> bool:
        """
        Connect to running KiCad instance.

        Tries multiple socket paths for cross-platform support.

        Returns:
            True if connection successful
        """
        if self._connected:
            return True

        try:
            from kipy import KiCad
        except ImportError:
            logger.error("kipy not installed. Install with: pip install kicad-python")
            return False

        # Try multiple socket paths
        socket_paths = self._get_socket_paths()
        last_error = None

        for path in socket_paths:
            try:
                if path:
                    self._kicad = KiCad(socket_path=path)
                else:
                    self._kicad = KiCad()

                # Verify connection
                self._kicad.ping()
                self._connected = True

                socket_desc = path or "auto-detected"
                logger.info("Connected to KiCad via %s", socket_desc)
                return True

            except Exception as e:
                last_error = e
                logger.debug("Failed to connect via %s: %s", path, e)
                continue

        logger.error("Could not connect to KiCad: %s", last_error)
        return False

    def disconnect(self):
        """Disconnect from KiCad."""
        self._kicad = None
        self._kipy_board = None
        self._connected = False
        self._current_commit = None

    def ensure_connected(self) -> bool:
        """Ensure connection is established, reconnecting if needed."""
        if self._connected:
            return True
        return self.connect()

    def _get_socket_paths(self) -> list:
        """Get list of socket paths to try."""
        # Use centralized socket path logic from backends module
        from .backends import _get_socket_paths

        return _get_socket_paths()

    # =========================================================================
    # Board Operations
    # =========================================================================

    def load(self, path: Path = None) -> "KiPySession":
        """
        Load board from running KiCad instance.

        If path is provided, it's stored as source_path but the board
        data is always read from KiCad's currently open board.

        Args:
            path: Optional path to associate with the board

        Returns:
            Self for chaining
        """
        if not self.ensure_connected():
            raise KiCadNotRunningError(
                "Cannot connect to KiCad. Ensure KiCad 9+ is running "
                "with API enabled (Preferences > Plugins)."
            )

        try:
            # Get board from running KiCad
            self._kipy_board = self._kicad.get_board()

            # Convert to atoplace Board model
            from .kipy_adapter import kipy_board_to_atoplace

            self.board = kipy_board_to_atoplace(self._kipy_board)

            # Store source path
            if path:
                self.source_path = Path(path)
            elif hasattr(self._kipy_board, "file_path") and self._kipy_board.file_path:
                self.source_path = Path(self._kipy_board.file_path)

            self._dirty = False
            self._dirty_refs.clear()

            logger.info(
                "Loaded board from KiCad: %d components", len(self.board.components)
            )
            return self

        except Exception as e:
            raise RuntimeError(f"Failed to load board from KiCad: {e}")

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save board via KiCad.

        In kipy mode, changes are applied directly to KiCad. Changes are
        synced to KiCad's memory, then user saves via Ctrl+S (Cmd+S on Mac).

        Args:
            path: Output path. If provided, user is instructed to use
                  KiCad's "File > Save As" menu for Save As functionality,
                  since KIPY mode works with the live KiCad instance.

        Returns:
            Path where board should be saved
        """
        if not self.board or not self._connected:
            raise ValueError("No board loaded or not connected")

        # Sync any pending local changes to KiCad
        self._sync_to_kicad()

        if path is not None:
            # Explicit path provided - but KIPY mode can't do Save As directly
            # User needs to use KiCad's native Save As
            logger.warning(
                "KIPY mode: Save As not supported. Use KiCad's 'File > Save As' menu. "
                "Changes are synced to KiCad."
            )
            raise ValueError(
                "Save As not supported in KIPY mode. "
                "Your changes are synced to KiCad. "
                "Use 'File > Save As' in KiCad to save to a different path."
            )
        else:
            # No path - changes synced to KiCad, user saves via Ctrl+S
            logger.info("Changes synced to KiCad. Use Ctrl+S (Cmd+S on Mac) to save.")
            self._dirty = False
            return self.source_path or Path("unknown")

    # =========================================================================
    # Undo/Redo (Delegates to KiCad)
    # =========================================================================

    def checkpoint(self, description: str = ""):
        """
        Create checkpoint for undo.

        In kipy mode, each update_component call creates its own commit
        with the description, so this is largely a no-op. Called for
        API compatibility.
        """
        # In kipy mode, commits are created per-operation
        pass

    def undo(self) -> bool:
        """
        Undo via KiCad's undo system.

        Note: kipy may not expose undo API directly. This refreshes
        the local model from KiCad's current state.
        """
        if not self._connected:
            return False

        try:
            # Refresh local model from KiCad
            self._refresh_from_kicad()
            return True
        except Exception as e:
            logger.error("Undo/refresh failed: %s", e)
            return False

    def redo(self) -> bool:
        """
        Redo via KiCad's redo system.

        Note: kipy may not expose redo API directly. This refreshes
        the local model from KiCad's current state.
        """
        if not self._connected:
            return False

        try:
            self._refresh_from_kicad()
            return True
        except Exception as e:
            logger.error("Redo/refresh failed: %s", e)
            return False

    # =========================================================================
    # Component Updates (Real-time)
    # =========================================================================

    def update_component(
        self,
        ref: str,
        x: float = None,
        y: float = None,
        rotation: float = None,
        locked: bool = None,
    ) -> bool:
        """
        Update component position/rotation in KiCad with instant refresh.

        This is the key method for real-time manipulation. Changes appear
        immediately in KiCad's UI and are added to KiCad's undo stack.

        Args:
            ref: Component reference designator
            x: New X position in mm (optional)
            y: New Y position in mm (optional)
            rotation: New rotation in degrees (optional)
            locked: Lock state (optional)

        Returns:
            True if update successful
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        if not self.board:
            raise ValueError("No board loaded. Call load() first.")

        from .kipy_adapter import find_kipy_footprint

        # Find the footprint in kipy
        fp = find_kipy_footprint(self._kipy_board, ref)
        if not fp:
            logger.error("Component not found in KiCad: %s", ref)
            return False

        # Import kipy geometry types
        try:
            from kipy.geometry import Angle, Vector2
            from kipy.util.units import from_mm
        except ImportError:
            from kipy.geometry import Angle, Vector2

            from .kipy_adapter import mm_to_nm as from_mm

        # Begin commit for undo support
        commit = self._kipy_board.begin_commit()

        try:
            # Update position if provided
            if x is not None or y is not None:
                current_pos = fp.position
                new_x = from_mm(x) if x is not None else current_pos.x
                new_y = from_mm(y) if y is not None else current_pos.y
                fp.position = Vector2.from_xy(new_x, new_y)

            # Update rotation if provided
            if rotation is not None:
                fp.orientation = Angle.from_degrees(rotation)

            # Update locked state if provided
            if locked is not None and hasattr(fp, "locked"):
                fp.locked = locked

            # Apply changes to KiCad (triggers UI refresh)
            self._kipy_board.update_items([fp])
            self._kipy_board.push_commit(commit, f"Move {ref}")

            # Update local model
            if ref in self.board.components:
                comp = self.board.components[ref]
                if x is not None:
                    comp.x = x
                if y is not None:
                    comp.y = y
                if rotation is not None:
                    comp.rotation = rotation
                if locked is not None:
                    comp.locked = locked

            self._dirty = True
            self._dirty_refs.add(ref)

            logger.debug("Updated %s via kipy (instant)", ref)
            return True

        except Exception as e:
            # Rollback on error
            try:
                self._kipy_board.drop_commit(commit)
            except Exception:
                pass
            logger.error("Failed to update %s: %s", ref, e)
            return False

    def update_components(self, updates: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Batch update multiple components in a single transaction.

        All updates are grouped into a single undo step.

        Args:
            updates: List of dicts with keys: ref, x, y, rotation, locked

        Returns:
            Dict mapping ref -> success boolean
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        if not self.board:
            raise ValueError("No board loaded. Call load() first.")

        from .kipy_adapter import find_kipy_footprints

        # Find all footprints
        refs = [u.get("ref") for u in updates if u.get("ref")]
        if not refs:
            return {}

        footprints = find_kipy_footprints(self._kipy_board, refs)

        # Import kipy types
        try:
            from kipy.geometry import Angle, Vector2
            from kipy.util.units import from_mm
        except ImportError:
            from kipy.geometry import Angle, Vector2

            from .kipy_adapter import mm_to_nm as from_mm

        results = {}
        modified_fps = []

        # Process updates first, then commit only if we have changes
        for update in updates:
            ref = update.get("ref")
            if not ref:
                continue

            fp = footprints.get(ref)
            if not fp:
                results[ref] = False
                continue

            # Apply updates
            x = update.get("x")
            y = update.get("y")
            rotation = update.get("rotation")
            locked = update.get("locked")

            if x is not None or y is not None:
                current_pos = fp.position
                new_x = from_mm(x) if x is not None else current_pos.x
                new_y = from_mm(y) if y is not None else current_pos.y
                fp.position = Vector2.from_xy(new_x, new_y)

            if rotation is not None:
                fp.orientation = Angle.from_degrees(rotation)

            if locked is not None and hasattr(fp, "locked"):
                fp.locked = locked

            modified_fps.append(fp)

            # Update local model
            if ref in self.board.components:
                comp = self.board.components[ref]
                if x is not None:
                    comp.x = x
                if y is not None:
                    comp.y = y
                if rotation is not None:
                    comp.rotation = rotation
                if locked is not None:
                    comp.locked = locked

            results[ref] = True
            self._dirty_refs.add(ref)

        # Only begin commit if we have changes to apply
        if not modified_fps:
            return results

        commit = self._kipy_board.begin_commit()
        try:
            self._kipy_board.update_items(modified_fps)
            self._kipy_board.push_commit(commit, f"Move {len(modified_fps)} components")
            self._dirty = True
            return results

        except Exception as e:
            # Rollback on error
            try:
                self._kipy_board.drop_commit(commit)
            except Exception:
                pass
            logger.error("Batch update failed: %s", e)
            return dict.fromkeys(refs, False)

    def mark_modified(self, refs: List[str]):
        """
        Mark components as modified and immediately sync to KiCad.

        In kipy mode, this triggers an instant sync for live feedback.
        In other modes, just marks dirty for later save.
        """
        self._dirty = True
        self._dirty_refs.update(refs)

        # Immediately sync to KiCad for live updates
        if self._connected and self._kipy_board:
            try:
                self._sync_to_kicad()
            except Exception as e:
                logger.warning("Failed to auto-sync to KiCad: %s", e)

    def clear_dirty(self):
        """Clear dirty state."""
        self._dirty_refs.clear()

    # =========================================================================
    # Track Management
    # =========================================================================

    def clear_unlocked_tracks(self, include_vias: bool = True) -> int:
        """
        Clear all unlocked tracks and optionally vias from the board.

        When running placement optimization, existing tracks become invalid
        as components move. This method removes all unlocked tracks and vias
        to prepare for re-routing.

        Args:
            include_vias: Also clear unlocked vias (default: True)

        Returns:
            Number of items deleted (tracks + vias)
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        try:
            items_to_remove = []

            # Get all tracks from the board
            tracks = list(self._kipy_board.get_tracks())
            unlocked_tracks = [t for t in tracks if not getattr(t, "locked", False)]
            items_to_remove.extend(unlocked_tracks)

            # Get all vias if requested
            vias_removed = 0
            if include_vias:
                try:
                    vias = list(self._kipy_board.get_vias())
                    unlocked_vias = [v for v in vias if not getattr(v, "locked", False)]
                    items_to_remove.extend(unlocked_vias)
                    vias_removed = len(unlocked_vias)
                except Exception as e:
                    logger.debug("Could not get vias: %s", e)

            if not items_to_remove:
                logger.info("No tracks/vias to clear")
                return 0

            # Begin commit for undo support
            commit = self._kipy_board.begin_commit()

            try:
                # Delete the unlocked items
                self._kipy_board.remove_items(items_to_remove)
                self._kipy_board.push_commit(
                    commit, f"Clear {len(items_to_remove)} unlocked tracks/vias"
                )

                logger.info(
                    "Cleared %d tracks and %d vias",
                    len(unlocked_tracks),
                    vias_removed,
                )
                return len(items_to_remove)

            except Exception as e:
                # Rollback on error
                try:
                    self._kipy_board.drop_commit(commit)
                except Exception:
                    pass
                raise e

        except AttributeError as e:
            # kipy might not have get_tracks - try alternative
            logger.warning("get_tracks not available: %s", e)
            return self._clear_tracks_pcbnew_style()
        except Exception as e:
            logger.error("Failed to clear tracks: %s", e)
            raise

    def _clear_tracks_pcbnew_style(self) -> int:
        """
        Fallback track clearing using pcbnew-style iteration.

        Used when kipy's get_tracks method is not available.
        """
        try:
            # Try to access tracks through iteration
            deleted = 0
            items_to_remove = []

            # Some kipy versions may expose tracks differently
            if hasattr(self._kipy_board, "items"):
                for item in self._kipy_board.items():
                    item_type = type(item).__name__
                    if "Track" in item_type or "Via" in item_type:
                        if not getattr(item, "locked", False):
                            items_to_remove.append(item)

            if items_to_remove:
                commit = self._kipy_board.begin_commit()
                try:
                    self._kipy_board.remove_items(items_to_remove)
                    self._kipy_board.push_commit(
                        commit, f"Clear {len(items_to_remove)} tracks"
                    )
                    deleted = len(items_to_remove)
                except Exception:
                    self._kipy_board.drop_commit(commit)
                    raise

            return deleted
        except Exception as e:
            logger.warning("pcbnew-style track clearing failed: %s", e)

    def add_tracks(self, segments: list, vias: list, net_names: dict) -> int:
        """
        Add routed tracks and vias to the live KiCad board.

        Args:
            segments: List of RouteSegment objects with start, end, layer, width, net_id
            vias: List of Via objects with x, y, drill_diameter, pad_diameter, net_id
            net_names: Dict mapping net_id to net name string

        Returns:
            Number of items added
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        try:
            from kipy.board import Track as KiTrack
            from kipy.board import Via as KiVia
            from kipy.board import Vector2, BoardLayer
            from kipy.util.units import from_mm
        except ImportError:
            from kipy.geometry import Vector2

            from .kipy_adapter import mm_to_nm as from_mm
            KiTrack = None
            KiVia = None
            BoardLayer = None

        # If kipy track/via creation not available, log and return
        if KiTrack is None:
            logger.warning(
                "KiPy track creation not available - tracks not synced to KiCad. "
                "Routing results are in memory only."
            )
            return 0

        # Map routing layer indices (0=top, 1=bottom, 2+=inner) to kipy BoardLayer values
        # BoardLayer.BL_F_Cu = 3, BL_B_Cu = 34, inner layers are 4-33
        def map_layer_to_kicad(layer_idx: int) -> int:
            if layer_idx == 0:
                return BoardLayer.BL_F_Cu  # 3
            elif layer_idx == 1:
                return BoardLayer.BL_B_Cu  # 34
            else:
                # Inner layers: In1_Cu=4, In2_Cu=5, etc.
                return BoardLayer.BL_In1_Cu + (layer_idx - 2)  # 4 + offset

        def simplify_segments(segs):
            """Merge consecutive collinear segments to reduce zig-zag artifacts."""
            if len(segs) < 2:
                return segs

            # Group by layer and net_id first
            from collections import defaultdict
            grouped = defaultdict(list)
            for seg in segs:
                key = (seg.layer, getattr(seg, 'net_id', None))
                grouped[key].append(seg)

            result = []
            for key, group in grouped.items():
                # Sort by position to ensure proper ordering
                # Simple merge: combine consecutive segments with same direction
                simplified = []
                for seg in group:
                    if not simplified:
                        simplified.append(seg)
                        continue

                    last = simplified[-1]
                    # Check if this segment continues from the last one
                    if (abs(last.end[0] - seg.start[0]) < 0.01 and
                        abs(last.end[1] - seg.start[1]) < 0.01):
                        # Check if same direction (collinear)
                        dx1 = last.end[0] - last.start[0]
                        dy1 = last.end[1] - last.start[1]
                        dx2 = seg.end[0] - seg.start[0]
                        dy2 = seg.end[1] - seg.start[1]

                        # Cross product for collinearity check
                        cross = dx1 * dy2 - dy1 * dx2
                        if abs(cross) < 0.001:  # Collinear
                            # Extend the last segment
                            last.end = seg.end
                            continue

                    simplified.append(seg)
                result.extend(simplified)

            return result

        # Simplify segments before creating tracks
        segments = simplify_segments(segments)

        items_to_add = []

        # Build net name to net object lookup
        net_lookup = {}
        try:
            for net in self._kipy_board.get_nets():
                net_lookup[net.name] = net
        except Exception as e:
            logger.warning("Could not get nets from board: %s", e)

        # Add track segments (skip zero-length segments)
        skipped_segments = 0
        for seg in segments:
            try:
                # Skip zero-length segments
                dx = seg.end[0] - seg.start[0]
                dy = seg.end[1] - seg.start[1]
                if abs(dx) < 0.001 and abs(dy) < 0.001:
                    skipped_segments += 1
                    continue

                track = KiTrack()
                track.start = Vector2.from_xy(from_mm(seg.start[0]), from_mm(seg.start[1]))
                track.end = Vector2.from_xy(from_mm(seg.end[0]), from_mm(seg.end[1]))
                track.width = from_mm(seg.width)
                track.layer = map_layer_to_kicad(seg.layer)
                # Set net if available
                if seg.net_id is not None and seg.net_id in net_names:
                    net_name = net_names[seg.net_id]
                    if net_name in net_lookup:
                        track.net = net_lookup[net_name]
                items_to_add.append(track)
            except Exception as e:
                logger.debug("Failed to create track: %s", e)

        if skipped_segments > 0:
            logger.debug("Skipped %d zero-length segments", skipped_segments)

        # Add vias
        for via in vias:
            try:
                kicad_via = KiVia()
                kicad_via.position = Vector2.from_xy(from_mm(via.x), from_mm(via.y))
                kicad_via.drill_diameter = from_mm(via.drill_diameter)
                kicad_via.diameter = from_mm(via.pad_diameter)
                if via.net_id is not None and via.net_id in net_names:
                    net_name = net_names[via.net_id]
                    if net_name in net_lookup:
                        kicad_via.net = net_lookup[net_name]
                items_to_add.append(kicad_via)
            except Exception as e:
                logger.debug("Failed to create via: %s", e)

        if not items_to_add:
            return 0

        # Add items to board using create_items (not add_items which doesn't exist)
        commit = self._kipy_board.begin_commit()
        try:
            self._kipy_board.create_items(items_to_add)
            self._kipy_board.push_commit(
                commit, f"Add {len(items_to_add)} tracks/vias"
            )
            logger.info("Added %d tracks/vias to KiCad", len(items_to_add))
            return len(items_to_add)
        except Exception as e:
            try:
                self._kipy_board.drop_commit(commit)
            except Exception:
                pass
            logger.error("Failed to add tracks to KiCad: %s", e)
            raise

    def add_board_outline(self, outline) -> bool:
        """
        Add or update the board outline (Edge.Cuts) in KiCad.

        Args:
            outline: BoardOutline object with polygon or dimensions

        Returns:
            True if outline was added successfully
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        try:
            from kipy.board import BoardShape, BoardLayer
            from kipy.util.units import from_mm
        except ImportError as e:
            logger.warning("Could not import kipy BoardShape: %s", e)
            return False

        # Get polygon points
        if hasattr(outline, "polygon") and outline.polygon and len(outline.polygon) >= 3:
            polygon = outline.polygon
        else:
            # Create rectangle from dimensions
            polygon = [
                (outline.origin_x, outline.origin_y),
                (outline.origin_x + outline.width, outline.origin_y),
                (outline.origin_x + outline.width, outline.origin_y + outline.height),
                (outline.origin_x, outline.origin_y + outline.height),
            ]

        # First, remove existing Edge.Cuts shapes
        try:
            shapes = list(self._kipy_board.get_shapes())
            edge_cuts_items = [
                s for s in shapes
                if hasattr(s, "layer") and s.layer == BoardLayer.BL_Edge_Cuts
            ]
            if edge_cuts_items:
                commit = self._kipy_board.begin_commit()
                try:
                    self._kipy_board.remove_items(edge_cuts_items)
                    self._kipy_board.push_commit(
                        commit, f"Remove {len(edge_cuts_items)} old Edge.Cuts"
                    )
                except Exception:
                    self._kipy_board.drop_commit(commit)
        except Exception as e:
            logger.debug("Could not remove old Edge.Cuts: %s", e)

        # Create new Edge.Cuts line segments using BoardShape proto manipulation
        items_to_add = []
        for i in range(len(polygon)):
            start = polygon[i]
            end = polygon[(i + 1) % len(polygon)]
            try:
                shape = BoardShape()
                shape.layer = BoardLayer.BL_Edge_Cuts
                # Set segment geometry via proto
                shape.proto.shape.segment.start.x_nm = int(from_mm(start[0]))
                shape.proto.shape.segment.start.y_nm = int(from_mm(start[1]))
                shape.proto.shape.segment.end.x_nm = int(from_mm(end[0]))
                shape.proto.shape.segment.end.y_nm = int(from_mm(end[1]))
                items_to_add.append(shape)
            except Exception as e:
                logger.debug("Failed to create Edge.Cuts segment: %s", e)

        if not items_to_add:
            logger.warning("No Edge.Cuts segments created")
            return False

        commit = self._kipy_board.begin_commit()
        try:
            self._kipy_board.create_items(items_to_add)
            self._kipy_board.push_commit(
                commit, f"Add board outline ({len(items_to_add)} segments)"
            )
            logger.info("Added board outline to KiCad (%d segments)", len(items_to_add))
            return True
        except Exception as e:
            try:
                self._kipy_board.drop_commit(commit)
            except Exception:
                pass
            logger.error("Failed to add board outline to KiCad: %s", e)
            return False

    # =========================================================================
    # Courtyard Sync
    # =========================================================================

    def sync_courtyards_to_kicad(self, refs: List[str] = None) -> dict:
        """
        Sync auto-generated courtyards to KiCad for components without them.

        Creates courtyard rectangles on F.CrtYd (top) or B.CrtYd (bottom) layers
        for components that don't have existing courtyard graphics.

        Args:
            refs: Optional list of component refs to sync. If None, syncs all.

        Returns:
            Dict with counts of synced/skipped/failed components
        """
        if not self._connected or not self._kipy_board:
            raise KiCadNotRunningError("Not connected to KiCad")

        if not self.board:
            raise ValueError("No board loaded")

        try:
            from kipy.board_types import BoardSegment, BoardLayer, Vector2
        except ImportError as e:
            logger.error("Could not import kipy BoardSegment: %s", e)
            return {"error": str(e)}

        # Determine which layer enums to use
        try:
            front_crtyd = BoardLayer.BL_F_CrtYd
            back_crtyd = BoardLayer.BL_B_CrtYd
        except AttributeError:
            logger.error("BoardLayer courtyard enums not available in this kipy version")
            return {"error": "Courtyard layer enums not available"}

        from .kipy_adapter import find_kipy_footprints, _get_footprint_reference
        from ..board.abstraction import Layer

        # Get components to process
        if refs:
            components = {r: self.board.components[r] for r in refs if r in self.board.components}
        else:
            components = self.board.components

        # Find corresponding kipy footprints
        kipy_fps = find_kipy_footprints(self._kipy_board, list(components.keys()))

        synced = 0
        skipped = 0
        failed = 0
        footprints_to_update = []

        for ref, comp in components.items():
            # Check if component has courtyard data
            if not comp.courtyards:
                logger.debug("Component %s has no courtyard data, skipping", ref)
                skipped += 1
                continue

            # Get the kipy footprint
            fp = kipy_fps.get(ref)
            if not fp:
                logger.debug("Footprint %s not found in KiCad, skipping", ref)
                skipped += 1
                continue

            # Check if footprint already has courtyard graphics
            if self._footprint_has_courtyard(fp):
                logger.debug("Footprint %s already has courtyard, skipping", ref)
                skipped += 1
                continue

            # Get the footprint definition to add graphics to
            fp_def = fp.definition if hasattr(fp, 'definition') else fp
            if not fp_def:
                logger.debug("Footprint %s has no definition, skipping", ref)
                skipped += 1
                continue

            component_synced = False
            for courtyard_key, courtyard in comp.courtyards.items():
                # Map courtyard layer to kipy BoardLayer
                if courtyard_key == Layer.BOTTOM_COURTYARD.value:
                    crtyd_layer = back_crtyd
                else:
                    crtyd_layer = front_crtyd

                # Get local courtyard bbox (relative to component center, unrotated)
                # These LOCAL coordinates get transformed by footprint position/rotation
                rel_min_x, rel_min_y, rel_max_x, rel_max_y = courtyard.bbox

                # Get local corners (no rotation - footprint handles that)
                local_corners = [
                    (rel_min_x, rel_min_y),
                    (rel_max_x, rel_min_y),
                    (rel_max_x, rel_max_y),
                    (rel_min_x, rel_max_y),
                ]

                try:
                    # Create 4 line segments for the courtyard rectangle
                    for i in range(4):
                        start = local_corners[i]
                        end = local_corners[(i + 1) % 4]

                        seg = BoardSegment()
                        seg.layer = crtyd_layer
                        seg.start = Vector2.from_xy_mm(start[0], start[1])
                        seg.end = Vector2.from_xy_mm(end[0], end[1])

                        # Add segment to footprint definition
                        fp_def.add_item(seg)

                    component_synced = True
                    logger.debug("Created courtyard for %s on %s", ref, courtyard_key)

                except Exception as e:
                    logger.warning("Failed to create courtyard for %s on %s: %s", ref, courtyard_key, e)

            if component_synced:
                synced += 1
                footprints_to_update.append(fp)
            else:
                failed += 1

        # Update all modified footprints in KiCad
        if footprints_to_update:
            commit = self._kipy_board.begin_commit()
            try:
                self._kipy_board.update_items(footprints_to_update)
                self._kipy_board.push_commit(
                    commit, f"Add courtyards for {synced} components"
                )
                logger.info("Added courtyard graphics to %d footprints", synced)
            except Exception as e:
                try:
                    self._kipy_board.drop_commit(commit)
                except Exception:
                    pass
                logger.error("Failed to update footprints in KiCad: %s", e)
                return {"synced": 0, "skipped": skipped, "failed": synced + failed, "error": str(e)}

        return {"synced": synced, "skipped": skipped, "failed": failed}

    def _footprint_has_courtyard(self, fp) -> bool:
        """
        Check if a kipy footprint already has courtyard graphics.

        Args:
            fp: kipy footprint object

        Returns:
            True if footprint has courtyard graphics
        """
        try:
            # Check footprint's graphical items for courtyard layer
            if hasattr(fp, "definition") and fp.definition:
                definition = fp.definition
                if hasattr(definition, "graphical_items"):
                    for item in definition.graphical_items:
                        layer = getattr(item, "layer", None)
                        if layer:
                            layer_str = str(layer)
                            if "CrtYd" in layer_str or "Courtyard" in layer_str:
                                return True

            # Also check direct graphical items on footprint
            if hasattr(fp, "graphical_items"):
                for item in fp.graphical_items:
                    layer = getattr(item, "layer", None)
                    if layer:
                        layer_str = str(layer)
                        if "CrtYd" in layer_str or "Courtyard" in layer_str:
                            return True

        except Exception as e:
            logger.debug("Error checking courtyard for footprint: %s", e)

        return False

    # =========================================================================
    # Internal Sync Methods
    # =========================================================================

    def _sync_to_kicad(self):
        """Sync local board changes to KiCad.

        Uses batch update to apply all changes in a single commit for
        better undo behavior and atomicity.
        """
        if not self._dirty_refs or not self._connected or not self.board:
            return

        # Build batch updates from dirty components
        updates = []
        for ref in list(self._dirty_refs):
            comp = self.board.components.get(ref)
            if comp:
                updates.append(
                    {
                        "ref": ref,
                        "x": comp.x,
                        "y": comp.y,
                        "rotation": comp.rotation,
                        "locked": comp.locked,
                    }
                )

        if updates:
            try:
                # Use batch update for single commit
                self.update_components(updates)
            except Exception as e:
                logger.error("Failed to sync batch: %s", e)

        self._dirty_refs.clear()

    def _refresh_from_kicad(self):
        """Refresh local model from KiCad's current state."""
        if not self._connected or not self._kicad:
            return

        try:
            self._kipy_board = self._kicad.get_board()
            from .kipy_adapter import kipy_board_to_atoplace

            self.board = kipy_board_to_atoplace(self._kipy_board)
            logger.debug("Refreshed board from KiCad")
        except Exception as e:
            logger.error("Failed to refresh from KiCad: %s", e)

    # =========================================================================
    # Session Info
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "loaded": self.is_loaded,
            "source": str(self.source_path) if self.source_path else None,
            "dirty": self.is_dirty,
            "dirty_count": len(self._dirty_refs),
            "undo_available": True,  # KiCad manages undo
            "redo_available": True,
            "components": len(self.board.components) if self.board else 0,
            "mode": "kipy",
            "connected": self._connected,
        }

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        board_info = (
            f"{len(self.board.components)} components" if self.board else "no board"
        )
        return f"KiPySession({status}, {board_info})"
