"""
KiCad RPC Worker

This script runs inside the KiCad Python environment (v3.9).
It listens on stdin for JSON-RPC requests, executes board operations,
and writes responses to stdout.
"""

import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure we can import atoplace
sys.path.insert(0, str(Path(__file__).parents[2]))

from atoplace.board.abstraction import Board
from atoplace.api.actions import LayoutActions
from atoplace.api.inspection import BoardInspector
from atoplace.mcp.context.micro import Microscope
from atoplace.mcp.context.macro import MacroContext
from atoplace.mcp.context.vision import VisionContext
from atoplace.rpc.protocol import RpcRequest, RpcResponse

# Configure logging to file since stdout is used for RPC
logging.basicConfig(
    filename='atoplace_worker.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class Worker:
    def __init__(self):
        self.board: Optional[Board] = None
        self.actions: Optional[LayoutActions] = None
        self.inspector: Optional[BoardInspector] = None
        self.source_path: Optional[Path] = None  # Track the loaded board path

    def handle_request(self, req: RpcRequest) -> Any:
        method = req.method
        params = req.params

        logger.info(f"Handling method: {method}")

        # Board management
        if method == "load_board":
            return self.load_board(params["path"])
        elif method == "save_board":
            return self.save_board(params.get("path"))

        # Placement actions
        elif method == "move_absolute":
            return self.move_absolute(**params)
        elif method == "move_relative":
            return self.move_relative(**params)
        elif method == "rotate":
            return self.rotate(**params)
        elif method == "place_next_to":
            return self.place_next_to(**params)
        elif method == "align_components":
            return self.align_components(**params)
        elif method == "distribute_evenly":
            return self.distribute_evenly(**params)
        elif method == "stack_components":
            return self.stack_components(**params)
        elif method == "lock_components":
            return self.lock_components(**params)
        elif method == "arrange_pattern":
            return self.arrange_pattern(**params)
        elif method == "cluster_around":
            return self.cluster_around(**params)

        # Context/inspection
        elif method == "inspect_region":
            return self.inspect_region(**params)
        elif method == "get_board_summary":
            return self.get_board_summary()
        elif method == "check_overlaps":
            return self.check_overlaps(**params)
        elif method == "get_unplaced_components":
            return self.get_unplaced_components()
        elif method == "find_components":
            return self.find_components(**params)

        # Validation
        elif method == "run_drc":
            return self.run_drc(**params)
        elif method == "validate_placement":
            return self.validate_placement()

        else:
            raise ValueError(f"Unknown method: {method}")

    def load_board(self, path: str) -> Dict:
        self.source_path = Path(path)  # Store the source path
        self.board = Board.from_kicad(self.source_path)
        self.actions = LayoutActions(self.board)
        self.inspector = BoardInspector(self.board)
        return {
            "component_count": len(self.board.components),
            "net_count": len(self.board.nets)
        }

    def save_board(self, path: Optional[str]) -> str:
        if not self.board: raise ValueError("No board loaded")

        # Use provided path, or fall back to source path
        if path:
            out_path = Path(path)
        elif self.source_path:
            out_path = self.source_path
        else:
            raise ValueError("No output path specified and no source path available")

        self.board.to_kicad(out_path)
        return str(out_path)

    def place_next_to(self, ref, target, side, clearance, align):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.place_next_to(ref, target, side, clearance, align)
        return {"success": res.success, "message": res.message}

    def inspect_region(self, refs, padding, include_image=False):
        if not self.board: raise ValueError("No board loaded")
        micro = Microscope(self.board)
        data = micro.inspect_region(refs, padding)
        result = json.loads(data.to_json())
        
        if include_image:
            vision = VisionContext(self.board)
            image = vision.render_region(refs, show_dimensions=True)
            result["image_svg"] = image.svg_content
            
        return result

    def get_board_summary(self):
        if not self.board: raise ValueError("No board loaded")
        macro = MacroContext(self.board)
        return json.loads(macro.get_summary().to_json())

    def check_overlaps(self, refs=None):
        if not self.inspector: raise ValueError("No board loaded")
        overlaps = self.inspector.check_overlaps(refs)
        return {"overlap_count": len(overlaps), "overlaps": overlaps[:20]}

    # Additional placement actions
    def move_absolute(self, ref, x, y, rotation=None):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.move_absolute(ref, x, y, rotation)
        return {"success": res.success, "message": res.message}

    def move_relative(self, ref, dx, dy):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.move_relative(ref, dx, dy)
        return {"success": res.success, "message": res.message}

    def rotate(self, ref, angle):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.rotate(ref, angle)
        return {"success": res.success, "message": res.message}

    def align_components(self, refs, axis="x", anchor="first"):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.align_components(refs, axis, anchor)
        return {"success": res.success, "message": res.message}

    def distribute_evenly(self, refs, start_ref=None, end_ref=None, axis="auto"):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.distribute_evenly(refs, start_ref, end_ref, axis)
        return {"success": res.success, "message": res.message}

    def stack_components(self, refs, direction="down", spacing=0.5, alignment="center"):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.stack_components(refs, direction, spacing, alignment)
        return {"success": res.success, "message": res.message}

    def lock_components(self, refs, locked=True):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.lock_components(refs, locked)
        return {"success": res.success, "message": res.message}

    def arrange_pattern(self, refs, pattern="grid", spacing=2, cols=None, radius=None, center_x=None, center_y=None):
        if not self.actions: raise ValueError("No board loaded")
        # Convert center_x, center_y to tuple if provided
        center = (center_x, center_y) if center_x is not None and center_y is not None else None
        res = self.actions.arrange_pattern(refs, pattern, spacing, cols, radius, center)
        return {"success": res.success, "message": res.message}

    def cluster_around(self, anchor_ref, target_refs, side="nearest", clearance=0.5):
        if not self.actions: raise ValueError("No board loaded")
        res = self.actions.cluster_around(anchor_ref, target_refs, side, clearance)
        return {"success": res.success, "message": res.message}

    # Discovery/inspection methods
    def get_unplaced_components(self):
        if not self.board: raise ValueError("No board loaded")
        from ..mcp.context.macro import MacroContext
        macro = MacroContext(self.board)
        unplaced = macro.get_unplaced_components()
        return {"count": len(unplaced), "refs": unplaced[:50]}

    def find_components(self, query, filter_by="ref"):
        if not self.inspector: raise ValueError("No board loaded")
        try:
            matches = self.inspector.find_components(query, filter_by)
            return {"count": len(matches), "matches": matches[:50]}
        except ValueError as e:
            raise ValueError(str(e))

    # Validation methods
    def run_drc(self, use_kicad=True, dfm_profile="jlcpcb_standard", severity_filter="all"):
        if not self.board: raise ValueError("No board loaded")
        from ..validation.drc import DRCChecker
        from ..dfm.profiles import get_profile

        dfm = get_profile(dfm_profile)
        checker = DRCChecker(self.board, dfm)
        passed, violations = checker.run_checks()

        # Filter by severity
        if severity_filter != "all":
            violations = [v for v in violations if v.severity == severity_filter]

        return {
            "passed": passed,
            "violation_count": len(violations),
            "violations": [{
                "rule": v.rule,
                "severity": v.severity,
                "message": v.message,
                "items": v.items,
                "location": {"x": v.location[0], "y": v.location[1]}
            } for v in violations[:50]]
        }

    def validate_placement(self):
        if not self.board: raise ValueError("No board loaded")
        from ..validation.confidence import ConfidenceScorer
        from ..dfm.profiles import get_profile

        # Create scorer with optional DFM profile
        dfm = get_profile("jlcpcb_standard")
        scorer = ConfidenceScorer(dfm_profile=dfm)
        report = scorer.assess(self.board, routing_done=False)

        return {
            "overall_score": report.overall_score,
            "placement_score": report.placement_score,
            "routing_score": report.routing_score,
            "dfm_score": report.dfm_score,
            "electrical_score": report.electrical_score,
            "flags": [{
                "severity": f.severity,
                "category": f.category,
                "message": f.message,
                "location": {"x": f.location[0], "y": f.location[1]} if f.location else None,
                "affected_items": f.affected_items
            } for f in report.flags[:50]],
            "component_count": report.component_count,
            "net_count": report.net_count
        }

    def run(self):
        logger.info("Worker started")
        while True:
            try:
                line = sys.stdin.readline()
                if not line: break
                
                req = RpcRequest.from_json(line)
                try:
                    result = self.handle_request(req)
                    resp = RpcResponse(id=req.id, result=result)
                except Exception as e:
                    logger.error(f"Error handling {req.method}: {e}")
                    logger.error(traceback.format_exc())
                    resp = RpcResponse(id=req.id, error=str(e))
                
                sys.stdout.write(resp.to_json() + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.critical(f"Fatal worker error: {e}")
                break

if __name__ == "__main__":
    Worker().run()
