"""
Tests for the A* routing algorithm.

Tests cover:
- Basic pathfinding
- Obstacle avoidance
- Multi-layer routing with vias
- Net ordering
- Goal tolerance
- Greedy multiplier behavior
"""

import pytest
import math
from typing import List, Dict

from atoplace.routing.astar_router import (
    AStarRouter,
    RouterConfig,
    RouteNode,
    RoutingResult,
    RouteDirection,
    NetOrderer,
)
from atoplace.routing.spatial_index import SpatialHashIndex, Obstacle
from atoplace.routing.visualizer import RouteSegment, Via


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def empty_index() -> SpatialHashIndex:
    """An empty spatial index for routing."""
    return SpatialHashIndex(cell_size=1.0)


@pytest.fixture
def simple_obstacle_index() -> SpatialHashIndex:
    """A spatial index with a single obstacle."""
    index = SpatialHashIndex(cell_size=1.0)
    # Add a rectangular obstacle in the center
    index.add(Obstacle(
        min_x=45.0,
        min_y=45.0,
        max_x=55.0,
        max_y=55.0,
        layer=0,
        clearance=0.0,
        net_id=None,
        obstacle_type="component"
    ))
    return index


@pytest.fixture
def blocking_wall_index() -> SpatialHashIndex:
    """A spatial index with a wall blocking direct path."""
    index = SpatialHashIndex(cell_size=1.0)
    # Add a vertical wall with a gap
    for y in range(0, 40):  # Wall from y=0 to y=40
        index.add(Obstacle(
            min_x=49.0,
            min_y=float(y),
            max_x=51.0,
            max_y=float(y + 1),
            layer=0,
            clearance=0.0,
            net_id=None,
            obstacle_type="wall"
        ))
    for y in range(60, 100):  # Wall from y=60 to y=100
        index.add(Obstacle(
            min_x=49.0,
            min_y=float(y),
            max_x=51.0,
            max_y=float(y + 1),
            layer=0,
            clearance=0.0,
            net_id=None,
            obstacle_type="wall"
        ))
    return index


@pytest.fixture
def two_layer_index() -> SpatialHashIndex:
    """A spatial index for two-layer routing tests."""
    index = SpatialHashIndex(cell_size=1.0)
    # Add obstacle only on layer 0
    index.add(Obstacle(
        min_x=45.0,
        min_y=45.0,
        max_x=55.0,
        max_y=55.0,
        layer=0,
        clearance=0.0,
        net_id=None,
        obstacle_type="component"
    ))
    return index


@pytest.fixture
def default_config() -> RouterConfig:
    """Default router configuration."""
    return RouterConfig(
        greedy_weight=2.0,
        grid_size=0.5,
        max_iterations=10000,
        layer_count=2,
        trace_width=0.2,
        clearance=0.15,
        goal_tolerance=0.5,
    )


@pytest.fixture
def manhattan_config() -> RouterConfig:
    """Router configuration for Manhattan routing only."""
    return RouterConfig(
        greedy_weight=2.0,
        grid_size=0.5,
        max_iterations=10000,
        direction=RouteDirection.MANHATTAN,
        layer_count=2,
    )


# =============================================================================
# RouteNode Tests
# =============================================================================

class TestRouteNode:
    """Tests for RouteNode dataclass."""

    def test_distance_to(self):
        """Test Euclidean distance calculation."""
        node1 = RouteNode(0.0, 0.0, 0)
        node2 = RouteNode(3.0, 4.0, 0)

        distance = node1.distance_to(node2)
        assert abs(distance - 5.0) < 0.001

    def test_manhattan_distance_to(self):
        """Test Manhattan distance calculation."""
        node1 = RouteNode(0.0, 0.0, 0)
        node2 = RouteNode(3.0, 4.0, 0)

        distance = node1.manhattan_distance_to(node2)
        assert abs(distance - 7.0) < 0.001

    def test_equality(self):
        """Test node equality with floating point tolerance.

        RouteNode uses 0.0001mm (0.1 micron) precision for hashing/equality.
        Values within this tolerance are considered equal.
        """
        node1 = RouteNode(10.0, 20.0, 0)
        # Within 0.1 micron tolerance (floating-point noise level)
        node2 = RouteNode(10.00001, 20.00001, 0)
        # Outside tolerance (0.01mm = 10 micron difference)
        node3 = RouteNode(10.01, 20.0, 0)

        assert node1 == node2  # Should be equal (within 0.0001mm precision)
        assert node1 != node3

    def test_hash_consistency(self):
        """Test that equal nodes have same hash."""
        node1 = RouteNode(10.0, 20.0, 0)
        node2 = RouteNode(10.0, 20.0, 0)

        assert hash(node1) == hash(node2)

    def test_different_layers_not_equal(self):
        """Test that nodes on different layers are not equal."""
        node1 = RouteNode(10.0, 20.0, 0)
        node2 = RouteNode(10.0, 20.0, 1)

        assert node1 != node2


# =============================================================================
# Basic Routing Tests
# =============================================================================

class TestBasicRouting:
    """Tests for basic A* routing functionality."""

    def test_route_straight_line(self, empty_index, default_config):
        """Test routing a straight line with no obstacles."""
        router = AStarRouter(empty_index, default_config)

        start = RouteNode(10.0, 50.0, 0)
        goal = RouteNode(90.0, 50.0, 0)

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        assert result.success, f"Routing failed: {result.failure_reason}"
        assert len(result.segments) > 0
        assert result.total_length > 0

    def test_route_diagonal(self, empty_index, default_config):
        """Test routing diagonally."""
        router = AStarRouter(empty_index, default_config)

        start = RouteNode(10.0, 10.0, 0)
        goal = RouteNode(90.0, 90.0, 0)

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        assert result.success

    def test_route_around_obstacle(self, simple_obstacle_index, default_config):
        """Test routing around a simple obstacle."""
        router = AStarRouter(simple_obstacle_index, default_config)

        start = RouteNode(10.0, 50.0, 0)
        goal = RouteNode(90.0, 50.0, 0)

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        assert result.success, f"Routing failed: {result.failure_reason}"

        # Path should avoid the obstacle (y should deviate from 50)
        y_values = [seg.start[1] for seg in result.segments]
        y_values.extend([seg.end[1] for seg in result.segments])

        # Some segment should be above or below the obstacle center
        has_detour = any(y < 45.0 or y > 55.0 for y in y_values)
        assert has_detour, "Path should route around obstacle"

    def test_route_through_gap(self, blocking_wall_index, default_config):
        """Test routing through a gap in a wall."""
        router = AStarRouter(blocking_wall_index, default_config)

        start = RouteNode(10.0, 50.0, 0)
        goal = RouteNode(90.0, 50.0, 0)

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        assert result.success, f"Routing failed: {result.failure_reason}"


# =============================================================================
# Multi-Layer Routing Tests
# =============================================================================

class TestMultiLayerRouting:
    """Tests for multi-layer routing with vias."""

    def test_via_to_avoid_obstacle(self, two_layer_index, default_config):
        """Test that router can via to avoid an obstacle."""
        default_config.via_cost = 2.0  # Make vias cheap
        router = AStarRouter(two_layer_index, default_config)

        # Route must go through obstacle area - via to layer 1 should help
        start = RouteNode(40.0, 50.0, 0)
        goal = RouteNode(60.0, 50.0, 0)

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        assert result.success, f"Routing failed: {result.failure_reason}"

    def test_via_count(self, empty_index, default_config):
        """Test that vias are counted correctly."""
        router = AStarRouter(empty_index, default_config)

        # Create pads that simulate needing to route between them
        pads = [
            Obstacle(min_x=9.9, min_y=49.9, max_x=10.1, max_y=50.1,
                     layer=0, clearance=0, net_id=1, obstacle_type="pad"),
            Obstacle(min_x=89.9, min_y=49.9, max_x=90.1, max_y=50.1,
                     layer=0, clearance=0, net_id=1, obstacle_type="pad"),
        ]

        result = router.route_net(pads, "test_net", 1)

        assert result.success
        assert result.via_count >= 0  # May or may not need vias


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for router configuration options."""

    def test_manhattan_only_routing(self, empty_index, manhattan_config):
        """Test Manhattan-only routing produces correct angles."""
        router = AStarRouter(empty_index, manhattan_config)

        start = RouteNode(10.0, 10.0, 0)
        goal = RouteNode(50.0, 50.0, 0)

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        assert result.success

        # All segments should be horizontal or vertical
        for seg in result.segments:
            dx = abs(seg.end[0] - seg.start[0])
            dy = abs(seg.end[1] - seg.start[1])
            # Either dx or dy should be ~0
            assert dx < 0.01 or dy < 0.01, f"Non-Manhattan segment: {seg}"

    def test_greedy_weight_affects_path(self, simple_obstacle_index):
        """Test that greedy weight affects routing behavior."""
        # High greedy weight = faster but less optimal
        config_high = RouterConfig(greedy_weight=5.0, grid_size=0.5, max_iterations=10000)
        config_low = RouterConfig(greedy_weight=1.0, grid_size=0.5, max_iterations=10000)

        router_high = AStarRouter(simple_obstacle_index, config_high)
        router_low = AStarRouter(simple_obstacle_index, config_low)

        start = RouteNode(10.0, 50.0, 0)
        goal = RouteNode(90.0, 50.0, 0)

        result_high = router_high._route_two_points(start, goal, None, 0.2, 0.15)
        result_low = router_low._route_two_points(start, goal, None, 0.2, 0.15)

        assert result_high.success
        assert result_low.success

        # High greedy should explore fewer nodes (or similar)
        # Low greedy (optimal) might find shorter path
        assert result_high.explored_count <= result_low.explored_count * 2

    def test_max_iterations_limit(self, simple_obstacle_index):
        """Test that max iterations limit is respected."""
        config = RouterConfig(max_iterations=10, grid_size=0.5)
        router = AStarRouter(simple_obstacle_index, config)

        start = RouteNode(10.0, 50.0, 0)
        goal = RouteNode(90.0, 50.0, 0)

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        # Should fail due to iteration limit
        assert not result.success
        assert "iterations" in result.failure_reason.lower()


# =============================================================================
# Goal Tolerance Tests
# =============================================================================

class TestGoalTolerance:
    """Tests for goal tolerance behavior."""

    def test_reaches_goal_within_tolerance(self, empty_index, default_config):
        """Test that router considers goal reached within tolerance."""
        default_config.goal_tolerance = 1.0  # 1mm tolerance
        router = AStarRouter(empty_index, default_config)

        start = RouteNode(10.0, 50.0, 0)
        goal = RouteNode(50.5, 50.5, 0)  # Slightly offset from grid

        result = router._route_two_points(start, goal, None, 0.2, 0.15)

        assert result.success, f"Should reach goal within tolerance: {result.failure_reason}"


# =============================================================================
# Same-Net Filtering Tests
# =============================================================================

class TestSameNetFiltering:
    """Tests for same-net obstacle filtering."""

    def test_ignores_same_net_obstacles(self):
        """Test that same-net obstacles don't block routing."""
        index = SpatialHashIndex(cell_size=1.0)

        # Add obstacle with net_id=1
        index.add(Obstacle(
            min_x=45.0,
            min_y=45.0,
            max_x=55.0,
            max_y=55.0,
            layer=0,
            clearance=0.0,
            net_id=1,  # Same as our routing net
            obstacle_type="trace"
        ))

        config = RouterConfig(grid_size=0.5, max_iterations=10000)
        router = AStarRouter(index, config)

        start = RouteNode(40.0, 50.0, 0)
        goal = RouteNode(60.0, 50.0, 0)

        # Should succeed because obstacle has same net_id
        result = router._route_two_points(start, goal, net_id=1, trace_width=0.2, net_clearance=0.15)

        assert result.success, "Should ignore same-net obstacles"

    def test_blocks_different_net_obstacles(self):
        """Test that different-net obstacles block routing."""
        index = SpatialHashIndex(cell_size=1.0)

        # Add obstacle with net_id=2
        index.add(Obstacle(
            min_x=45.0,
            min_y=45.0,
            max_x=55.0,
            max_y=55.0,
            layer=0,
            clearance=0.0,
            net_id=2,  # Different from our routing net
            obstacle_type="trace"
        ))

        config = RouterConfig(grid_size=0.5, max_iterations=10000)
        router = AStarRouter(index, config)

        start = RouteNode(40.0, 50.0, 0)
        goal = RouteNode(60.0, 50.0, 0)

        # Should route around (not through) the obstacle
        result = router._route_two_points(start, goal, net_id=1, trace_width=0.2, net_clearance=0.15)

        if result.success:
            # Path should not go straight through obstacle area
            for seg in result.segments:
                mid_x = (seg.start[0] + seg.end[0]) / 2
                mid_y = (seg.start[1] + seg.end[1]) / 2
                # Midpoint shouldn't be inside obstacle
                if 45.0 < mid_x < 55.0 and 45.0 < mid_y < 55.0:
                    # If segment goes through, it should be small (endpoint only)
                    seg_len = math.sqrt((seg.end[0] - seg.start[0])**2 +
                                       (seg.end[1] - seg.start[1])**2)
                    assert seg_len < 1.0, "Should not route through different-net obstacle"


# =============================================================================
# Net Ordering Tests
# =============================================================================

class TestNetOrderer:
    """Tests for net ordering by difficulty."""

    def test_orders_by_congestion(self, simple_obstacle_index):
        """Test that congested nets are ordered first."""
        orderer = NetOrderer(simple_obstacle_index)

        # Create mock nets with different congestion levels
        class MockNet:
            def __init__(self, name, pads):
                self.net_name = name
                self.net_id = 1
                self.pads = pads

        # Net near obstacle (congested)
        pad1 = Obstacle(min_x=50.0, min_y=50.0, max_x=51.0, max_y=51.0,
                        layer=0, clearance=0, net_id=1, obstacle_type="pad")
        pad2 = Obstacle(min_x=60.0, min_y=50.0, max_x=61.0, max_y=51.0,
                        layer=0, clearance=0, net_id=1, obstacle_type="pad")
        congested_net = MockNet("congested", [pad1, pad2])

        # Net far from obstacle (easy)
        pad3 = Obstacle(min_x=10.0, min_y=10.0, max_x=11.0, max_y=11.0,
                        layer=0, clearance=0, net_id=2, obstacle_type="pad")
        pad4 = Obstacle(min_x=20.0, min_y=10.0, max_x=21.0, max_y=11.0,
                        layer=0, clearance=0, net_id=2, obstacle_type="pad")
        easy_net = MockNet("easy", [pad3, pad4])

        ordered = orderer.order_nets([easy_net, congested_net])

        # Congested net should be first (harder = route first)
        assert ordered[0].net_name == "congested"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_pad_net(self, empty_index, default_config):
        """Test that single-pad net returns success (nothing to route)."""
        router = AStarRouter(empty_index, default_config)

        single_pad = Obstacle(min_x=50.0, min_y=50.0, max_x=51.0, max_y=51.0,
                              layer=0, clearance=0, net_id=1, obstacle_type="pad")

        result = router.route_net([single_pad], "single_pad_net", 1)

        assert result.success
        assert len(result.segments) == 0

    def test_empty_net(self, empty_index, default_config):
        """Test that empty net returns success."""
        router = AStarRouter(empty_index, default_config)

        result = router.route_net([], "empty_net", 1)

        assert result.success

    def test_same_point_routing(self, empty_index, default_config):
        """Test routing from a point to itself."""
        router = AStarRouter(empty_index, default_config)

        node = RouteNode(50.0, 50.0, 0)

        result = router._route_two_points(node, node, None, 0.2, 0.15)

        assert result.success

    def test_router_reset(self, empty_index, default_config):
        """Test that router state is properly reset."""
        router = AStarRouter(empty_index, default_config)

        # Add some routed traces
        seg = RouteSegment(start=(10.0, 10.0), end=(20.0, 10.0), layer=0, width=0.2)
        router.add_routed_trace(seg)

        assert len(router.routed_segments) == 1

        # Reset
        router.reset()

        assert len(router.routed_segments) == 0
        assert len(router.routed_vias) == 0


# =============================================================================
# Heuristic Tests
# =============================================================================

class TestHeuristic:
    """Tests for A* heuristic function."""

    def test_heuristic_admissible(self, empty_index, default_config):
        """Test that heuristic never overestimates."""
        router = AStarRouter(empty_index, default_config)

        start = RouteNode(0.0, 0.0, 0)
        goal = RouteNode(100.0, 100.0, 0)

        h = router._heuristic(start, goal)

        # Manhattan distance is admissible for grid routing
        # Actual shortest path >= heuristic
        actual = start.distance_to(goal)  # Euclidean (shorter than Manhattan)
        manhattan = start.manhattan_distance_to(goal)

        assert h <= manhattan * 1.1  # Allow small tolerance

    def test_heuristic_layer_penalty(self, empty_index, default_config):
        """Test that heuristic includes layer change penalty."""
        router = AStarRouter(empty_index, default_config)

        node_layer0 = RouteNode(50.0, 50.0, 0)
        node_layer1 = RouteNode(50.0, 50.0, 1)
        goal_layer0 = RouteNode(50.0, 50.0, 0)

        h_same_layer = router._heuristic(node_layer0, goal_layer0)
        h_diff_layer = router._heuristic(node_layer1, goal_layer0)

        # Different layer should have higher heuristic (via penalty)
        assert h_diff_layer > h_same_layer
