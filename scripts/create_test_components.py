#!/usr/bin/env python3
"""Add test components to the tutorial board"""
from pathlib import Path
from atoplace.board.kicad_adapter import load_kicad_board, save_kicad_board
from atoplace.board.abstraction import Component, Pad, Net, Layer

# Load the empty tutorial PCB
pcb_path = Path("examples/tutorial/elec/layout/default/tutorial.kicad_pcb")
print(f"Loading empty PCB: {pcb_path}")
board = load_kicad_board(pcb_path)
print(f"✓ Board loaded - Components: {len(board.components)}, Nets: {len(board.nets)}")

# Create 8 simple SMD components matching the tutorial netlist
# We'll use realistic SMD footprint dimensions

print("\nCreating components...")
components = []

# 0805 capacitors (2.0mm x 1.25mm body)
for ref in ["C1", "C2"]:
    comp = Component(
        reference=ref,
        footprint="C0805",
        value="10uF",
        x=140.0 + len(components) * 5,  # Space them out horizontally
        y=90.0,
        rotation=0.0,
        layer=Layer.TOP_COPPER,
        width=2.0,
        height=1.25,
        pads=[
            Pad(number="1", x=-0.95, y=0, width=1.0, height=1.3),
            Pad(number="2", x=0.95, y=0, width=1.0, height=1.3),
        ]
    )
    components.append(comp)
    print(f"  Created {ref}: {comp.footprint}")

# 0402 resistors (1.0mm x 0.5mm body)
for ref in ["R1", "R2", "R3"]:
    comp = Component(
        reference=ref,
        footprint="R0402",
        value="10k",
        x=140.0 + len(components) * 5,
        y=95.0,
        rotation=0.0,
        layer=Layer.TOP_COPPER,
        width=1.0,
        height=0.5,
        pads=[
            Pad(number="1", x=-0.5, y=0, width=0.6, height=0.6),
            Pad(number="2", x=0.5, y=0, width=0.6, height=0.6),
        ]
    )
    components.append(comp)
    print(f"  Created {ref}: {comp.footprint}")

# 0402 capacitors (1.0mm x 0.5mm body)
for ref in ["C3", "C4"]:
    comp = Component(
        reference=ref,
        footprint="C0402",
        value="100nF",
        x=140.0 + len(components) * 5,
        y=100.0,
        rotation=0.0,
        layer=Layer.TOP_COPPER,
        width=1.0,
        height=0.5,
        pads=[
            Pad(number="1", x=-0.5, y=0, width=0.6, height=0.6),
            Pad(number="2", x=0.5, y=0, width=0.6, height=0.6),
        ]
    )
    components.append(comp)
    print(f"  Created {ref}: {comp.footprint}")

# 1206 capacitor (3.2mm x 1.6mm body)
comp = Component(
    reference="C5",
    footprint="C1206",
    value="47uF",
    x=140.0 + len(components) * 5,
    y=105.0,
    rotation=0.0,
    layer=Layer.TOP_COPPER,
    width=3.2,
    height=1.6,
    pads=[
        Pad(number="1", x=-1.4, y=0, width=1.6, height=1.8),
        Pad(number="2", x=1.4, y=0, width=1.6, height=1.8),
    ]
)
components.append(comp)
print(f"  Created {comp.reference}: {comp.footprint}")

# Add components to board
board.components = components

# Create simple nets for connectivity
# Following the power filter circuit pattern
nets = [
    Net(name="VCC", connections=[(comp, "1") for comp in [components[0], components[2]]]),
    Net(name="GND", connections=[(comp, "2") for comp in components]),
    Net(name="VCC_FILTERED", connections=[(comp, "1") for comp in components[3:]]),
]
board.nets = nets

print(f"\n✓ Added {len(components)} components to board")
print(f"✓ Created {len(nets)} nets")

# Save the modified board
output_path = Path("examples/tutorial/tutorial_with_components.kicad_pcb")
save_kicad_board(board, output_path)
print(f"\n✓ Saved board with components to: {output_path}")

# Print summary
print("\nBoard summary:")
print(f"  - Components: {len(board.components)}")
print(f"  - Nets: {len(board.nets)}")
print(f"  - Board size: {board.outline.width:.1f}mm x {board.outline.height:.1f}mm")
print("\nThis board is now ready for atoplace testing!")
