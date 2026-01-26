#!/usr/bin/env python3
"""
OhanaAI - Generate Interactive Fan Chart with Predictions

Creates an HTML fan chart visualization showing:
- Known ancestors (solid)
- Predicted missing ancestors (dashed/highlighted)
- Candidate suggestions on hover

Usage: python generate_fanchart.py path/to/file.ged [--person ID] [--generations N]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import html

import numpy as np

from prepare_data import parse_gedcom, prepare_training_data, Individual


@dataclass
class FanChartNode:
    id: str
    name: str
    birth_year: Optional[int]
    death_year: Optional[int]
    gender: Optional[str]
    generation: int
    position: int  # Position within generation (0 = leftmost)
    is_missing: bool = False
    is_predicted: bool = False
    missing_probability: float = 0.0
    candidates: List[Dict] = None

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


def load_model(model_path: Path) -> Dict:
    """Load model weights."""
    weights = np.load(model_path)
    return {k: weights[k] for k in weights.files}


def predict(features: np.ndarray, weights: Dict) -> np.ndarray:
    """Run inference."""
    x = features
    x = np.maximum(x @ weights['fc1_weight'].T + weights['fc1_bias'], 0)
    x = np.maximum(x @ weights['fc2_weight'].T + weights['fc2_bias'], 0)
    x = np.maximum(x @ weights['fc3_weight'].T + weights['fc3_bias'], 0)
    x = 1 / (1 + np.exp(-(x @ weights['out_weight'].T + weights['out_bias'])))
    return x


def find_candidates(person: Individual, individuals: Dict[str, Individual],
                   relation_type: str, limit: int = 3) -> List[Dict]:
    """Find candidate relatives."""
    candidates = []

    for cand_id, cand in individuals.items():
        if cand_id == person.id:
            continue
        if person.father == cand_id or person.mother == cand_id:
            continue

        score = 0.0

        if relation_type == 'father':
            if cand.gender != 'M':
                continue
            if cand.birth_year and person.birth_year:
                age_diff = person.birth_year - cand.birth_year
                if 15 <= age_diff <= 55:
                    score += 0.3
                else:
                    continue
            if person.surname and cand.surname:
                if person.surname.lower() == cand.surname.lower():
                    score += 0.5

        elif relation_type == 'mother':
            if cand.gender != 'F':
                continue
            if cand.birth_year and person.birth_year:
                age_diff = person.birth_year - cand.birth_year
                if 12 <= age_diff <= 50:
                    score += 0.3
                else:
                    continue

        if score > 0:
            name = f"{cand.given_name or ''} {cand.surname or ''}".strip() or cand_id
            candidates.append({
                'id': cand_id,
                'name': name,
                'birth_year': cand.birth_year,
                'score': score
            })

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:limit]


def build_ancestor_tree(
    root_id: str,
    individuals: Dict[str, Individual],
    predictions: Dict[str, np.ndarray],
    max_generations: int = 5
) -> List[FanChartNode]:
    """Build ancestor tree for fan chart."""
    nodes = []

    root = individuals.get(root_id)
    if not root:
        return nodes

    # BFS to build tree
    queue = [(root_id, 0, 0)]  # (id, generation, position)
    visited = set()

    while queue:
        current_id, gen, pos = queue.pop(0)

        if current_id in visited or gen > max_generations:
            continue
        visited.add(current_id)

        person = individuals.get(current_id)
        if not person:
            continue

        name = f"{person.given_name or ''} {person.surname or ''}".strip() or current_id
        pred = predictions.get(current_id, np.zeros(5))

        node = FanChartNode(
            id=current_id,
            name=name,
            birth_year=person.birth_year,
            death_year=person.death_year,
            gender=person.gender,
            generation=gen,
            position=pos,
            is_missing=False,
            is_predicted=False,
            missing_probability=0.0
        )
        nodes.append(node)

        # Add parents
        next_gen = gen + 1
        if next_gen <= max_generations:
            father_pos = pos * 2
            mother_pos = pos * 2 + 1

            if person.father:
                queue.append((person.father, next_gen, father_pos))
            else:
                # Add placeholder for missing father
                missing_node = FanChartNode(
                    id=f"missing_father_{current_id}",
                    name="Unknown Father",
                    birth_year=person.birth_year - 30 if person.birth_year else None,
                    death_year=None,
                    gender='M',
                    generation=next_gen,
                    position=father_pos,
                    is_missing=True,
                    is_predicted=True,
                    missing_probability=float(pred[0]) if len(pred) > 0 else 0.5,
                    candidates=find_candidates(person, individuals, 'father')
                )
                nodes.append(missing_node)

            if person.mother:
                queue.append((person.mother, next_gen, mother_pos))
            else:
                # Add placeholder for missing mother
                missing_node = FanChartNode(
                    id=f"missing_mother_{current_id}",
                    name="Unknown Mother",
                    birth_year=person.birth_year - 28 if person.birth_year else None,
                    death_year=None,
                    gender='F',
                    generation=next_gen,
                    position=mother_pos,
                    is_missing=True,
                    is_predicted=True,
                    missing_probability=float(pred[1]) if len(pred) > 1 else 0.5,
                    candidates=find_candidates(person, individuals, 'mother')
                )
                nodes.append(missing_node)

    return nodes


def generate_html(nodes: List[FanChartNode], root_name: str, output_path: Path):
    """Generate interactive HTML fan chart."""

    # Convert nodes to JSON
    nodes_json = json.dumps([{
        'id': n.id,
        'name': n.name,
        'birth_year': n.birth_year,
        'death_year': n.death_year,
        'gender': n.gender,
        'generation': n.generation,
        'position': n.position,
        'is_missing': n.is_missing,
        'is_predicted': n.is_predicted,
        'missing_probability': n.missing_probability,
        'candidates': n.candidates
    } for n in nodes])

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OhanaAI - Ancestor Fan Chart: {html.escape(root_name)}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}
        .header {{
            padding: 20px;
            text-align: center;
            background: rgba(0,0,0,0.3);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        .header p {{
            opacity: 0.7;
            font-size: 0.9em;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            padding: 15px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85em;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        .chart-container {{
            display: flex;
            justify-content: center;
            padding: 20px;
            overflow: auto;
        }}
        #fanchart {{
            background: rgba(255,255,255,0.02);
            border-radius: 50%;
        }}
        .arc {{
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        .arc:hover {{
            opacity: 0.8;
        }}
        .arc-label {{
            font-size: 10px;
            fill: #fff;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
        }}
        .tooltip {{
            position: fixed;
            background: rgba(20, 20, 40, 0.95);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 13px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .tooltip h3 {{
            margin-bottom: 8px;
            font-size: 14px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            padding-bottom: 6px;
        }}
        .tooltip .dates {{
            opacity: 0.7;
            margin-bottom: 8px;
        }}
        .tooltip .prediction {{
            background: rgba(255,100,100,0.2);
            border-left: 3px solid #ff6b6b;
            padding: 8px;
            margin: 8px 0;
            border-radius: 0 4px 4px 0;
        }}
        .tooltip .candidates {{
            margin-top: 10px;
        }}
        .tooltip .candidates h4 {{
            font-size: 12px;
            margin-bottom: 6px;
            color: #4ecdc4;
        }}
        .tooltip .candidate {{
            padding: 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .tooltip .candidate:last-child {{
            border-bottom: none;
        }}
        .missing-indicator {{
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌺 Ancestor Fan Chart</h1>
        <p>Root: {html.escape(root_name)} | Showing predicted missing ancestors</p>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: #4a90d9;"></div>
            <span>Male (Known)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #d94a7b;"></div>
            <span>Female (Known)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: repeating-linear-gradient(45deg, #ff6b6b, #ff6b6b 2px, transparent 2px, transparent 4px);"></div>
            <span>Missing (Predicted)</span>
        </div>
    </div>

    <div class="chart-container">
        <svg id="fanchart"></svg>
    </div>

    <div class="tooltip" style="display: none;"></div>

    <script>
        const nodes = {nodes_json};

        const width = 900;
        const height = 900;
        const centerX = width / 2;
        const centerY = height / 2;
        const innerRadius = 60;
        const radiusStep = 70;

        const svg = d3.select("#fanchart")
            .attr("width", width)
            .attr("height", height);

        const g = svg.append("g")
            .attr("transform", `translate(${{centerX}}, ${{centerY}})`);

        const tooltip = d3.select(".tooltip");

        // Group nodes by generation
        const generations = {{}};
        nodes.forEach(node => {{
            if (!generations[node.generation]) {{
                generations[node.generation] = [];
            }}
            generations[node.generation].push(node);
        }});

        // Sort each generation by position
        Object.keys(generations).forEach(gen => {{
            generations[gen].sort((a, b) => a.position - b.position);
        }});

        // Draw arcs for each generation
        Object.keys(generations).forEach(gen => {{
            const genNum = parseInt(gen);
            const genNodes = generations[gen];
            const numSlots = Math.pow(2, genNum);
            const arcAngle = Math.PI / numSlots;

            const inner = innerRadius + genNum * radiusStep;
            const outer = inner + radiusStep - 5;

            genNodes.forEach((node, i) => {{
                const startAngle = -Math.PI/2 + node.position * arcAngle * 2;
                const endAngle = startAngle + arcAngle * 2 - 0.02;

                const arc = d3.arc()
                    .innerRadius(inner)
                    .outerRadius(outer)
                    .startAngle(startAngle)
                    .endAngle(endAngle)
                    .padAngle(0.01);

                // Determine color
                let fillColor;
                if (node.is_missing) {{
                    fillColor = node.gender === 'M' ? '#8b4a4a' : '#8b4a6b';
                }} else {{
                    fillColor = node.gender === 'M' ? '#4a90d9' : '#d94a7b';
                    if (node.gender === null || node.gender === 'U') {{
                        fillColor = '#6b7b8b';
                    }}
                }}

                const arcGroup = g.append("g")
                    .attr("class", "arc" + (node.is_missing ? " missing-indicator" : ""));

                // Main arc
                arcGroup.append("path")
                    .attr("d", arc)
                    .attr("fill", fillColor)
                    .attr("stroke", node.is_missing ? "#ff6b6b" : "rgba(255,255,255,0.3)")
                    .attr("stroke-width", node.is_missing ? 2 : 1)
                    .attr("stroke-dasharray", node.is_missing ? "5,3" : "none");

                // Label
                const midAngle = (startAngle + endAngle) / 2;
                const labelRadius = (inner + outer) / 2;
                const labelX = Math.cos(midAngle) * labelRadius;
                const labelY = Math.sin(midAngle) * labelRadius;

                // Only show labels for first few generations or if there's space
                if (genNum <= 3 || genNodes.length <= 8) {{
                    const displayName = node.name.length > 15
                        ? node.name.substring(0, 12) + "..."
                        : node.name;

                    arcGroup.append("text")
                        .attr("class", "arc-label")
                        .attr("x", labelX)
                        .attr("y", labelY)
                        .attr("text-anchor", "middle")
                        .attr("dominant-baseline", "middle")
                        .attr("transform", `rotate(${{midAngle * 180 / Math.PI + 90}}, ${{labelX}}, ${{labelY}})`)
                        .text(displayName);
                }}

                // Tooltip interaction
                arcGroup.on("mouseover", (event) => {{
                    let html = `<h3>${{node.name}}</h3>`;

                    if (node.birth_year || node.death_year) {{
                        html += `<div class="dates">`;
                        if (node.birth_year) html += `b. ${{node.birth_year}}`;
                        if (node.birth_year && node.death_year) html += ` - `;
                        if (node.death_year) html += `d. ${{node.death_year}}`;
                        html += `</div>`;
                    }}

                    if (node.is_missing) {{
                        html += `<div class="prediction">`;
                        html += `<strong>⚠️ Missing Ancestor</strong><br>`;
                        html += `Confidence: ${{(node.missing_probability * 100).toFixed(1)}}%`;
                        html += `</div>`;

                        if (node.candidates && node.candidates.length > 0) {{
                            html += `<div class="candidates">`;
                            html += `<h4>Possible Matches:</h4>`;
                            node.candidates.forEach(c => {{
                                html += `<div class="candidate">`;
                                html += `${{c.name}}`;
                                if (c.birth_year) html += ` (b. ${{c.birth_year}})`;
                                html += `</div>`;
                            }});
                            html += `</div>`;
                        }}
                    }}

                    tooltip.html(html)
                        .style("display", "block")
                        .style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mousemove", (event) => {{
                    tooltip.style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", () => {{
                    tooltip.style("display", "none");
                }});
            }});
        }});

        // Center circle with root person
        const rootNode = nodes.find(n => n.generation === 0);
        if (rootNode) {{
            g.append("circle")
                .attr("r", innerRadius - 5)
                .attr("fill", rootNode.gender === 'M' ? '#4a90d9' : '#d94a7b')
                .attr("stroke", "#fff")
                .attr("stroke-width", 2);

            g.append("text")
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .attr("fill", "#fff")
                .attr("font-size", "12px")
                .attr("font-weight", "bold")
                .text(rootNode.name.length > 20 ? rootNode.name.substring(0, 17) + "..." : rootNode.name);
        }}
    </script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Fan chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate ancestor fan chart with predictions')
    parser.add_argument('gedcom_file', help='Path to GEDCOM file')
    parser.add_argument('--person', help='Person ID to use as root (default: first person)')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations to show')
    parser.add_argument('--model', default='models/family_tree_gnn/best_model.npz')
    parser.add_argument('--output', help='Output HTML file path')
    args = parser.parse_args()

    gedcom_path = Path(args.gedcom_file)
    model_path = Path(args.model)

    if not model_path.is_absolute():
        model_path = Path(__file__).parent.parent.parent / model_path

    if not gedcom_path.exists():
        print(f"Error: GEDCOM file not found: {gedcom_path}")
        sys.exit(1)

    print(f"Loading GEDCOM: {gedcom_path}")
    individuals, families = parse_gedcom(gedcom_path)
    print(f"  {len(individuals)} individuals")

    print(f"\nLoading model: {model_path}")
    weights = load_model(model_path)

    print("Preparing features and running predictions...")
    training_data = prepare_training_data(individuals, families)
    features = np.array(training_data['nodeFeatures'], dtype=np.float32)
    node_ids = training_data['nodeIds']

    preds = predict(features, weights)
    predictions = {node_ids[i]: preds[i] for i in range(len(node_ids))}

    # Find root person
    root_id = args.person
    if not root_id:
        # Default to first person with most descendants or data
        for nid, ind in individuals.items():
            if ind.given_name and ind.surname:
                root_id = nid
                break
        if not root_id:
            root_id = list(individuals.keys())[0]

    if root_id not in individuals:
        print(f"Error: Person {root_id} not found")
        print("Available IDs (first 10):", list(individuals.keys())[:10])
        sys.exit(1)

    root_person = individuals[root_id]
    root_name = f"{root_person.given_name or ''} {root_person.surname or ''}".strip()

    print(f"\nBuilding fan chart for: {root_name}")
    nodes = build_ancestor_tree(root_id, individuals, predictions, args.generations)
    print(f"  {len(nodes)} nodes in chart")

    # Count missing
    missing = sum(1 for n in nodes if n.is_missing)
    print(f"  {missing} predicted missing ancestors")

    # Output path
    output_path = Path(args.output) if args.output else gedcom_path.with_suffix('.fanchart.html')

    generate_html(nodes, root_name, output_path)
    print(f"\nOpen in browser: file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
