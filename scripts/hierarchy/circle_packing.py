"""Deterministic tangent-circle packing for the empirical molecule table."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import math

import sys

from pathlib import Path



import numpy as np

sys.path.insert(0, str(_paper_path(__file__).resolve().parents[2]))

WIDTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

def tangent_intersections(center_a, distance_a, center_b, distance_b):
    delta = center_b - center_a
    separation = float(np.linalg.norm(delta))
    if separation == 0 or separation > distance_a + distance_b or separation < abs(distance_a - distance_b):
        return []
    along = (distance_a ** 2 - distance_b ** 2 + separation ** 2) / (2 * separation)
    height_sq = max(distance_a ** 2 - along ** 2, 0.0)
    midpoint = center_a + along * delta / separation
    perpendicular = np.array([-delta[1], delta[0]]) / separation
    height = math.sqrt(height_sq)
    return [midpoint + height * perpendicular, midpoint - height * perpendicular]

def is_nonoverlapping(center, radius, placed, tolerance=1e-7):
    return all(np.linalg.norm(center - other_center) >= radius + other_radius - tolerance for _, _, other_radius, other_center in placed)

def packing_score(center, radius, placed):
    points = [(other_center, other_radius) for _, _, other_radius, other_center in placed] + [(center, radius)]
    extent = max(float(np.linalg.norm(point)) + item_radius for point, item_radius in points)
    centroid = np.mean([point for point, _ in points], axis=0)
    return extent + 0.05 * float(np.linalg.norm(centroid))

def pack_circles(entries, radius_power: float = 0.5):
    ordered = sorted(entries, key=lambda item: (-item[1], item[0]))
    placed = []
    for category, coefficient in ordered:
        radius = coefficient ** radius_power
        if not placed:
            center = np.zeros(2)
        else:
            candidates = []
            for first in range(len(placed)):
                _, _, first_radius, first_center = placed[first]
                for second in range(first + 1, len(placed)):
                    _, _, second_radius, second_center = placed[second]
                    for candidate in tangent_intersections(first_center, first_radius + radius, second_center, second_radius + radius):
                        if is_nonoverlapping(candidate, radius, placed):
                            candidates.append(candidate)
            if not candidates:
                for _, _, other_radius, other_center in placed:
                    for angle in np.linspace(0, 2 * math.pi, 180, endpoint=False):
                        candidate = other_center + (other_radius + radius) * np.array([math.cos(angle), math.sin(angle)])
                        if is_nonoverlapping(candidate, radius, placed):
                            candidates.append(candidate)
            if not candidates:
                raise RuntimeError(f"Could not place circle for {category}")
            center = min(candidates, key=lambda candidate: packing_score(candidate, radius, placed))
        placed.append((category, coefficient, radius, center))

    weighted_center = sum(radius ** 2 * center for _, _, radius, center in placed) / sum(radius ** 2 for _, _, radius, _ in placed)
    return [(category, coefficient, radius, center - weighted_center) for category, coefficient, radius, center in placed]

def validate_packing(packed, tolerance=1e-6):
    for index, (_, _, radius, center) in enumerate(packed):
        for _, _, other_radius, other_center in packed[index + 1:]:
            if np.linalg.norm(center - other_center) < radius + other_radius - tolerance:
                raise ValueError("Packed circles overlap")
    if len(packed) > 1:
        for index, (_, _, radius, center) in enumerate(packed):
            tangent = any(abs(np.linalg.norm(center - other_center) - radius - other_radius) < 1e-5 for other_index, (_, _, other_radius, other_center) in enumerate(packed) if other_index != index)
            if not tangent:
                raise ValueError(f"Circle {index} is not tangent to another circle")
