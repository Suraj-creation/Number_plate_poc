from __future__ import annotations


def plate_association(plate_bbox: list, vehicles: list[dict], minimum=0.60, margin=0.20) -> dict:
    """Choose a vehicle based on visible geometry, or retain the ambiguity."""
    x1, y1, x2, y2 = plate_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    area = (x2 - x1) * (y2 - y1)
    candidates = []
    if area > 0:
        for index, vehicle in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                overlap = max(0, min(x2, vx2) - max(x1, vx1)) * max(0, min(y2, vy2) - max(y1, vy1))
                candidates.append((overlap / area, index))
    candidates.sort(reverse=True)
    top, index = candidates[0] if candidates else (0.0, None)
    next_best = candidates[1][0] if len(candidates) > 1 else 0.0
    if top < minimum:
        state = "INSUFFICIENT_CONTAINMENT" if candidates else "NO_CONTAINING_VEHICLE"
        index = None
    elif top - next_best < margin and len(candidates) > 1:
        state = "AMBIGUOUS"
        index = None
    else:
        state = "ASSOCIATED"
    return {
        "state": state, "vehicle_index": index, "containment": round(top, 4),
        "runner_up_containment": round(next_best, 4), "candidate_count": len(candidates),
    }


def associate_plate(plate_bbox: list[float], vehicles: list[dict]) -> dict | None:
    index = plate_association(plate_bbox, vehicles)["vehicle_index"]
    return vehicles[index] if index is not None else None
