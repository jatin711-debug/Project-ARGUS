"""Steer SimSat to coordinates with interesting ground targets.

Uses the SimSat dashboard's programmatic control API to set start times
that put the satellite over airports, military bases, and ports.
"""

import sys
import time
import requests

DASHBOARD = "http://localhost:8000"
API = "http://localhost:9005"

# ── Known interesting start times / locations ────────────────────────
# These are approximate times that place the polar orbit over land areas
# with airports, ports, and dense infrastructure.
PRESETS = {
    "mideast": {
        "desc": "Eastern Mediterranean / Middle East (lat ~35N, lon ~36E)",
        "start_time": "2026-03-15T08:30:00Z",
    },
    "pacific": {
        "desc": "Pacific near US West Coast (lat ~37N, lon ~-149E)",
        "start_time": "2026-03-15T08:00:00Z",
    },
    "northpole": {
        "desc": "High latitude pass (polar, lat ~81N)",
        "start_time": "2026-03-15T08:15:00Z",
    },
}


def get_position():
    """Fetch current satellite position."""
    try:
        resp = requests.get(f"{API}/data/current/position", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pos = data["lon-lat-alt"]
        ts = data.get("timestamp", "?")
        return pos, ts
    except Exception as e:
        print(f"  ✗ Could not fetch position: {e}")
        return None, None


def steer_to(preset_name: str):
    """Set SimSat to a preset start time and start the simulation."""
    preset = PRESETS.get(preset_name)
    if not preset:
        print(f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"  Steering to: {preset['desc']}")
    print(f"  Start time:  {preset['start_time']}")
    print(f"{'='*60}\n")

    # Stop any running simulation
    try:
        requests.post(f"{DASHBOARD}/api/commands/", json={"command": "stop"}, timeout=5)
        time.sleep(0.5)
    except Exception:
        pass

    # Start with the preset time, reasonable speed
    try:
        resp = requests.post(
            f"{DASHBOARD}/api/commands/",
            json={
                "command": "start",
                "start_time": preset["start_time"],
                "step_size_seconds": 10,
                "replay_speed": 5.0,
            },
            timeout=10,
        )
        print(f"  Dashboard response: {resp.status_code}")
    except Exception as e:
        print(f"  ✗ Failed to send command: {e}")
        return

    # Wait and show position updates
    print("\n  Tracking satellite position...\n")
    for i in range(6):
        time.sleep(2)
        pos, ts = get_position()
        if pos:
            lon, lat, alt = pos
            print(f"  [{ts}]  lon={lon:.2f}  lat={lat:.2f}  alt={alt:.1f} km")


def scan_presets():
    """Try each preset and show where the satellite ends up."""
    print("\n  Available presets:\n")
    for name, info in PRESETS.items():
        print(f"    {name:12s} — {info['desc']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python steer_simsat.py <preset>")
        scan_presets()

        # Show current position
        print("  Current satellite position:")
        pos, ts = get_position()
        if pos:
            lon, lat, alt = pos
            print(f"    [{ts}]  lon={lon:.2f}  lat={lat:.2f}  alt={alt:.1f} km")
        sys.exit(0)

    steer_to(sys.argv[1])
