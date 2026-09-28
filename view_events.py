"""
Quick way to check what's in the database without installing a separate
SQLite browser.

Usage:
    python view_events.py
"""

from db import get_session, VehicleEvent


def main():
    session = get_session()
    try:
        events = session.query(VehicleEvent).order_by(VehicleEvent.timestamp.desc()).all()
        if not events:
            print("No events logged yet.")
            return
        for e in events:
            vtype = e.vehicle_type or "?"
            print(f"{e.timestamp}  {e.direction:5s}  {vtype:10s}  {e.plate_number:12s}  "
                  f"conf={e.confidence:.2f}  track={e.track_id}  source={e.source}")
        print(f"\n{len(events)} total event(s).")
    finally:
        session.close()


if __name__ == "__main__":
    main()