"""Cursor on Target (CoT) Protocol Implementation.

CoT is an XML-based protocol for sharing situational awareness data.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import TypedDict
from xml.etree.ElementTree import Element, SubElement, tostring

logger = logging.getLogger(__name__)


class ParsedMessage(TypedDict):
    latitude: float
    longitude: float
    description: str


def format_cot_message(
    latitude: float,
    longitude: float,
    target_description: str,
    cot_type: str = "a-h-G",
    how: str = "h-g-i",
    uid: str | None = None,
) -> str:
    """Format a CoT message for ATAK.

    Args:
        latitude: Target latitude
        longitude: Target longitude
        target_description: Description of the target
        cot_type: CoT event type (default: a-h-G for hostile ground)
        how: Source of information (default: h-g-i for human/guest/internet)
        uid: Unique identifier (auto-generated if not provided)

    Returns:
        XML string in CoT format

    CoT Type Reference:
        a = Atom (point)
        h = Hostile
        G = Ground

        Examples:
        - a-h-G: Hostile ground unit
        - a-f-G: Friendly ground unit
        - a-n-G: Neutral ground unit
        - a-u-G: Unknown ground unit
    """
    if uid is None:
        uid = str(uuid.uuid4())

    now = datetime.now(UTC)
    stale = datetime.fromtimestamp(now.timestamp() + 3600, tz=UTC)

    # Create event element
    event = Element("event")
    event.set("version", "2.0")
    event.set("uid", uid)
    event.set("type", cot_type)
    event.set("time", now.isoformat(timespec="milliseconds").replace("+00:00", "Z"))
    event.set("start", now.isoformat(timespec="milliseconds").replace("+00:00", "Z"))
    event.set("stale", stale.isoformat(timespec="milliseconds").replace("+00:00", "Z"))
    event.set("how", how)

    # Point element
    point = SubElement(event, "point")
    point.set("lat", str(latitude))
    point.set("lon", str(longitude))
    point.set("hae", "0")  # Height above ellipsoid
    point.set("ce", "10")  # Circular error (meters)
    point.set("le", "10")  # Linear error (meters)

    # Detail element with description
    detail = SubElement(event, "detail")

    # Contact info
    contact = SubElement(detail, "contact")
    contact.set("callsign", target_description)

    # Remarks
    remarks = SubElement(detail, "remarks")
    remarks.text = target_description

    # Usericon for visualization
    usericon = SubElement(detail, "usericon")
    usericon.set("iconsetpath", f"359f1b2a-8d52-4dc7-9c2a-1c7a7a7a7a7a/Target/{cot_type}")

    return tostring(event, encoding="unicode")


def parse_signal_message(message: str, *, order: str = "lonlat") -> ParsedMessage:
    parts = message.strip().split(maxsplit=2)

    if len(parts) < 3:
        raise ValueError(
            f"Invalid message format: {message!r}. Expected '<coord1> <coord2> '"
            "'<target_description>'."
        )

    order = order.strip().lower()
    if order not in {"lonlat", "latlon"}:
        raise ValueError(f"Invalid order: {order!r}. Expected 'lonlat' or 'latlon'.")

    try:
        first = float(parts[0])
        second = float(parts[1])
    except ValueError as e:
        raise ValueError(f"Invalid coordinates in message: {parts[0]}, {parts[1]}") from e

    if order == "lonlat":
        longitude = first
        latitude = second
    else:
        latitude = first
        longitude = second

    # Validate coordinate ranges
    if not -90 <= latitude <= 90:
        raise ValueError(f"Latitude out of range: {latitude}")
    if not -180 <= longitude <= 180:
        raise ValueError(f"Longitude out of range: {longitude}")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "description": parts[2],
    }


if __name__ == "__main__":
    # Test CoT message generation
    cot = format_cot_message(48.567123, 39.87897, "tank")
    logger.info(cot)
