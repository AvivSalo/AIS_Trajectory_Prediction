#!/usr/bin/env python3
"""
Shared utility functions for AIS data conversion.
Contains common geographic calculations and data parsing functions.
"""

import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points in meters using Haversine formula.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2 in degrees (0-360).

    Args:
        lat1, lon1: Starting point coordinates (degrees)
        lat2, lon2: End point coordinates (degrees)

    Returns:
        Bearing in degrees (0-360, where 0=North, 90=East)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_cpa_tcpa(own_lat, own_lon, own_sog, own_cog,
                        target_lat, target_lon, target_sog, target_cog):
    """
    Calculate Closest Point of Approach (CPA) and Time to CPA (TCPA).

    Maritime collision avoidance metrics based on vessel positions and velocities.

    Args:
        own_lat, own_lon: Own vessel position (degrees)
        own_sog: Own vessel Speed Over Ground (knots)
        own_cog: Own vessel Course Over Ground (degrees)
        target_lat, target_lon: Target vessel position (degrees)
        target_sog: Target vessel Speed Over Ground (knots)
        target_cog: Target vessel Course Over Ground (degrees)

    Returns:
        tuple: (cpa, tcpa)
            cpa: Distance at closest approach (meters)
            tcpa: Time to closest approach (seconds)
    """
    # Convert speeds from knots to m/s
    own_v = own_sog * 0.514444
    target_v = target_sog * 0.514444

    # Convert headings to velocity components
    own_vx = own_v * np.sin(np.radians(own_cog))
    own_vy = own_v * np.cos(np.radians(own_cog))
    target_vx = target_v * np.sin(np.radians(target_cog))
    target_vy = target_v * np.cos(np.radians(target_cog))

    # Relative velocity
    rel_vx = target_vx - own_vx
    rel_vy = target_vy - own_vy
    rel_v_squared = rel_vx**2 + rel_vy**2

    # If relative velocity is near zero, vessels are moving together
    if rel_v_squared < 1e-6:
        distance = haversine_distance(own_lat, own_lon, target_lat, target_lon)
        return distance, 0.0

    # Convert lat/lon to relative position (approximate for small distances)
    lat_to_m = 111320  # meters per degree latitude
    lon_to_m = 111320 * np.cos(np.radians((own_lat + target_lat) / 2))

    rel_x = (target_lon - own_lon) * lon_to_m
    rel_y = (target_lat - own_lat) * lat_to_m

    # Time to CPA: when relative position is perpendicular to relative velocity
    tcpa = -(rel_x * rel_vx + rel_y * rel_vy) / rel_v_squared

    # If TCPA is negative, CPA is in the past (vessels diverging)
    if tcpa < 0:
        tcpa = 0
        cpa = haversine_distance(own_lat, own_lon, target_lat, target_lon)
    else:
        # Calculate position at TCPA
        cpa_x = rel_x + rel_vx * tcpa
        cpa_y = rel_y + rel_vy * tcpa
        cpa = np.sqrt(cpa_x**2 + cpa_y**2)

    return cpa, tcpa


def parse_danish_timestamp(timestamp_str):
    """
    Convert Danish timestamp format to ISO format.

    Args:
        timestamp_str: Danish format "DD/MM/YYYY HH:MM:SS"

    Returns:
        ISO format "YYYY-MM-DD HH:MM:SS" or None if parsing fails
    """
    try:
        dt = datetime.strptime(timestamp_str, "%d/%m/%Y %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return None


def parse_influxdb_timestamp(timestamp_str):
    """
    Parse InfluxDB timestamp format (already ISO format).

    Args:
        timestamp_str: ISO format "YYYY-MM-DD HH:MM:SS"

    Returns:
        ISO format "YYYY-MM-DD HH:MM:SS" or None if invalid
    """
    try:
        # Validate it's a proper ISO timestamp
        datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return timestamp_str
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return None


def latlon_to_meters(lat, lon, reference_lat, reference_lon):
    """
    Convert lat/lon to relative x,y in meters from a reference point.

    Args:
        lat, lon: Point coordinates (degrees)
        reference_lat, reference_lon: Reference point coordinates (degrees)

    Returns:
        tuple: (x_meters, y_meters) relative position in meters
    """
    lat_diff = lat - reference_lat
    lon_diff = lon - reference_lon

    # Convert to meters
    x_meters = lon_diff * 111320 * np.cos(np.radians(reference_lat))
    y_meters = lat_diff * 110540

    return x_meters, y_meters


def knots_to_ms(speed_knots):
    """Convert speed from knots to meters per second."""
    return speed_knots * 0.514444


def course_speed_to_velocity(course_degrees, speed_ms):
    """
    Convert course and speed to velocity components.

    Args:
        course_degrees: Course Over Ground in degrees (0=North, 90=East)
        speed_ms: Speed in meters per second

    Returns:
        tuple: (vx, vy) velocity components in m/s
    """
    heading_rads = np.radians(course_degrees)
    vx = speed_ms * np.sin(heading_rads)
    vy = speed_ms * np.cos(heading_rads)
    return vx, vy
