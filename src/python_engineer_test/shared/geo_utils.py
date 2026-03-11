"""Geographic coordinate utilities for drone video processing.

Provides utilities for converting between meters and geographic coordinates,
and calculating ground coverage from camera geometry.
"""

import math


class CoordinateConverter:
    """Converts between meters and geographic coordinates.

    Uses WGS84 Earth radius approximations. 1° latitude is constant at
    ~111,132 meters, while 1° longitude varies with latitude.
    """

    METERS_PER_DEGREE_LATITUDE = 111132.0  # meters per degree latitude (constant)

    @staticmethod
    def meters_per_degree_latitude() -> float:
        """Return meters per degree latitude.

        Returns:
            Meters per degree latitude (~111,132m).
        """
        return CoordinateConverter.METERS_PER_DEGREE_LATITUDE

    @staticmethod
    def meters_per_degree_longitude(latitude_deg: float) -> float:
        """Return meters per degree longitude at given latitude.

        Args:
            latitude_deg: Latitude in degrees.

        Returns:
            Meters per degree longitude (varies with latitude).
        """
        latitude_rad = math.radians(latitude_deg)
        return CoordinateConverter.METERS_PER_DEGREE_LATITUDE * math.cos(latitude_rad)

    @staticmethod
    def meters_to_degrees(dx_m: float, dy_m: float, latitude_deg: float) -> tuple[float, float]:
        """Convert meter offsets to degree offsets.

        Args:
            dx_m: East-West offset in meters (positive = east).
            dy_m: North-South offset in meters (positive = north).
            latitude_deg: Reference latitude in degrees.

        Returns:
            Tuple of (delta_longitude, delta_latitude) in degrees.
        """
        meters_per_deg_lon = CoordinateConverter.meters_per_degree_longitude(latitude_deg)

        dlon = dx_m / meters_per_deg_lon if meters_per_deg_lon != 0 else 0.0
        dlat = dy_m / CoordinateConverter.METERS_PER_DEGREE_LATITUDE

        return (dlon, dlat)

    @staticmethod
    def degrees_to_meters(
        dlon_deg: float, dlat_deg: float, latitude_deg: float
    ) -> tuple[float, float]:
        """Convert degree offsets to meter offsets.

        Args:
            dlon_deg: Longitude offset in degrees.
            dlat_deg: Latitude offset in degrees.
            latitude_deg: Reference latitude in degrees.

        Returns:
            Tuple of (dx_m, dy_m) in meters.
        """
        meters_per_deg_lon = CoordinateConverter.meters_per_degree_longitude(latitude_deg)

        dx_m = dlon_deg * meters_per_deg_lon
        dy_m = dlat_deg * CoordinateConverter.METERS_PER_DEGREE_LATITUDE

        return (dx_m, dy_m)


class DroneCameraGeometry:
    """Calculates ground coverage and GSD from drone camera geometry.

    Assumes nadir (straight down) viewing geometry.
    """

    def __init__(
        self,
        altitude_m: float,
        video_width_px: int,
        video_height_px: int,
        focal_length_mm: float = 24.0,
        sensor_width_mm: float = 17.3,
        sensor_height_mm: float = 13.0,
    ):
        """Initialize drone camera geometry.

        Args:
            altitude_m: Drone altitude above ground in meters.
            video_width_px: Video frame width in pixels.
            video_height_px: Video frame height in pixels.
            focal_length_mm: Lens focal length in mm (default: 24mm equivalent).
            sensor_width_mm: Sensor width in mm (default: 4/3" sensor = 17.3mm).
            sensor_height_mm: Sensor height in mm (default: 4/3" sensor = 13.0mm).
        """
        self.altitude_m = altitude_m
        self.video_width_px = video_width_px
        self.video_height_px = video_height_px
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm

    def calculate_fov(self) -> tuple[float, float]:
        """Calculate horizontal and vertical field of view.

        Uses: FOV = 2 × atan(sensor_size / (2 × focal_length))

        Returns:
            Tuple of (fov_h_deg, fov_v_deg) in degrees.
        """
        fov_h_rad = 2 * math.atan(self.sensor_width_mm / (2 * self.focal_length_mm))
        fov_v_rad = 2 * math.atan(self.sensor_height_mm / (2 * self.focal_length_mm))

        return (math.degrees(fov_h_rad), math.degrees(fov_v_rad))

    def calculate_ground_coverage(self) -> tuple[float, float]:
        """Calculate ground coverage dimensions at Nadir.

        Uses: ground_width = 2 × altitude × tan(FOV / 2)

        Returns:
            Tuple of (ground_width_m, ground_height_m) in meters.
        """
        fov_h_deg, fov_v_deg = self.calculate_fov()

        fov_h_rad = math.radians(fov_h_deg)
        fov_v_rad = math.radians(fov_v_deg)

        ground_width = 2 * self.altitude_m * math.tan(fov_h_rad / 2)
        ground_height = 2 * self.altitude_m * math.tan(fov_v_rad / 2)

        return (ground_width, ground_height)

    def calculate_gsd(self) -> tuple[float, float]:
        """Calculate Ground Sample Distance (GSD).

        GSD = ground_dimension / pixel_count

        Returns:
            Tuple of (gsd_h_m_per_px, gsd_v_m_per_px) in meters per pixel.
        """
        ground_width, ground_height = self.calculate_ground_coverage()

        gsd_h = ground_width / self.video_width_px
        gsd_v = ground_height / self.video_height_px

        return (gsd_h, gsd_v)

    def get_scale_degrees_per_pixel(self, latitude_deg: float) -> tuple[float, float]:
        """Get degrees per pixel scale at given latitude.

        Args:
            latitude_deg: Reference latitude in degrees.

        Returns:
            Tuple of (scale_lon_deg_per_px, scale_lat_deg_per_px).
        """
        gsd_h, gsd_v = self.calculate_gsd()

        meters_per_deg_lon = CoordinateConverter.meters_per_degree_longitude(latitude_deg)
        meters_per_deg_lat = CoordinateConverter.meters_per_degree_latitude()

        scale_lon = gsd_h / meters_per_deg_lon
        scale_lat = gsd_v / meters_per_deg_lat

        return (scale_lon, scale_lat)
