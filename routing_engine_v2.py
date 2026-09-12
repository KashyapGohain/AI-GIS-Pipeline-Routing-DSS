# ============================================================
# ROUTING ENGINE V2
# ============================================================
# Dynamic Multi-Route GIS Routing Engine
#
# Project:
# AI + GIS Pipeline Routing Decision Support System
#
# V2 FEATURES
# ------------------------------------------------------------
# 1. Accept user-defined WGS84 coordinates
# 2. Convert coordinates -> raster row/column
# 3. Validate / snap start and destination
# 4. Generate four routing alternatives
# 5. Calculate physical geodesic distance
# 6. Calculate economic/base cost
# 7. Calculate environmental impact
# 8. Calculate route statistics
# 9. Compare routes
# 10. Save JSON result
#
# ROUTES
# ------------------------------------------------------------
# Route 1 = Minimum Cost
# Route 2 = Environmental Impact
# Route 3 = Balanced
# Route 4 = Shortest Feasible
#
# IMPORTANT
# ------------------------------------------------------------
# The original V6 A* engine is NOT modified.
# This module imports the existing AStar implementation.
#
# ============================================================


import os
import sys
import json
import math
import time

import numpy as np
import rasterio
from pyproj import Geod, Transformer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"I:/Kashyap/Route"

CODES_DIR = os.path.join(
    BASE_DIR,
    "codes"
)

MULTI_CRITERIA_DIR = os.path.join(
    CODES_DIR,
    "multi_criteria"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "OUTPUT",
    "dashboard"
)


# ============================================================
# INPUT FILES
# ============================================================

FEATURE_PATH = os.path.join(
    CODES_DIR,
    "feature_stack.npy"
)

BASE_COST_PATH = os.path.join(
    CODES_DIR,
    "cost_surface.npy"
)

ENVIRONMENTAL_COST_PATH = os.path.join(
    MULTI_CRITERIA_DIR,
    "environmental_cost_surface.npy"
)

MINIMUM_COST_PATH = os.path.join(
    MULTI_CRITERIA_DIR,
    "minimum_cost_surface.npy"
)

BALANCED_COST_PATH = os.path.join(
    MULTI_CRITERIA_DIR,
    "balanced_cost_surface.npy"
)

SHORTEST_FEASIBLE_PATH = os.path.join(
    MULTI_CRITERIA_DIR,
    "shortest_feasible_cost_surface.npy"
)

DEM_PATH = os.path.join(
    BASE_DIR,
    "DEM",
    "DEM_ASSAM.tif"
)

if not os.path.exists(DEM_PATH):

    DEM_PATH = os.path.join(
        BASE_DIR,
        "DEM_ASSAM.tif"
    )


# ============================================================
# PATHFINDING V6
# ============================================================

if CODES_DIR not in sys.path:

    sys.path.insert(
        0,
        CODES_DIR
    )


try:

    from pathfinding_v6 import AStar

except ImportError as e:

    raise ImportError(
        "\nCould not import pathfinding_v6.py.\n"
        f"Expected location:\n{CODES_DIR}\n\n"
        f"Original error:\n{e}"
    )


# ============================================================
# GEODESIC
# ============================================================

GEOD = Geod(
    ellps="WGS84"
)


# ============================================================
# ROUTE DEFINITIONS
# ============================================================

ROUTE_DEFINITIONS = {

    "route_1": {
        "id": 1,
        "name": "Route 1 - Minimum Cost",
        "surface": "minimum"
    },

    "route_2": {
        "id": 2,
        "name": "Route 2 - Environmental Impact",
        "surface": "environmental"
    },

    "route_3": {
        "id": 3,
        "name": "Route 3 - Balanced",
        "surface": "balanced"
    },

    "route_4": {
        "id": 4,
        "name": "Route 4 - Shortest Feasible",
        "surface": "shortest"
    }
}


# ============================================================
# ROUTING ENGINE
# ============================================================

class RoutingEngineV2:

    def __init__(
        self,
        feature_path=FEATURE_PATH,
        base_cost_path=BASE_COST_PATH,
        environmental_path=ENVIRONMENTAL_COST_PATH,
        minimum_path=MINIMUM_COST_PATH,
        balanced_path=BALANCED_COST_PATH,
        shortest_path=SHORTEST_FEASIBLE_PATH,
        dem_path=DEM_PATH,
        max_snap_pixels=25
    ):

        print("=" * 70)
        print("INITIALIZING ROUTING ENGINE V2")
        print("=" * 70)

        self.feature_path = feature_path
        self.base_cost_path = base_cost_path
        self.environmental_path = environmental_path
        self.minimum_path = minimum_path
        self.balanced_path = balanced_path
        self.shortest_path = shortest_path
        self.dem_path = dem_path

        self.max_snap_pixels = max_snap_pixels

        # ----------------------------------------------------
        # LOAD FEATURE STACK
        # ----------------------------------------------------

        if not os.path.exists(
            feature_path
        ):

            raise FileNotFoundError(
                f"Feature stack not found:\n{feature_path}"
            )

        self.features = np.load(
            feature_path
        )

        print(
            "\nFeature Stack:",
            self.features.shape
        )

        # ----------------------------------------------------
        # LOAD BASE COST
        # ----------------------------------------------------

        self.base_cost = self._load_surface(
            base_cost_path,
            "Base Cost Surface"
        )

        # ----------------------------------------------------
        # LOAD MULTI-CRITERIA SURFACES
        # ----------------------------------------------------

        self.environmental_cost = self._load_surface(
            environmental_path,
            "Environmental Cost Surface"
        )

        self.minimum_cost = self._load_surface(
            minimum_path,
            "Minimum Cost Surface"
        )

        self.balanced_cost = self._load_surface(
            balanced_path,
            "Balanced Cost Surface"
        )

        self.shortest_cost = self._load_surface(
            shortest_path,
            "Shortest Feasible Cost Surface"
        )

        # ----------------------------------------------------
        # GRID DIMENSIONS
        # ----------------------------------------------------

        self.height = self.base_cost.shape[0]

        self.width = self.base_cost.shape[1]

        expected_shape = (
            self.height,
            self.width
        )

        surfaces = {

            "environmental":
            self.environmental_cost,

            "minimum":
            self.minimum_cost,

            "balanced":
            self.balanced_cost,

            "shortest":
            self.shortest_cost
        }

        for name, surface in surfaces.items():

            if surface.shape != expected_shape:

                raise ValueError(
                    f"\n{name} surface shape mismatch.\n"
                    f"Expected: {expected_shape}\n"
                    f"Received: {surface.shape}"
                )

        # ----------------------------------------------------
        # DEM / RASTER REFERENCE
        # ----------------------------------------------------

        if not os.path.exists(
            dem_path
        ):

            raise FileNotFoundError(
                f"\nDEM not found:\n{dem_path}"
            )

        with rasterio.open(
            dem_path
        ) as src:

            self.raster_crs = src.crs

            self.transform = src.transform

            self.bounds = src.bounds

            self.raster_height = src.height

            self.raster_width = src.width

            self.nodata = src.nodata

        print(
            "Raster CRS:",
            self.raster_crs
        )

        print(
            "Raster Size:",
            self.raster_height,
            "x",
            self.raster_width
        )

        print(
            "Raster Bounds:",
            self.bounds
        )

        # ----------------------------------------------------
        # DIMENSION CHECK
        # ----------------------------------------------------

        if (
            self.raster_height != self.height
            or
            self.raster_width != self.width
        ):

            raise ValueError(
                "\nRouting surfaces and DEM dimensions "
                "do not match.\n"
                f"Cost surface: {self.height} x {self.width}\n"
                f"DEM: {self.raster_height} x "
                f"{self.raster_width}"
            )

        # ----------------------------------------------------
        # COORDINATE TRANSFORMERS
        # ----------------------------------------------------

        self.to_raster = Transformer.from_crs(
            "EPSG:4326",
            self.raster_crs,
            always_xy=True
        )

        self.to_wgs84 = Transformer.from_crs(
            self.raster_crs,
            "EPSG:4326",
            always_xy=True
        )

        # ----------------------------------------------------
        # ROUTING ENGINES
        # ----------------------------------------------------

        print(
            "\nInitializing routing engines..."
        )

        self.astar_minimum = AStar(
            self.minimum_cost
        )

        self.astar_environmental = AStar(
            self.environmental_cost
        )

        self.astar_balanced = AStar(
            self.balanced_cost
        )

        self.astar_shortest = AStar(
            self.shortest_cost
        )

        print(
            "Minimum Cost A* ........ READY"
        )

        print(
            "Environmental A* ....... READY"
        )

        print(
            "Balanced A* ............ READY"
        )

        print(
            "Shortest Feasible A* ... READY"
        )

        print("=" * 70)


    # ========================================================
    # LOAD SURFACE
    # ========================================================

    def _load_surface(
        self,
        path,
        name
    ):

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                f"\n{name} not found:\n{path}"
            )

        surface = np.load(
            path
        )

        print(
            f"{name}:",
            surface.shape
        )

        if surface.ndim != 2:

            raise ValueError(
                f"{name} must be a 2D raster surface."
            )

        return surface.astype(
            np.float64,
            copy=False
        )


    # ========================================================
    # VALIDATE COORDINATE
    # ========================================================

    def validate_coordinate(
        self,
        lat,
        lon
    ):

        try:

            lat = float(lat)

            lon = float(lon)

        except Exception:

            raise ValueError(
                "Latitude and longitude must be numeric."
            )

        if not np.isfinite(lat):

            raise ValueError(
                "Latitude is not finite."
            )

        if not np.isfinite(lon):

            raise ValueError(
                "Longitude is not finite."
            )

        if not -90 <= lat <= 90:

            raise ValueError(
                f"Invalid latitude: {lat}"
            )

        if not -180 <= lon <= 180:

            raise ValueError(
                f"Invalid longitude: {lon}"
            )

        return lat, lon


    # ========================================================
    # WGS84 -> PIXEL
    # ========================================================

    def coordinate_to_pixel(
        self,
        lat,
        lon
    ):

        lat, lon = self.validate_coordinate(
            lat,
            lon
        )

        x, y = self.to_raster.transform(
            lon,
            lat
        )

        row, col = rasterio.transform.rowcol(
            self.transform,
            x,
            y
        )

        return int(row), int(col)


    # ========================================================
    # PIXEL -> WGS84
    # ========================================================

    def pixel_to_coordinate(
        self,
        row,
        col
    ):

        x, y = rasterio.transform.xy(
            self.transform,
            row,
            col,
            offset="center"
        )

        lon, lat = self.to_wgs84.transform(
            x,
            y
        )

        return float(lat), float(lon)


    # ========================================================
    # INSIDE RASTER
    # ========================================================

    def is_inside(
        self,
        row,
        col
    ):

        return (
            0 <= row < self.height
            and
            0 <= col < self.width
        )


    # ========================================================
    # VALID CELL
    # ========================================================

    def is_valid_cell(
        self,
        row,
        col,
        surface=None
    ):

        if not self.is_inside(
            row,
            col
        ):

            return False

        if surface is None:

            surface = self.base_cost

        value = surface[
            row,
            col
        ]

        if not np.isfinite(
            value
        ):

            return False

        if value < 0:

            return False

        return True


    # ========================================================
    # FIND NEAREST VALID CELL
    # ========================================================

    def find_nearest_valid_cell(
        self,
        row,
        col,
        surface,
        max_radius=None
    ):

        if max_radius is None:

            max_radius = self.max_snap_pixels

        if self.is_valid_cell(
            row,
            col,
            surface
        ):

            return (
                row,
                col,
                0.0
            )

        best = None

        best_distance = float(
            "inf"
        )

        for radius in range(
            1,
            max_radius + 1
        ):

            row_min = max(
                0,
                row - radius
            )

            row_max = min(
                self.height - 1,
                row + radius
            )

            col_min = max(
                0,
                col - radius
            )

            col_max = min(
                self.width - 1,
                col + radius
            )

            for r in range(
                row_min,
                row_max + 1
            ):

                for c in range(
                    col_min,
                    col_max + 1
                ):

                    if not self.is_valid_cell(
                        r,
                        c,
                        surface
                    ):

                        continue

                    distance = math.sqrt(
                        (r - row) ** 2
                        +
                        (c - col) ** 2
                    )

                    if distance < best_distance:

                        best_distance = distance

                        best = (
                            r,
                            c
                        )

            if best is not None:

                break

        if best is None:

            return None

        return (
            best[0],
            best[1],
            best_distance
        )


    # ========================================================
    # PREPARE LOCATION
    # ========================================================

    def prepare_location(
        self,
        lat,
        lon,
        name
    ):

        row, col = self.coordinate_to_pixel(
            lat,
            lon
        )

        if not self.is_inside(
            row,
            col
        ):

            raise ValueError(
                f"\n{name} is outside the project raster.\n"
                f"Coordinate: {lat}, {lon}\n"
                f"Pixel: {row}, {col}"
            )

        snapped = self.find_nearest_valid_cell(
            row,
            col,
            self.base_cost
        )

        if snapped is None:

            raise ValueError(
                f"\n{name} is not routable and "
                "no nearby valid cell was found."
            )

        snap_row, snap_col, snap_pixels = snapped

        snap_lat, snap_lon = (
            self.pixel_to_coordinate(
                snap_row,
                snap_col
            )
        )

        _, _, snap_distance_m = GEOD.inv(
            lon,
            lat,
            snap_lon,
            snap_lat
        )

        return {

            "input_lat":
            float(lat),

            "input_lon":
            float(lon),

            "input_row":
            int(row),

            "input_col":
            int(col),

            "row":
            int(snap_row),

            "col":
            int(snap_col),

            "snapped":
            bool(snap_pixels > 0),

            "snap_distance_pixels":
            float(snap_pixels),

            "snap_distance_m":
            float(abs(snap_distance_m)),

            "snap_lat":
            float(snap_lat),

            "snap_lon":
            float(snap_lon)
        }


    # ========================================================
    # ROUTE PIXEL DISTANCE
    # ========================================================

    def pixel_distance(
        self,
        route
    ):

        if route is None:

            return 0.0

        distance = 0.0

        for i in range(
            len(route) - 1
        ):

            r1, c1 = route[i]

            r2, c2 = route[i + 1]

            distance += math.sqrt(
                (r2 - r1) ** 2
                +
                (c2 - c1) ** 2
            )

        return float(
            distance
        )


    # ========================================================
    # GEODESIC DISTANCE
    # ========================================================

    def geodesic_distance(
        self,
        route
    ):

        if route is None:

            return 0.0

        if len(route) < 2:

            return 0.0

        total = 0.0

        for i in range(
            len(route) - 1
        ):

            r1, c1 = route[i]

            r2, c2 = route[i + 1]

            lat1, lon1 = (
                self.pixel_to_coordinate(
                    r1,
                    c1
                )
            )

            lat2, lon2 = (
                self.pixel_to_coordinate(
                    r2,
                    c2
                )
            )

            _, _, distance = GEOD.inv(
                lon1,
                lat1,
                lon2,
                lat2
            )

            total += abs(
                distance
            )

        return float(
            total
        )


    # ========================================================
    # SURFACE SUM
    # ========================================================

    def route_surface_sum(
        self,
        route,
        surface
    ):

        if route is None:

            return 0.0

        values = []

        for row, col in route:

            value = surface[
                row,
                col
            ]

            if np.isfinite(
                value
            ):

                values.append(
                    float(value)
                )

        if len(values) == 0:

            return 0.0

        return float(
            np.sum(values)
        )


    # ========================================================
    # SURFACE MEAN
    # ========================================================

    def route_surface_mean(
        self,
        route,
        surface
    ):

        if route is None:

            return 0.0

        values = []

        for row, col in route:

            value = surface[
                row,
                col
            ]

            if np.isfinite(
                value
            ):

                values.append(
                    float(value)
                )

        if len(values) == 0:

            return 0.0

        return float(
            np.mean(values)
        )


    # ========================================================
    # ROUTE STATISTICS
    # ========================================================

    def calculate_statistics(
        self,
        route,
        routing_surface
    ):

        if route is None:

            return None

        cells = len(
            route
        )

        unique_cells = len(
            set(route)
        )

        repeated_cells = (
            cells
            -
            unique_cells
        )

        pixel_dist = (
            self.pixel_distance(
                route
            )
        )

        geodesic_m = (
            self.geodesic_distance(
                route
            )
        )

        total_routing_cost = (
            self.route_surface_sum(
                route,
                routing_surface
            )
        )

        mean_routing_cost = (
            total_routing_cost / cells
            if cells > 0
            else 0.0
        )

        economic_cost = (
            self.route_surface_sum(
                route,
                self.base_cost
            )
        )

        economic_mean = (
            economic_cost / cells
            if cells > 0
            else 0.0
        )

        environmental_impact = (
            self.route_surface_sum(
                route,
                self.environmental_cost
            )
        )

        environmental_mean = (
            environmental_impact / cells
            if cells > 0
            else 0.0
        )

        return {

            "route_cells":
            int(cells),

            "unique_cells":
            int(unique_cells),

            "repeated_cells":
            int(repeated_cells),

            "pixel_distance":
            float(pixel_dist),

            "geodesic_distance_m":
            float(geodesic_m),

            "geodesic_distance_km":
            float(
                geodesic_m / 1000.0
            ),

            "routing_surface_cost":
            float(total_routing_cost),

            "routing_surface_mean":
            float(mean_routing_cost),

            "economic_cost":
            float(economic_cost),

            "economic_mean_cell_cost":
            float(economic_mean),

            "environmental_impact_index":
            float(environmental_impact),

            "environmental_mean_cell_impact":
            float(environmental_mean),

            "repetition_rate":
            float(
                repeated_cells / cells
                if cells > 0
                else 0.0
            )
        }


    # ========================================================
    # RUN ONE ROUTE
    # ========================================================

    def _run_single_route(
        self,
        route_key,
        start,
        goal
    ):

        definition = ROUTE_DEFINITIONS[
            route_key
        ]

        route_name = definition[
            "name"
        ]

        surface_type = definition[
            "surface"
        ]

        # ----------------------------------------------------
        # Select routing surface
        # ----------------------------------------------------

        if surface_type == "minimum":

            surface = self.minimum_cost

            astar = self.astar_minimum

        elif surface_type == "environmental":

            surface = self.environmental_cost

            astar = self.astar_environmental

        elif surface_type == "balanced":

            surface = self.balanced_cost

            astar = self.astar_balanced

        elif surface_type == "shortest":

            surface = self.shortest_cost

            astar = self.astar_shortest

        else:

            raise ValueError(
                f"Unknown route surface: {surface_type}"
            )

        print(
            "\n" + "-" * 70
        )

        print(
            route_name
        )

        print(
            "-" * 70
        )

        print(
            "Start:",
            start
        )

        print(
            "Goal:",
            goal
        )

        start_time = time.time()

        # ----------------------------------------------------
        # Run A*
        # ----------------------------------------------------

        route = astar.search(
            start,
            goal
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        if route is None:

            print(
                "No route found."
            )

            return {

                "route_id":
                definition["id"],

                "route_key":
                route_key,

                "route_name":
                route_name,

                "status":
                "failed",

                "route":
                None,

                "processing_time_seconds":
                float(elapsed)
            }

        statistics = (
            self.calculate_statistics(
                route,
                surface
            )
        )

        statistics[
            "processing_time_seconds"
        ] = float(
            elapsed
        )

        statistics[
            "route_id"
        ] = definition["id"]

        statistics[
            "route_key"
        ] = route_key

        statistics[
            "route_name"
        ] = route_name

        statistics[
            "status"
        ] = "success"

        print(
            "Destination reached!"
        )

        print(
            "Route cells:",
            statistics[
                "route_cells"
            ]
        )

        print(
            "Physical distance:",
            round(
                statistics[
                    "geodesic_distance_km"
                ],
                3
            ),
            "km"
        )

        print(
            "Economic cost:",
            round(
                statistics[
                    "economic_cost"
                ],
                3
            )
        )

        print(
            "Environmental impact:",
            round(
                statistics[
                    "environmental_impact_index"
                ],
                3
            )
        )

        print(
            "Processing time:",
            round(
                elapsed,
                3
            ),
            "seconds"
        )

        statistics[
            "route"
        ] = route

        return statistics


    # ========================================================
    # ROUTE COMPARISON
    # ========================================================

    def compare_routes(
        self,
        route_results,
        direct_distance_m
    ):

        comparison = []

        for route_key, result in route_results.items():

            if result[
                "status"
            ] != "success":

                continue

            geodesic_m = result[
                "geodesic_distance_m"
            ]

            if direct_distance_m > 0:

                ratio = (
                    geodesic_m
                    /
                    direct_distance_m
                )

                straightness = (
                    direct_distance_m
                    /
                    geodesic_m
                )

            else:

                ratio = 0.0

                straightness = 0.0

            result[
                "direct_distance_m"
            ] = float(
                direct_distance_m
            )

            result[
                "direct_distance_km"
            ] = float(
                direct_distance_m / 1000.0
            )

            result[
                "route_to_direct_ratio"
            ] = float(
                ratio
            )

            result[
                "straightness"
            ] = float(
                straightness
            )

            comparison.append(
                {
                    "route_id":
                    result["route_id"],

                    "route_key":
                    route_key,

                    "route_name":
                    result["route_name"],

                    "geodesic_distance_km":
                    result["geodesic_distance_km"],

                    "economic_cost":
                    result["economic_cost"],

                    "environmental_impact_index":
                    result[
                        "environmental_impact_index"
                    ],

                    "routing_surface_cost":
                    result[
                        "routing_surface_cost"
                    ],

                    "combined_cells":
                    result["route_cells"],

                    "unique_cells":
                    result["unique_cells"],

                    "repeated_cells":
                    result["repeated_cells"],

                    "route_to_direct_ratio":
                    result[
                        "route_to_direct_ratio"
                    ],

                    "straightness":
                    result[
                        "straightness"
                    ],

                    "processing_time_seconds":
                    result[
                        "processing_time_seconds"
                    ]
                }
            )

        comparison.sort(
            key=lambda x: x["route_id"]
        )

        return comparison


    # ========================================================
    # ROUTE SIMILARITY
    # ========================================================

    def calculate_similarity(
        self,
        route_results
    ):

        successful = {}

        for key, result in route_results.items():

            if (
                result["status"]
                ==
                "success"
            ):

                successful[key] = set(
                    result["route"]
                )

        similarity = []

        keys = list(
            successful.keys()
        )

        for i in range(
            len(keys)
        ):

            for j in range(
                i + 1,
                len(keys)
            ):

                key_a = keys[i]

                key_b = keys[j]

                set_a = successful[
                    key_a
                ]

                set_b = successful[
                    key_b
                ]

                shared = (
                    set_a
                    &
                    set_b
                )

                union = (
                    set_a
                    |
                    set_b
                )

                if len(union) > 0:

                    jaccard = (
                        len(shared)
                        /
                        len(union)
                    )

                else:

                    jaccard = 0.0

                shorter = min(
                    len(set_a),
                    len(set_b)
                )

                if shorter > 0:

                    overlap = (
                        len(shared)
                        /
                        shorter
                    )

                else:

                    overlap = 0.0

                similarity.append(
                    {

                        "route_a":
                        ROUTE_DEFINITIONS[
                            key_a
                        ]["name"],

                        "route_b":
                        ROUTE_DEFINITIONS[
                            key_b
                        ]["name"],

                        "shared_cells":
                        int(
                            len(shared)
                        ),

                        "jaccard_similarity":
                        float(
                            jaccard
                        ),

                        "overlap_shorter_route":
                        float(
                            overlap
                        ),

                        "route_difference":
                        float(
                            1.0 - overlap
                        )
                    }
                )

        return similarity


    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def run_route_analysis(
        self,
        start_lat,
        start_lon,
        end_lat,
        end_lon
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "DYNAMIC MULTI-ROUTE ANALYSIS V2"
        )

        print(
            "=" * 70
        )

        print(
            "\nStart:"
        )

        print(
            f"Latitude:  {start_lat}"
        )

        print(
            f"Longitude: {start_lon}"
        )

        print(
            "\nDestination:"
        )

        print(
            f"Latitude:  {end_lat}"
        )

        print(
            f"Longitude: {end_lon}"
        )

        # ----------------------------------------------------
        # Prepare start
        # ----------------------------------------------------

        start_info = self.prepare_location(
            start_lat,
            start_lon,
            "Start"
        )

        # ----------------------------------------------------
        # Prepare destination
        # ----------------------------------------------------

        end_info = self.prepare_location(
            end_lat,
            end_lon,
            "Destination"
        )

        start = (
            start_info["row"],
            start_info["col"]
        )

        goal = (
            end_info["row"],
            end_info["col"]
        )

        print(
            "\nRouting coordinates:"
        )

        print(
            "Start pixel:",
            start
        )

        print(
            "Goal pixel:",
            goal
        )

        # ----------------------------------------------------
        # Direct distance
        # ----------------------------------------------------

        _, _, direct_distance_m = GEOD.inv(
            start_lon,
            start_lat,
            end_lon,
            end_lat
        )

        direct_distance_m = abs(
            direct_distance_m
        )

        # ----------------------------------------------------
        # Generate routes
        # ----------------------------------------------------

        route_results = {}

        for route_key in ROUTE_DEFINITIONS:

            route_results[
                route_key
            ] = self._run_single_route(
                route_key,
                start,
                goal
            )

        # ----------------------------------------------------
        # Compare
        # ----------------------------------------------------

        comparison = self.compare_routes(
            route_results,
            direct_distance_m
        )

        # ----------------------------------------------------
        # Similarity
        # ----------------------------------------------------

        similarity = (
            self.calculate_similarity(
                route_results
            )
        )

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result = {

            "status":
            "success",

            "engine":
            "Routing Engine V2",

            "routing_algorithm":
            "V6 A*",

            "start":
            start_info,

            "destination":
            end_info,

            "direct_distance_m":
            float(
                direct_distance_m
            ),

            "direct_distance_km":
            float(
                direct_distance_m / 1000.0
            ),

            "routes":
            route_results,

            "comparison":
            comparison,

            "similarity":
            similarity
        }

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "MULTI-ROUTE SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            "\nDirect distance:",
            round(
                direct_distance_m / 1000.0,
                3
            ),
            "km"
        )

        for row in comparison:

            print(
                f"\n{row['route_name']}"
            )

            print(
                "  Distance:",
                round(
                    row[
                        "geodesic_distance_km"
                    ],
                    3
                ),
                "km"
            )

            print(
                "  Economic cost:",
                round(
                    row[
                        "economic_cost"
                    ],
                    3
                )
            )

            print(
                "  Environmental impact:",
                round(
                    row[
                        "environmental_impact_index"
                    ],
                    3
                )
            )

            print(
                "  Straightness:",
                round(
                    row[
                        "straightness"
                    ],
                    4
                )
            )

        print(
            "\n"
            + "=" * 70
        )

        return result


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_safe(
    obj
):

    if isinstance(
        obj,
        np.integer
    ):

        return int(obj)

    if isinstance(
        obj,
        np.floating
    ):

        return float(obj)

    if isinstance(
        obj,
        np.ndarray
    ):

        return obj.tolist()

    if isinstance(
        obj,
        tuple
    ):

        return list(obj)

    if isinstance(
        obj,
        dict
    ):

        return {
            key:
            make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        list
    ):

        return [
            make_json_safe(value)
            for value in obj
        ]

    return obj


# ============================================================
# SAVE JSON
# ============================================================

def save_result(
    result,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    safe = make_json_safe(
        result
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            safe,
            f,
            indent=4
        )

    print(
        "\nResult saved:"
    )

    print(
        path
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ROUTING ENGINE V2 TEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # VALIDATED PROJECT TEST COORDINATES
    # --------------------------------------------------------

    START_LAT = 27.5619

    START_LON = 96.0374

    END_LAT = 27.383898

    END_LON = 95.333229

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    engine = RoutingEngineV2()

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = engine.run_route_analysis(

        start_lat=START_LAT,

        start_lon=START_LON,

        end_lat=END_LAT,

        end_lon=END_LON
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "routing_engine_v2_test.json"
    )

    save_result(
        result,
        output_file
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ROUTING ENGINE V2 TEST COMPLETE"
    )

    print(
        "=" * 70
    )