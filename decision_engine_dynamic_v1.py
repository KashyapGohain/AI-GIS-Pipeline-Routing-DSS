# ============================================================
# DECISION ENGINE DYNAMIC V1
# ============================================================
# AI + GIS Pipeline Routing Decision Support System
#
# PURPOSE
# ------------------------------------------------------------
# Takes dynamically generated route + risk results and:
#
#   1. Normalizes route performance
#   2. Calculates DSS scores
#   3. Supports multiple decision profiles
#   4. Identifies the best route for each profile
#   5. Generates an overall recommendation
#   6. Produces a human-readable recommendation
#   7. Saves JSON + CSV outputs
#
# INPUT
# ------------------------------------------------------------
# routing_engine_v2.py
# risk_engine_dynamic_v1.py
#
# OUTPUT
# ------------------------------------------------------------
# decision_engine_dynamic_v1.json
# decision_engine_dynamic_v1_summary.csv
# decision_engine_dynamic_v1_profiles.csv
# decision_engine_dynamic_v1_recommendation.txt
#
# ============================================================


import os
import sys
import json

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"I:/Kashyap/Route"

CODES_DIR = os.path.join(
    BASE_DIR,
    "codes"
)

DASHBOARD_DIR = os.path.join(
    CODES_DIR,
    "dashboard"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "OUTPUT",
    "dashboard"
)


# ============================================================
# DECISION PROFILES
# ============================================================
#
# Higher weight = greater importance.
#
# All weights sum to 1.0.
#
# ============================================================

DECISION_PROFILES = {

    "Cost Priority": {

        "economic_cost": 0.50,

        "environmental_impact": 0.20,

        "risk": 0.20,

        "distance": 0.10
    },

    "Environmental Priority": {

        "economic_cost": 0.10,

        "environmental_impact": 0.50,

        "risk": 0.30,

        "distance": 0.10
    },

    "Balanced": {

        "economic_cost": 0.25,

        "environmental_impact": 0.25,

        "risk": 0.30,

        "distance": 0.20
    },

    "Shortest Feasible": {

        "economic_cost": 0.10,

        "environmental_impact": 0.10,

        "risk": 0.10,

        "distance": 0.70
    }
}


# ============================================================
# ROUTE DECISION ENGINE
# ============================================================

class DynamicDecisionEngine:

    def __init__(
        self,
        profiles=None
    ):

        print("=" * 70)

        print(
            "INITIALIZING DYNAMIC DECISION ENGINE V1"
        )

        print("=" * 70)

        if profiles is None:

            profiles = DECISION_PROFILES

        self.profiles = profiles

        # ----------------------------------------------------
        # Validate profiles
        # ----------------------------------------------------

        self.validate_profiles()

        print(
            "\nDecision profiles:"
        )

        for name, weights in self.profiles.items():

            print(
                f"  {name}:"
            )

            print(
                f"    Economic Cost       = "
                f"{weights['economic_cost']:.2f}"
            )

            print(
                f"    Environmental Impact = "
                f"{weights['environmental_impact']:.2f}"
            )

            print(
                f"    Risk                = "
                f"{weights['risk']:.2f}"
            )

            print(
                f"    Distance            = "
                f"{weights['distance']:.2f}"
            )

        print("=" * 70)


    # ========================================================
    # VALIDATE PROFILES
    # ========================================================

    def validate_profiles(
        self
    ):

        required = {

            "economic_cost",

            "environmental_impact",

            "risk",

            "distance"
        }

        for profile_name, weights in self.profiles.items():

            if set(
                weights.keys()
            ) != required:

                raise ValueError(
                    f"\nInvalid profile: "
                    f"{profile_name}\n"
                    f"Required weights: {required}"
                )

            total = sum(
                weights.values()
            )

            if not np.isclose(
                total,
                1.0
            ):

                raise ValueError(
                    f"\nWeights for "
                    f"'{profile_name}' "
                    f"must sum to 1.0.\n"
                    f"Current total: {total}"
                )

            for key, value in weights.items():

                if value < 0:

                    raise ValueError(
                        f"Negative weight in "
                        f"{profile_name}: {key}"
                    )


    # ========================================================
    # MIN-MAX NORMALIZATION
    # ========================================================
    #
    # For all performance indicators:
    #
    # Lower value = better.
    #
    # Normalized performance:
    #
    #       max(x) - x
    # -------------------------
    #       max(x) - min(x)
    #
    # Therefore:
    #
    #   1.0 = best
    #   0.0 = worst
    #
    # ========================================================

    def normalize_lower_is_better(
        self,
        values
    ):

        values = np.asarray(
            values,
            dtype=float
        )

        if len(values) == 0:

            return values

        min_value = np.nanmin(
            values
        )

        max_value = np.nanmax(
            values
        )

        difference = (
            max_value
            -
            min_value
        )

        if np.isclose(
            difference,
            0.0
        ):

            # ------------------------------------------------
            # If all routes are identical for this metric,
            # give every route full performance.
            # ------------------------------------------------

            return np.ones_like(
                values,
                dtype=float
            )

        return (
            max_value
            -
            values
        ) / difference


    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_dataframe(
        self,
        routing_result,
        risk_results
    ):

        rows = []

        routes = routing_result.get(
            "routes",
            {}
        )

        for route_key, route_result in routes.items():

            if route_result.get(
                "status"
            ) != "success":

                continue

            risk_result = risk_results.get(
                route_key
            )

            if risk_result is None:

                continue

            if risk_result.get(
                "status"
            ) != "success":

                continue

            row = {

                "route_key":
                route_key,

                "route_id":
                route_result[
                    "route_id"
                ],

                "route_name":
                route_result[
                    "route_name"
                ],

                # ------------------------------------------------
                # Physical distance
                # ------------------------------------------------

                "distance_km":
                route_result[
                    "geodesic_distance_km"
                ],

                # ------------------------------------------------
                # Base/economic cost
                # ------------------------------------------------

                "economic_cost":
                route_result[
                    "economic_cost"
                ],

                # ------------------------------------------------
                # Environmental index
                # ------------------------------------------------

                "environmental_impact":
                route_result[
                    "environmental_impact_index"
                ],

                # ------------------------------------------------
                # Risk
                # ------------------------------------------------

                "combined_risk":
                risk_result[
                    "combined_risk"
                ],

                "risk_category":
                risk_result[
                    "risk_category"
                ],

                # ------------------------------------------------
                # Additional risk metrics
                # ------------------------------------------------

                "terrain_risk":
                risk_result[
                    "terrain_risk"
                ],

                "lulc_risk":
                risk_result[
                    "lulc_risk"
                ],

                "mean_slope_deg":
                risk_result[
                    "mean_slope_deg"
                ],

                "max_slope_deg":
                risk_result[
                    "max_slope_deg"
                ],

                "high_risk_cells":
                risk_result[
                    "high_risk_cells"
                ],

                "high_risk_percentage":
                risk_result[
                    "high_risk_percentage"
                ],

                "high_risk_distance_km":
                risk_result[
                    "high_risk_distance_km"
                ],

                "high_risk_sections":
                risk_result[
                    "high_risk_sections"
                ]
            }

            rows.append(
                row
            )

        if len(rows) == 0:

            raise RuntimeError(
                "\nNo valid routes available "
                "for decision analysis."
            )

        df = pd.DataFrame(
            rows
        )

        df = df.sort_values(
            "route_id"
        ).reset_index(
            drop=True
        )

        return df


    # ========================================================
    # NORMALIZE PERFORMANCE
    # ========================================================

    def calculate_performance(
        self,
        df
    ):

        df = df.copy()

        # ----------------------------------------------------
        # Economic cost
        # ----------------------------------------------------

        df[
            "economic_performance"
        ] = self.normalize_lower_is_better(
            df[
                "economic_cost"
            ].values
        )

        # ----------------------------------------------------
        # Environmental impact
        # ----------------------------------------------------

        df[
            "environmental_performance"
        ] = self.normalize_lower_is_better(
            df[
                "environmental_impact"
            ].values
        )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        df[
            "risk_performance"
        ] = self.normalize_lower_is_better(
            df[
                "combined_risk"
            ].values
        )

        # ----------------------------------------------------
        # Physical distance
        # ----------------------------------------------------

        df[
            "distance_performance"
        ] = self.normalize_lower_is_better(
            df[
                "distance_km"
            ].values
        )

        return df


    # ========================================================
    # CALCULATE PROFILE SCORES
    # ========================================================

    def calculate_profile_scores(
        self,
        df
    ):

        profile_tables = {}

        for profile_name, weights in self.profiles.items():

            scores = (

                weights[
                    "economic_cost"
                ]
                *
                df[
                    "economic_performance"
                ]

                +

                weights[
                    "environmental_impact"
                ]
                *
                df[
                    "environmental_performance"
                ]

                +

                weights[
                    "risk"
                ]
                *
                df[
                    "risk_performance"
                ]

                +

                weights[
                    "distance"
                ]
                *
                df[
                    "distance_performance"
                ]
            )

            table = df[
                [
                    "route_key",
                    "route_id",
                    "route_name"
                ]
            ].copy()

            table[
                "profile"
            ] = profile_name

            table[
                "economic_weight"
            ] = weights[
                "economic_cost"
            ]

            table[
                "environmental_weight"
            ] = weights[
                "environmental_impact"
            ]

            table[
                "risk_weight"
            ] = weights[
                "risk"
            ]

            table[
                "distance_weight"
            ] = weights[
                "distance"
            ]

            table[
                "dss_score"
            ] = scores

            table = table.sort_values(
                "dss_score",
                ascending=False
            ).reset_index(
                drop=True
            )

            table[
                "rank"
            ] = np.arange(
                1,
                len(table) + 1
            )

            profile_tables[
                profile_name
            ] = table

        return profile_tables


    # ========================================================
    # OVERALL SCORE
    # ========================================================
    #
    # Equal importance across the four decision profiles.
    #
    # ========================================================

    def calculate_overall_scores(
        self,
        profile_tables
    ):

        all_scores = {}

        for profile_name, table in profile_tables.items():

            for _, row in table.iterrows():

                route_key = row[
                    "route_key"
                ]

                if route_key not in all_scores:

                    all_scores[
                        route_key
                    ] = []

                all_scores[
                    route_key
                ].append(
                    float(
                        row[
                            "dss_score"
                        ]
                    )
                )

        overall_rows = []

        for route_key, scores in all_scores.items():

            profile_scores = np.array(
                scores,
                dtype=float
            )

            mean_score = float(
                np.mean(
                    profile_scores
                )
            )

            minimum_score = float(
                np.min(
                    profile_scores
                )
            )

            maximum_score = float(
                np.max(
                    profile_scores
                )
            )

            overall_rows.append(
                {

                    "route_key":
                    route_key,

                    "overall_dss_score":
                    mean_score,

                    "minimum_profile_score":
                    minimum_score,

                    "maximum_profile_score":
                    maximum_score
                }
            )

        overall = pd.DataFrame(
            overall_rows
        )

        overall = overall.sort_values(
            "overall_dss_score",
            ascending=False
        ).reset_index(
            drop=True
        )

        overall[
            "overall_rank"
        ] = np.arange(
            1,
            len(overall) + 1
        )

        return overall


    # ========================================================
    # PROFILE RECOMMENDATIONS
    # ========================================================

    def profile_recommendations(
        self,
        profile_tables
    ):

        rows = []

        for profile_name, table in profile_tables.items():

            if len(table) == 0:

                continue

            best = table.iloc[
                0
            ]

            rows.append(
                {

                    "profile":
                    profile_name,

                    "recommended_route_key":
                    best[
                        "route_key"
                    ],

                    "recommended_route":
                    best[
                        "route_name"
                    ],

                    "dss_score":
                    float(
                        best[
                            "dss_score"
                        ]
                    )
                }
            )

        return pd.DataFrame(
            rows
        )


    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    def generate_recommendation(
        self,
        df,
        overall,
        profile_recommendations
    ):

        if len(overall) == 0:

            raise RuntimeError(
                "No routes available."
            )

        # ----------------------------------------------------
        # Overall best
        # ----------------------------------------------------

        best = overall.iloc[
            0
        ]

        best_key = best[
            "route_key"
        ]

        route_row = df[
            df[
                "route_key"
            ]
            ==
            best_key
        ].iloc[
            0
        ]

        # ----------------------------------------------------
        # Profile vote
        # ----------------------------------------------------

        votes = {}

        for _, row in profile_recommendations.iterrows():

            route_key = row[
                "recommended_route_key"
            ]

            votes[
                route_key
            ] = votes.get(
                route_key,
                0
            ) + 1

        if len(votes) > 0:

            majority_key = max(
                votes,
                key=votes.get
            )

        else:

            majority_key = best_key

        majority_row = df[
            df[
                "route_key"
            ]
            ==
            majority_key
        ].iloc[
            0
        ]

        # ----------------------------------------------------
        # Build explanation
        # ----------------------------------------------------

        text = []

        text.append(
            "DYNAMIC ROUTE RECOMMENDATION"
        )

        text.append(
            "=" * 70
        )

        text.append(
            ""
        )

        text.append(
            f"Recommended route: "
            f"{route_row['route_name']}"
        )

        text.append(
            f"Overall DSS score: "
            f"{best['overall_dss_score']:.4f}"
        )

        text.append(
            ""
        )

        text.append(
            "Route characteristics:"
        )

        text.append(
            f"  Physical distance: "
            f"{route_row['distance_km']:.3f} km"
        )

        text.append(
            f"  Economic cost: "
            f"{route_row['economic_cost']:.3f}"
        )

        text.append(
            f"  Environmental Impact Index: "
            f"{route_row['environmental_impact']:.3f}"
        )

        text.append(
            f"  Combined risk: "
            f"{route_row['combined_risk']:.4f}"
        )

        text.append(
            f"  Risk category: "
            f"{route_row['risk_category']}"
        )

        text.append(
            ""
        )

        text.append(
            "Decision profile recommendations:"
        )

        for _, row in profile_recommendations.iterrows():

            text.append(
                f"  {row['profile']}: "
                f"{row['recommended_route']} "
                f"(score "
                f"{row['dss_score']:.4f})"
            )

        text.append(
            ""
        )

        text.append(
            f"Profile majority route: "
            f"{majority_row['route_name']}"
        )

        text.append(
            f"Profile votes: "
            f"{votes.get(majority_key, 0)} / "
            f"{len(profile_recommendations)}"
        )

        text.append(
            ""
        )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        if best_key == majority_key:

            text.append(
                "The overall DSS ranking and the "
                "majority of decision profiles agree "
                "on the recommended route."
            )

        else:

            text.append(
                "The overall DSS ranking and profile "
                "majority differ, indicating a "
                "meaningful trade-off between route "
                "objectives."
            )

        text.append(
            ""
        )

        text.append(
            "Note: Environmental Impact is reported "
            "as an index derived from the environmental "
            "cost surface, not as a physical "
            "environmental quantity."
        )

        recommendation_text = "\n".join(
            text
        )

        return {

            "recommended_route_key":
            best_key,

            "recommended_route":
            route_row[
                "route_name"
            ],

            "overall_dss_score":
            float(
                best[
                    "overall_dss_score"
                ]
            ),

            "profile_majority_route_key":
            majority_key,

            "profile_majority_route":
            majority_row[
                "route_name"
            ],

            "profile_votes":
            votes,

            "text":
            recommendation_text
        }


    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def run_decision_analysis(
        self,
        routing_result,
        risk_results
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RUNNING DYNAMIC DSS ANALYSIS"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        df = self.prepare_dataframe(
            routing_result,
            risk_results
        )

        print(
            "\nValid routes:",
            len(df)
        )

        # ----------------------------------------------------
        # Performance normalization
        # ----------------------------------------------------

        df = self.calculate_performance(
            df
        )

        # ----------------------------------------------------
        # Profile scores
        # ----------------------------------------------------

        profile_tables = (
            self.calculate_profile_scores(
                df
            )
        )

        # ----------------------------------------------------
        # Overall scores
        # ----------------------------------------------------

        overall = (
            self.calculate_overall_scores(
                profile_tables
            )
        )

        # ----------------------------------------------------
        # Profile recommendations
        # ----------------------------------------------------

        profile_recs = (
            self.profile_recommendations(
                profile_tables
            )
        )

        # ----------------------------------------------------
        # Final recommendation
        # ----------------------------------------------------

        recommendation = (
            self.generate_recommendation(
                df,
                overall,
                profile_recs
            )
        )

        # ----------------------------------------------------
        # Print route performance
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            "ROUTE PERFORMANCE"
        )

        print(
            "-" * 70
        )

        for _, row in df.iterrows():

            print(
                f"\n{row['route_name']}"
            )

            print(
                f"  Distance: "
                f"{row['distance_km']:.3f} km"
            )

            print(
                f"  Economic cost: "
                f"{row['economic_cost']:.3f}"
            )

            print(
                f"  Environmental impact: "
                f"{row['environmental_impact']:.3f}"
            )

            print(
                f"  Risk: "
                f"{row['combined_risk']:.4f}"
            )

            print(
                f"  Risk category: "
                f"{row['risk_category']}"
            )

        # ----------------------------------------------------
        # Print normalized performance
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            "NORMALIZED PERFORMANCE"
        )

        print(
            "-" * 70
        )

        for _, row in df.iterrows():

            print(
                f"\n{row['route_name']}"
            )

            print(
                f"  Economic: "
                f"{row['economic_performance']:.4f}"
            )

            print(
                f"  Environmental: "
                f"{row['environmental_performance']:.4f}"
            )

            print(
                f"  Risk: "
                f"{row['risk_performance']:.4f}"
            )

            print(
                f"  Distance: "
                f"{row['distance_performance']:.4f}"
            )

        # ----------------------------------------------------
        # Print profiles
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            "DECISION PROFILE RESULTS"
        )

        print(
            "-" * 70
        )

        for profile_name, table in profile_tables.items():

            print(
                f"\n{profile_name}"
            )

            for _, row in table.iterrows():

                print(
                    f"  {int(row['rank'])}. "
                    f"{row['route_name']} "
                    f"-> "
                    f"{row['dss_score']:.4f}"
                )

        # ----------------------------------------------------
        # Print overall
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            "OVERALL DSS RANKING"
        )

        print(
            "-" * 70
        )

        for _, row in overall.iterrows():

            route_name = df[
                df[
                    "route_key"
                ]
                ==
                row[
                    "route_key"
                ]
            ].iloc[
                0
            ][
                "route_name"
            ]

            print(
                f"{int(row['overall_rank'])}. "
                f"{route_name} "
                f"-> "
                f"{row['overall_dss_score']:.4f}"
            )

        # ----------------------------------------------------
        # Final recommendation
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL RECOMMENDATION"
        )

        print(
            "=" * 70
        )

        print(
            f"\nRecommended route:"
        )

        print(
            recommendation[
                "recommended_route"
            ]
        )

        print(
            "\nOverall DSS score:"
        )

        print(
            f"{recommendation['overall_dss_score']:.4f}"
        )

        print(
            "\nProfile majority:"
        )

        print(
            recommendation[
                "profile_majority_route"
            ]
        )

        print(
            "\n"
            + "=" * 70
        )

        return {

            "status":
            "success",

            "route_performance":
            df,

            "profile_scores":
            profile_tables,

            "overall_ranking":
            overall,

            "profile_recommendations":
            profile_recs,

            "recommendation":
            recommendation
        }


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def make_json_safe(
    obj
):

    if isinstance(
        obj,
        pd.DataFrame
    ):

        return [
            make_json_safe(
                row
            )
            for row in obj.to_dict(
                orient="records"
            )
        ]

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

def save_json(
    result,
    output_path
):

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )

    safe = make_json_safe(
        result
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            safe,
            f,
            indent=4
        )

    print(
        "\nJSON saved:"
    )

    print(
        output_path
    )


# ============================================================
# SAVE OUTPUT TABLES
# ============================================================

def save_outputs(
    result
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Route performance
    # --------------------------------------------------------

    performance_path = os.path.join(
        OUTPUT_DIR,
        "decision_engine_dynamic_v1_summary.csv"
    )

    result[
        "route_performance"
    ].to_csv(
        performance_path,
        index=False
    )

    # --------------------------------------------------------
    # Profile recommendations
    # --------------------------------------------------------

    profile_path = os.path.join(
        OUTPUT_DIR,
        "decision_engine_dynamic_v1_profiles.csv"
    )

    result[
        "profile_recommendations"
    ].to_csv(
        profile_path,
        index=False
    )

    # --------------------------------------------------------
    # Overall ranking
    # --------------------------------------------------------

    overall = result[
        "overall_ranking"
    ].copy()

    # Add route name

    performance = result[
        "route_performance"
    ]

    overall = overall.merge(

        performance[
            [
                "route_key",
                "route_name",
                "distance_km",
                "economic_cost",
                "environmental_impact",
                "combined_risk",
                "risk_category"
            ]
        ],

        on="route_key",

        how="left"
    )

    overall_path = os.path.join(
        OUTPUT_DIR,
        "decision_engine_dynamic_v1_overall.csv"
    )

    overall.to_csv(
        overall_path,
        index=False
    )

    # --------------------------------------------------------
    # Recommendation text
    # --------------------------------------------------------

    recommendation_path = os.path.join(
        OUTPUT_DIR,
        "decision_engine_dynamic_v1_recommendation.txt"
    )

    with open(
        recommendation_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            result[
                "recommendation"
            ][
                "text"
            ]
        )

    print(
        "\nOutputs saved:"
    )

    print(
        performance_path
    )

    print(
        profile_path
    )

    print(
        overall_path
    )

    print(
        recommendation_path
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
        "DYNAMIC DECISION ENGINE V1 TEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Import previous engines
    # --------------------------------------------------------

    from routing_engine_v2 import (
        RoutingEngineV2
    )

    from risk_engine_dynamic_v1 import (
        DynamicRiskEngine
    )

    # --------------------------------------------------------
    # Test coordinates
    # --------------------------------------------------------

    START_LAT = 27.5619

    START_LON = 96.0374

    END_LAT = 27.383898

    END_LON = 95.333229

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    routing_engine = (
        RoutingEngineV2()
    )

    routing_result = (
        routing_engine.run_route_analysis(

            start_lat=START_LAT,

            start_lon=START_LON,

            end_lat=END_LAT,

            end_lon=END_LON
        )
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk_engine = (
        DynamicRiskEngine()
    )

    risk_results = (
        risk_engine.analyze_routes(
            routing_result[
                "routes"
            ]
        )
    )

    # --------------------------------------------------------
    # DSS
    # --------------------------------------------------------

    decision_engine = (
        DynamicDecisionEngine()
    )

    result = (
        decision_engine.run_decision_analysis(

            routing_result,

            risk_results
        )
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "decision_engine_dynamic_v1.json"
    )

    save_json(
        result,
        json_path
    )

    # --------------------------------------------------------
    # SAVE CSV/TXT
    # --------------------------------------------------------

    save_outputs(
        result
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DYNAMIC DECISION ENGINE V1 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nRecommended Route:"
    )

    print(
        result[
            "recommendation"
        ][
            "recommended_route"
        ]
    )

    print(
        "\nOverall DSS Score:"
    )

    print(
        round(
            result[
                "recommendation"
            ][
                "overall_dss_score"
            ],
            4
        )
    )

    print(
        "\nProfile Majority Route:"
    )

    print(
        result[
            "recommendation"
        ][
            "profile_majority_route"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )