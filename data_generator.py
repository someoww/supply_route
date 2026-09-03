import argparse
import os

import numpy as np
import pandas as pd


DIVISION_DISTRICTS = {
    "Dhaka": [
        "Dhaka", "Gazipur", "Narayanganj", "Tangail",
        "Kishoreganj", "Manikganj"
    ],
    "Chattogram": [
        "Chattogram", "Coxs Bazar", "Cumilla", "Feni"
    ],
    "Sylhet": [
        "Sylhet", "Moulvibazar", "Sunamganj", "Habiganj"
    ],
    "Rajshahi": [
        "Rajshahi", "Bogura", "Pabna", "Sirajganj"
    ],
    "Khulna": [
        "Khulna", "Jashore", "Satkhira", "Bagerhat"
    ],
    "Barishal": [
        "Barishal", "Bhola", "Patuakhali"
    ],
    "Rangpur": [
        "Rangpur", "Dinajpur", "Gaibandha"
    ],
    "Mymensingh": [
        "Mymensingh", "Jamalpur", "Netrokona"
    ],
}

# product_name, perishability, imported_input_dependency
PRODUCTS = [
    ("local_rice", 0.10, 0.15),
    ("local_vegetables", 0.85, 0.25),
    ("local_fish", 0.95, 0.35),
    ("local_poultry_meat", 0.80, 0.45),
    ("local_dairy_milk", 0.90, 0.35),
    ("local_fruits", 0.75, 0.20),
    ("local_noodles", 0.20, 0.55),
    ("local_pharma", 0.20, 0.65),
    ("local_rmg", 0.10, 0.70),
    ("local_jute_goods", 0.20, 0.30),
]


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def generate(
    n_rows: int = 100_000,
    out_path: str = "data/synthetic_supply_routes_bd.csv",
    seed: int = 42,
) -> None:
    n_rows = int(n_rows)
    rng = np.random.default_rng(seed)
    fixed_rng = np.random.default_rng(seed + 123)

    all_districts = []
    all_divisions = []
    for division, districts in DIVISION_DISTRICTS.items():
        for district in districts:
            all_districts.append(district)
            all_divisions.append(division)

    division_names = list(DIVISION_DISTRICTS.keys())
    division_map = {name: i for i, name in enumerate(division_names)}
    division_codes_for_district = np.array(
        [division_map[division] for division in all_divisions],
        dtype=int
    )
    district_codes = np.arange(len(all_districts), dtype=int)

    # Fixed district-level demographic/geographic attributes
    district_pop_density_base = fixed_rng.uniform(400, 3200, len(all_districts))
    district_urban_base = fixed_rng.uniform(18, 85, len(all_districts))
    district_income_base = fixed_rng.uniform(0.25, 0.85, len(all_districts))
    district_elevation_base = fixed_rng.uniform(2, 70, len(all_districts))

    for i, name in enumerate(all_districts):
        if name == "Dhaka":
            district_pop_density_base[i] = 4500
            district_urban_base[i] = 98
            district_income_base[i] = 0.90
            district_elevation_base[i] = 8
        elif name == "Gazipur":
            district_pop_density_base[i] = 3200
            district_urban_base[i] = 85
            district_income_base[i] = 0.78
        elif name == "Chattogram":
            district_pop_density_base[i] = 2800
            district_urban_base[i] = 90
            district_income_base[i] = 0.82
        elif name == "Sylhet":
            district_elevation_base[i] = 45
        elif name == "Coxs Bazar":
            district_elevation_base[i] = 35
            district_urban_base[i] = 55

    product_perishabilities = np.array([p[1] for p in PRODUCTS], dtype=float)
    product_input_dependencies = np.array([p[2] for p in PRODUCTS], dtype=float)

    # Row identifiers
    district_idx = rng.integers(0, len(all_districts), n_rows)
    division_code = division_codes_for_district[district_idx]
    district_code = district_codes[district_idx]
    route_id = district_code * 1000 + rng.integers(1, 60, n_rows)

    product_idx = rng.integers(0, len(PRODUCTS), n_rows)
    product_code = product_idx
    product_perishability = product_perishabilities[product_idx]
    product_input_dependency = product_input_dependencies[product_idx]

    # Time features
    timestamps = pd.date_range(
        end=pd.Timestamp("2026-06-15 23:00:00"),
        periods=n_rows,
        freq="h"
    )
    hour = timestamps.hour.to_numpy()
    month = timestamps.month.to_numpy()
    dayofweek = timestamps.dayofweek.to_numpy()
    doy = timestamps.dayofyear.to_numpy()
    is_night = ((hour >= 22) | (hour <= 5)).astype(int)

    # International / macro shocks affecting domestic production inputs
    wave = np.sin(2 * np.pi * np.arange(n_rows) / 8760)
    war_risk_index = _clip01(rng.beta(1.8, 6.0, n_rows) + 0.12 * wave)

    global_fuel_price_usd = np.clip(
        78 + 14 * wave + rng.normal(0, 3, n_rows) + 18 * war_risk_index,
        55,
        140
    )

    shipping_lane_disruption_index = _clip01(
        0.25 * war_risk_index + rng.beta(1.4, 7.0, n_rows)
    )

    forex_reserve_pressure_index = _clip01(
        rng.beta(2.0, 5.0, n_rows) + 0.22 * shipping_lane_disruption_index
    )

    global_fertilizer_price_usd = np.clip(
        340
        + 95 * war_risk_index
        + 80 * shipping_lane_disruption_index
        + rng.normal(0, 18, n_rows),
        220,
        900
    )

    # Natural disaster cluster
    monsoon = np.isin(month, [6, 7, 8, 9]).astype(float)
    pre_monsoon = np.isin(month, [3, 4, 5]).astype(float)
    cyclone_season = np.isin(month, [4, 5, 10, 11]).astype(float)

    rainfall_mm = np.clip(
        rng.gamma(1.4, 3.4, n_rows)
        * (1 + 2.6 * monsoon + 0.7 * pre_monsoon),
        0,
        350
    )

    river_water_level_m = np.clip(
        2.5 + 2.8 * monsoon + 0.04 * rainfall_mm + rng.normal(0, 0.18, n_rows),
        0.5,
        9.5
    )

    flood_index = _clip01(
        0.45 * monsoon
        + 0.015 * rainfall_mm
        + 0.12 * np.clip(river_water_level_m - 4.0, 0, 4)
    )

    cyclone_distance_km = np.clip(
        rng.exponential(220, n_rows) - 120 * cyclone_season,
        1,
        2000
    )

    cyclone_near = _clip01(1 - cyclone_distance_km / 600)

    temperature_c = (
        23
        + 7 * np.sin(2 * np.pi * (doy - 100) / 365)
        + rng.normal(0, 1.4, n_rows)
    )

    humidity_pct = np.clip(
        55 + 20 * monsoon + 0.08 * rainfall_mm + rng.normal(0, 7, n_rows),
        30,
        100
    )

    elevation_m = np.clip(
        district_elevation_base[district_idx] + rng.normal(0, 1, n_rows),
        1,
        200
    )

    landslide_risk = _clip01(
        0.45 * np.clip(elevation_m / 80, 0, 1) * np.clip(rainfall_mm / 120, 0, 1)
        + 0.15 * cyclone_near
    )

    soil_saturation = _clip01(
        0.35 * monsoon
        + 0.012 * rainfall_mm
        + rng.normal(0, 0.05, n_rows)
    )

    disaster_alert_level = np.digitize(
        _clip01(
            0.55 * flood_index
            + 0.35 * cyclone_near
            + 0.10 * landslide_risk
        ),
        [0.25, 0.55, 0.80]
    ).astype(int)

    # Festival / seasonal demand
    festival_months = np.isin(month, [4, 5, 10, 11, 12])
    festival_demand_multiplier = np.clip(
        1 + 0.45 * festival_months + rng.normal(0, 0.05, n_rows),
        0.8,
        2.2
    )

    # Man-made disruption cluster
    strike_event = rng.random(n_rows) < 0.04

    political_unrest_index = _clip01(
        rng.beta(1.7, 5.5, n_rows)
        + 0.45 * strike_event
        + 0.08 * np.sin(2 * np.pi * doy / 365)
    )

    transport_strike_flag = (
        (rng.random(n_rows) < 0.025) | (political_unrest_index > 0.82)
    ).astype(int)

    curfew_flag = (political_unrest_index > 0.93).astype(int)

    extortion_risk_index = _clip01(
        rng.beta(2.0, 6.0, n_rows)
        + 0.25 * political_unrest_index
        + 0.12 * transport_strike_flag
    )

    power_outage_hours = np.clip(
        rng.gamma(1.3, 1.2, n_rows)
        * (
            1
            + 0.6 * forex_reserve_pressure_index
            + 0.25 * (global_fuel_price_usd - 75) / 25
        ),
        0,
        12
    )

    gas_supply_index = _clip01(
        0.75
        - 0.20 * forex_reserve_pressure_index
        - 0.08 * (power_outage_hours / 12)
        + rng.normal(0, 0.05, n_rows)
    )

    labor_shortage_index = _clip01(
        rng.beta(2.0, 6.0, n_rows)
        + 0.18 * (festival_demand_multiplier - 1)
        + 0.12 * political_unrest_index
    )

    # Communication / digital cluster
    mobile_network_latency_ms = np.clip(
        rng.lognormal(3.6, 0.45, n_rows)
        * (1 + 0.8 * political_unrest_index + 0.7 * curfew_flag),
        20,
        1200
    )

    internet_outage_flag = (
        (rng.random(n_rows) < 0.02) | (political_unrest_index > 0.90)
    ).astype(int)

    radio_coverage_score = _clip01(
        0.8 - 0.2 * internet_outage_flag + rng.normal(0, 0.05, n_rows)
    )

    logistics_app_adoption_pct = np.clip(
        district_urban_base[district_idx] * 0.8 + rng.normal(0, 5, n_rows),
        0,
        100
    )

    digital_payment_success_rate = _clip01(
        0.85
        - 0.25 * internet_outage_flag
        - 0.0001 * mobile_network_latency_ms
        + rng.normal(0, 0.04, n_rows)
    )

    distress_call_volume = np.clip(
        rng.lognormal(2.5, 0.7, n_rows)
        * (
            1
            + 2.0 * flood_index
            + 1.2 * cyclone_near
            + 0.8 * political_unrest_index
        ),
        0,
        1200
    )

    # Demographic / consumption cluster
    population_density_per_km2 = district_pop_density_base[district_idx]
    urban_pct = district_urban_base[district_idx]
    income_index = district_income_base[district_idx]

    historical_daily_demand = np.clip(
        population_density_per_km2
        * 0.12
        * (1 + product_perishability)
        * festival_demand_multiplier
        + rng.normal(0, 18, n_rows),
        5,
        6000
    )

    panic_buying_index = _clip01(
        0.35 * (disaster_alert_level / 3)
        + 0.25 * political_unrest_index
        + 0.25 * np.clip(distress_call_volume / 300, 0, 1)
        + rng.normal(0, 0.05, n_rows)
    )

    demand_volatility = _clip01(
        0.25
        + 0.35 * panic_buying_index
        + 0.18 * (festival_demand_multiplier - 1)
        + rng.normal(0, 0.05, n_rows)
    )

    substitute_availability_score = _clip01(
        0.70
        - 0.25 * flood_index
        - 0.15 * political_unrest_index
        + rng.normal(0, 0.06, n_rows)
    )

    hoarding_syndicate_index = _clip01(
        0.22
        + 0.30 * extortion_risk_index
        + 0.25 * panic_buying_index
        + 0.12 * political_unrest_index
        + rng.normal(0, 0.05, n_rows)
    )

    # Transport / physical route cluster
    route_type_code = rng.integers(0, 3, n_rows)
    route_distance_km = np.clip(rng.lognormal(3.2, 0.55, n_rows), 5, 600)

    bridge_congestion_index = _clip01(
        rng.beta(2.0, 5.0, n_rows)
        + 0.20 * transport_strike_flag
        + 0.18 * (festival_demand_multiplier - 1)
        + 0.12 * flood_index
    )

    road_condition_score = np.clip(
        85
        - 45 * flood_index
        - 15 * political_unrest_index
        - 8 * (power_outage_hours / 12)
        + rng.normal(0, 4, n_rows),
        0,
        100
    )

    base_speed = np.full(n_rows, 32.0)
    base_speed[route_type_code == 0] = 48.0
    base_speed[route_type_code == 1] = 18.0

    highway_traffic_speed_kmh = np.clip(
        base_speed
        * (
            1
            - 0.45 * flood_index
            - 0.25 * transport_strike_flag
            - 0.12 * bridge_congestion_index
            - 0.08 * political_unrest_index
        )
        + rng.normal(0, 3, n_rows),
        5,
        80
    )

    ferry_base = rng.gamma(1.6, 0.9, n_rows)
    ferry_wait_hours = np.where(
        route_type_code > 0,
        np.clip(
            ferry_base * (1 + 1.7 * flood_index + 0.8 * bridge_congestion_index),
            0,
            48
        ),
        np.clip(ferry_base * 0.05, 0, 2)
    )

    fuel_price_bdt_per_litre = np.clip(
        95
        + 18 * (global_fuel_price_usd - 75) / 15
        + 12 * forex_reserve_pressure_index
        + rng.normal(0, 2, n_rows),
        80,
        180
    )

    truck_availability_index = _clip01(
        0.80
        - 0.35 * transport_strike_flag
        - 0.18 * (fuel_price_bdt_per_litre - 95) / 50
        + rng.normal(0, 0.06, n_rows)
    )

    checkpost_delay_hours = np.clip(
        rng.gamma(1.6, 0.5, n_rows)
        * (1 + 1.2 * extortion_risk_index + 0.8 * transport_strike_flag),
        0,
        24
    )

    cold_chain_availability = _clip01(
        0.65
        + 0.002 * logistics_app_adoption_pct
        - 0.25 * (power_outage_hours / 12)
        - 0.12 * flood_index
        + rng.normal(0, 0.05, n_rows)
    )

    travel_time_hours = np.clip(
        route_distance_km / highway_traffic_speed_kmh + ferry_wait_hours,
        0.25,
        168
    )

    inland_waterway_active = (
        (route_type_code > 0)
        & (flood_index < 0.85)
        & (cyclone_near < 0.8)
    ).astype(int)

    # Wastage cluster
    packaging_quality_index = _clip01(
        0.58
        + 0.0015 * logistics_app_adoption_pct
        + rng.normal(0, 0.06, n_rows)
    )

    handling_loss_pct = np.clip(
        2
        + 8 * product_perishability * (1 - cold_chain_availability)
        + 4 * flood_index
        + rng.normal(0, 0.8, n_rows),
        0,
        30
    )

    spoilage_rate_pct = np.clip(
        product_perishability
        * (
            2
            + 15 * (1 - cold_chain_availability)
            + 10 * (travel_time_hours / 48)
            + 6 * (power_outage_hours / 12)
        )
        + rng.normal(0, 0.7, n_rows),
        0,
        50
    )

    warehouse_capacity_utilization = _clip01(
        0.60
        + 0.22 * (festival_demand_multiplier - 1)
        + 0.15 * flood_index
        + rng.normal(0, 0.06, n_rows)
    )

    cold_storage_capacity_pct = _clip01(
        0.55
        + 0.0015 * logistics_app_adoption_pct
        - 0.20 * (power_outage_hours / 12)
        + rng.normal(0, 0.05, n_rows)
    )

    # Government / policy cluster
    policy_support_index = _clip01(
        0.50
        + 0.20 * (disaster_alert_level / 3)
        - 0.12 * forex_reserve_pressure_index
        + rng.normal(0, 0.06, n_rows)
    )

    import_lc_delay_index = _clip01(
        0.30
        + 0.35 * forex_reserve_pressure_index
        + 0.20 * shipping_lane_disruption_index
        + rng.normal(0, 0.06, n_rows)
    )

    subsidy_active_flag = (
        (rng.random(n_rows) < 0.12) | (disaster_alert_level >= 2)
    ).astype(int)

    tcb_intervention_flag = (
        (disaster_alert_level >= 2)
        | (panic_buying_index > 0.70)
        | (rng.random(n_rows) < 0.03)
    ).astype(int)

    ngo_relief_flag = (
        (disaster_alert_level >= 1) & (rng.random(n_rows) < 0.4)
    ).astype(int)

    customs_delay_days = np.clip(
        rng.gamma(1.5, 1.2, n_rows) * (1 + 1.5 * import_lc_delay_index),
        0,
        30
    )

    export_restriction_flag = (
        (hoarding_syndicate_index > 0.75) | (rng.random(n_rows) < 0.04)
    ).astype(int)

    # Contracts / business risk cluster
    contract_penalty_index = _clip01(
        rng.beta(2.0, 6.0, n_rows) + 0.12 * (travel_time_hours / 48)
    )

    buyer_concentration_index = _clip01(rng.beta(2.0, 5.0, n_rows))

    payment_delay_days = np.clip(
        rng.gamma(1.7, 4.0, n_rows)
        * (1 + 0.5 * forex_reserve_pressure_index),
        0,
        90
    )

    credit_access_index = _clip01(
        0.55
        + 0.002 * logistics_app_adoption_pct
        - 0.20 * forex_reserve_pressure_index
        + rng.normal(0, 0.06, n_rows)
    )

    insurance_coverage_pct = np.clip(
        30
        + 25 * credit_access_index
        + 10 * policy_support_index
        + rng.normal(0, 4, n_rows),
        0,
        100
    )

    # Engineered variables
    route_submersion_probability = _clip01(
        0.50 * flood_index
        + 0.20 * cyclone_near
        + 0.15 * soil_saturation
        + 0.15 * np.clip(river_water_level_m / 8, 0, 1)
    )

    physical_digital_friction_index = _clip01(
        0.35 * transport_strike_flag
        + 0.25 * flood_index
        + 0.20 * internet_outage_flag
        + 0.20 * np.clip(mobile_network_latency_ms / 800, 0, 1)
    )

    input_shock_transmission_score = _clip01(
        product_input_dependency
        * (
            0.45 * np.clip((global_fuel_price_usd - 70) / 50, 0, 1)
            + 0.30 * shipping_lane_disruption_index
            + 0.25 * import_lc_delay_index
        )
    )

    syndicate_hoard_prob = _clip01(
        0.55 * hoarding_syndicate_index
        + 0.25 * extortion_risk_index
        + 0.20 * panic_buying_index
    )

    post_disaster_recovery_lag = np.clip(
        12 * flood_index
        + 8 * cyclone_near
        + 4 * physical_digital_friction_index
        + rng.normal(0, 2, n_rows),
        0,
        72
    )

    demand_shock_index = _clip01(
        0.45 * panic_buying_index
        + 0.35 * (festival_demand_multiplier - 1)
        + 0.20 * demand_volatility
    )

    logistics_cost_index = _clip01(
        0.35 * np.clip((fuel_price_bdt_per_litre - 90) / 60, 0, 1)
        + 0.25 * np.clip(checkpost_delay_hours / 24, 0, 1)
        + 0.20 * np.clip(ferry_wait_hours / 24, 0, 1)
        + 0.20 * transport_strike_flag
    )

    route_vulnerability_score = _clip01(
        0.30 * route_submersion_probability
        + 0.25 * physical_digital_friction_index
        + 0.20 * syndicate_hoard_prob
        + 0.15 * input_shock_transmission_score
        + 0.10 * labor_shortage_index
    )

    # Targets
    delay_hours = np.clip(
        0.15 * travel_time_hours
        + 0.70 * checkpost_delay_hours
        + 0.50 * ferry_wait_hours
        + 18 * route_submersion_probability
        + 10 * transport_strike_flag
        + 6 * syndicate_hoard_prob
        + 4 * (power_outage_hours / 12)
        + rng.normal(0, 1.2, n_rows),
        0,
        120
    )

    supply_route_condition_score = np.clip(
        100
        - 2.2 * delay_hours
        - 15 * route_vulnerability_score
        - 0.3 * spoilage_rate_pct
        + 5 * policy_support_index
        + rng.normal(0, 2, n_rows),
        0,
        100
    )

    condition_class = np.select(
        [
            supply_route_condition_score >= 75,
            supply_route_condition_score >= 50
        ],
        [0, 1],
        default=2
    ).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "division_code": division_code,
            "district_code": district_code,
            "route_id": route_id,
            "product_code": product_code,
            "product_perishability": product_perishability,
            "product_input_dependency": product_input_dependency,
            "hour": hour,
            "month": month,
            "dayofweek": dayofweek,
            "is_night": is_night,
            "war_risk_index": war_risk_index,
            "global_fuel_price_usd": global_fuel_price_usd,
            "shipping_lane_disruption_index": shipping_lane_disruption_index,
            "forex_reserve_pressure_index": forex_reserve_pressure_index,
            "global_fertilizer_price_usd": global_fertilizer_price_usd,
            "rainfall_mm": rainfall_mm,
            "river_water_level_m": river_water_level_m,
            "flood_index": flood_index,
            "cyclone_distance_km": cyclone_distance_km,
            "cyclone_near": cyclone_near,
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "elevation_m": elevation_m,
            "landslide_risk": landslide_risk,
            "soil_saturation": soil_saturation,
            "disaster_alert_level": disaster_alert_level,
            "festival_demand_multiplier": festival_demand_multiplier,
            "political_unrest_index": political_unrest_index,
            "transport_strike_flag": transport_strike_flag,
            "curfew_flag": curfew_flag,
            "extortion_risk_index": extortion_risk_index,
            "power_outage_hours": power_outage_hours,
            "gas_supply_index": gas_supply_index,
            "labor_shortage_index": labor_shortage_index,
            "mobile_network_latency_ms": mobile_network_latency_ms,
            "internet_outage_flag": internet_outage_flag,
            "radio_coverage_score": radio_coverage_score,
            "logistics_app_adoption_pct": logistics_app_adoption_pct,
            "digital_payment_success_rate": digital_payment_success_rate,
            "distress_call_volume": distress_call_volume,
            "population_density_per_km2": population_density_per_km2,
            "urban_pct": urban_pct,
            "income_index": income_index,
            "historical_daily_demand": historical_daily_demand,
            "panic_buying_index": panic_buying_index,
            "demand_volatility": demand_volatility,
            "substitute_availability_score": substitute_availability_score,
            "hoarding_syndicate_index": hoarding_syndicate_index,
            "route_type_code": route_type_code,
            "route_distance_km": route_distance_km,
            "bridge_congestion_index": bridge_congestion_index,
            "road_condition_score": road_condition_score,
            "highway_traffic_speed_kmh": highway_traffic_speed_kmh,
            "ferry_wait_hours": ferry_wait_hours,
            "fuel_price_bdt_per_litre": fuel_price_bdt_per_litre,
            "truck_availability_index": truck_availability_index,
            "checkpost_delay_hours": checkpost_delay_hours,
            "cold_chain_availability": cold_chain_availability,
            "travel_time_hours": travel_time_hours,
            "inland_waterway_active": inland_waterway_active,
            "packaging_quality_index": packaging_quality_index,
            "handling_loss_pct": handling_loss_pct,
            "spoilage_rate_pct": spoilage_rate_pct,
            "warehouse_capacity_utilization": warehouse_capacity_utilization,
            "cold_storage_capacity_pct": cold_storage_capacity_pct,
            "policy_support_index": policy_support_index,
            "import_lc_delay_index": import_lc_delay_index,
            "subsidy_active_flag": subsidy_active_flag,
            "tcb_intervention_flag": tcb_intervention_flag,
            "ngo_relief_flag": ngo_relief_flag,
            "customs_delay_days": customs_delay_days,
            "export_restriction_flag": export_restriction_flag,
            "contract_penalty_index": contract_penalty_index,
            "buyer_concentration_index": buyer_concentration_index,
            "payment_delay_days": payment_delay_days,
            "credit_access_index": credit_access_index,
            "insurance_coverage_pct": insurance_coverage_pct,
            "route_submersion_probability": route_submersion_probability,
            "physical_digital_friction_index": physical_digital_friction_index,
            "input_shock_transmission_score": input_shock_transmission_score,
            "syndicate_hoard_prob": syndicate_hoard_prob,
            "post_disaster_recovery_lag": post_disaster_recovery_lag,
            "demand_shock_index": demand_shock_index,
            "logistics_cost_index": logistics_cost_index,
            "route_vulnerability_score": route_vulnerability_score,
            "delay_hours": delay_hours,
            "supply_route_condition_score": supply_route_condition_score,
            "condition_class": condition_class,
        }
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Bangladesh supply-route dataset."
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument(
        "--out",
        type=str,
        default="data/synthetic_supply_routes_bd.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(n_rows=args.rows, out_path=args.out, seed=args.seed)