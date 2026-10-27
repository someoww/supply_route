"""Structural causal model (SCM) of Bangladesh supply routes. One row = one district-day.

Each node is declared with its parents, so the causal graph and the data-generating
equations are one object (`SCM`). Every node owns a noise stream, which makes
`simulate(do=...)` return exact counterfactuals for the very same units.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

DIVISIONS = {
    "Dhaka": ["Dhaka", "Gazipur", "Narayanganj", "Tangail", "Kishoreganj", "Manikganj"],
    "Chattogram": ["Chattogram", "Coxs Bazar", "Cumilla", "Feni"],
    "Sylhet": ["Sylhet", "Moulvibazar", "Sunamganj", "Habiganj"],
    "Rajshahi": ["Rajshahi", "Bogura", "Pabna", "Sirajganj"],
    "Khulna": ["Khulna", "Jashore", "Satkhira", "Bagerhat"],
    "Barishal": ["Barishal", "Bhola", "Patuakhali"],
    "Rangpur": ["Rangpur", "Dinajpur", "Gaibandha"],
    "Mymensingh": ["Mymensingh", "Jamalpur", "Netrokona"],
}
NAMES = [d for ds in DIVISIONS.values() for d in ds]
D = len(NAMES)

COASTAL = {"Chattogram", "Coxs Bazar", "Feni", "Khulna", "Satkhira", "Bagerhat", "Barishal", "Bhola", "Patuakhali"}
FERRY = {"Barishal", "Bhola", "Patuakhali", "Khulna", "Bagerhat", "Sirajganj", "Jamalpur", "Sunamganj", "Netrokona"}
LOWLAND = {"Sunamganj", "Habiganj", "Netrokona", "Kishoreganj", "Bhola", "Sirajganj", "Gaibandha", "Jamalpur"}

_g = np.random.default_rng(7)  # geography is a fixed fact, independent of the simulation seed
GEO = {
    "density": _g.uniform(400, 3200, D),
    "urban_pct": _g.uniform(18, 85, D),
    "elevation_m": _g.uniform(10, 60, D),
    "route_km": np.clip(_g.lognormal(5.0, 0.5, D), 40, 450),
    "coastal": np.array([n in COASTAL for n in NAMES], float),
    "ferry_dep": np.array([n in FERRY for n in NAMES], int),
}
GEO["elevation_m"][[NAMES.index(n) for n in LOWLAND]] = 3
for _n, _dens, _urb in [("Dhaka", 4500, 98), ("Gazipur", 3200, 85), ("Chattogram", 2800, 90)]:
    GEO["density"][NAMES.index(_n)], GEO["urban_pct"][NAMES.index(_n)] = _dens, _urb
GEO["route_km"][NAMES.index("Dhaka")] = 20

PRODUCTS = [  # name, perishability, imported-input dependency
    ("rice", .10, .15), ("vegetables", .85, .25), ("fish", .95, .35), ("poultry", .80, .45),
    ("milk", .90, .35), ("fruits", .75, .20), ("noodles", .20, .55), ("pharma", .20, .65),
    ("rmg", .10, .70), ("jute_goods", .20, .30),
]
PERISH, INPUT_DEP = np.array([p[1:] for p in PRODUCTS]).T
BASE_SPEED = np.array([45.0, 28.0])  # km/h on road-only vs ferry-dependent routes

# Known only once the shipment has moved, so never usable as predictors.
REALIZED = {"checkpost_h", "ferry_wait_h", "travel_h", "delay_hours", "spoilage_pct",
            "shortfall", "condition_score", "condition_class"}
IDENTIFIERS = {"timestamp", "district", "product"}  # geography and product traits already carry this


def exante(columns):
    """Columns observable at dispatch time."""
    return [c for c in columns if c not in REALIZED | IDENTIFIERS]


def sig(x):
    return 1 / (1 + np.exp(-x))


class Env:
    """Row layout (district-major) plus noise helpers drawing from one node's own stream."""

    def __init__(self, rng, days, dates):
        self.rng, self.T, self.dates = rng, days, dates
        self.n = D * days
        self.d = np.repeat(np.arange(D), days)

    def __getattr__(self, name):  # e.normal(0, 1), e.gamma(2, 1), ... -> one draw per row
        draw = getattr(self.rng, name)
        return lambda *a: draw(*a, size=self.n)

    def ar(self, phi, local=False):
        """Unit-variance AR(1) series: national by default, per district if local."""
        eps = self.rng.standard_normal((D if local else 1, self.T))
        eps[:, 0] /= np.sqrt(1 - phi**2)  # start in the stationary distribution
        x = lfilter([np.sqrt(1 - phi**2)], [1, -phi], eps, axis=1)
        return np.broadcast_to(x, (D, self.T)).ravel()


SCM = {}  # name -> (parents, mechanism)


def node(*parents, name=None):
    def register(fn):
        key = name or fn.__name__
        assert all(p in SCM for p in parents), f"{key}: declare parents first"
        SCM[key] = (parents, fn)
        return fn
    return register


# --- calendar, geography, product -------------------------------------------------
@node()
def district(e): return e.d
@node()
def month(e): return e.dates.month.to_numpy()
@node()
def product(e): return e.integers(0, len(PRODUCTS))

for _k, _v in GEO.items():
    node("district", name=_k)(lambda e, d, v=_v: v[d])
node("product", name="perishability")(lambda e, p: PERISH[p])
node("product", name="input_dependency")(lambda e, p: INPUT_DEP[p])


# --- international shocks ---------------------------------------------------------
@node()
def war_risk(e): return sig(-1.3 + e.ar(.995))

@node("war_risk")
def fuel_usd(e, war): return np.clip(76 + 25 * war + 6 * e.ar(.98), 55, 140)

@node("war_risk")
def shipping_disruption(e, war): return sig(-2.2 + 3 * war + .5 * e.ar(.95))

@node("shipping_disruption", "fuel_usd")
def forex_pressure(e, ship, fuel): return sig(-1.1 + 1.5 * ship + .03 * (fuel - 76) + .6 * e.ar(.99))

@node("fuel_usd", "forex_pressure")
def fuel_bdt(e, fuel, fx): return np.clip(95 + 1.2 * (fuel - 76) + 12 * fx + e.normal(0, 1.5), 80, 180)

@node("input_dependency", "fuel_usd", "shipping_disruption", "forex_pressure")
def input_shock(e, dep, fuel, ship, fx): return dep * (.45 * np.clip((fuel - 70) / 50, 0, 1) + .3 * ship + .25 * fx)


# --- weather ----------------------------------------------------------------------
@node("month")
def monsoon(e, m): return np.isin(m, [6, 7, 8, 9]).astype(float)

@node("month", "monsoon")
def rain_mm(e, m, mon):
    season = 1 + 2.6 * mon + .7 * np.isin(m, [3, 4, 5])
    return np.clip(e.gamma(1.4, 3.4) * season * np.exp(.5 * e.ar(.8, True)), 0, 350)

@node("monsoon", "rain_mm", "elevation_m")
def river_m(e, mon, rain, elev):
    return np.clip(2.5 + 2.8 * mon + .1 * rain - .03 * elev + .2 * e.ar(.9, True), .5, 9.5)

@node("rain_mm", "river_m")
def flood_index(e, rain, river): return sig(1.4 * (river - 6.3) + .012 * (rain - 100))

@node("coastal", "month")
def cyclone_near(e, coastal, m):
    storm = sig(3 * (e.ar(.85) - np.where(np.isin(m, [4, 5, 10, 11]), 1.7, 2.6)))
    return storm * (.2 + .8 * coastal)

@node("month")
def temp_c(e, m): return 25 + 6 * np.sin(2 * np.pi * (m - 2) / 12) + e.normal(0, 1.5)

@node("flood_index", "cyclone_near")
def alert_level(e, flood, cyc): return np.digitize(np.maximum(flood, cyc), [.3, .55, .8])

@node("flood_index", "cyclone_near", "river_m")
def route_cut_prob(e, flood, cyc, river):
    return np.clip(.6 * flood + .3 * cyc + .1 * river / 9 + e.normal(0, .03), 0, 1)


# --- people: unrest, strikes, labour, communications ------------------------------
@node("month")
def festival_demand(e, m): return 1 + .45 * np.isin(m, [4, 5, 10, 11, 12]) + e.normal(0, .05)

@node()
def unrest(e): return sig(-1.6 + e.ar(.98) + .6 * e.ar(.8, True))

@node("unrest")
def strike(e, u): return (e.ar(.75, True) + 3.2 * u > 3.0).astype(int)

@node("unrest")
def curfew(e, u): return (e.ar(.7, True) + 3.2 * u > 4.0).astype(int)

@node("unrest", "strike")
def extortion(e, u, s): return sig(-2 + 2 * u + .8 * s + e.normal(0, .5))

@node("forex_pressure", "fuel_usd")
def outage_h(e, fx, fuel): return np.clip(e.gamma(1.3, 1.2) * (1 + .6 * fx + .25 * (fuel - 75) / 25), 0, 12)

@node("festival_demand", "unrest")
def labor_shortage(e, fest, u): return sig(-1.8 + 2 * (fest - 1) + 1.5 * u + e.normal(0, .4))

@node("unrest", "curfew")
def internet_outage(e, u, c): return (e.random() < .01 + .1 * u + .5 * c).astype(int)

@node("flood_index", "cyclone_near", "unrest")
def distress_calls(e, flood, cyc, u):
    return np.clip(e.lognormal(2.5, .7) * (1 + 2 * flood + 1.2 * cyc + .8 * u), 0, 1200)


# --- demand and policy response ---------------------------------------------------
@node("alert_level", "unrest", "distress_calls", "internet_outage")
def panic(e, alert, u, calls, net):
    return sig(-2.6 + .9 * alert + 2 * u + 1.2 * np.clip(calls / 300, 0, 1) + .6 * net + e.normal(0, .3))

@node("panic", "festival_demand", "density")
def demand_surge(e, panic, fest, dens): return panic + .8 * (fest - 1) + .0002 * (dens - 1500)

@node("panic", "alert_level")  # state market intervention (TCB-style), triggered by stress
def tcb(e, panic, alert): return (e.random() < sig(-3.5 + 5 * panic + .5 * alert)).astype(int)

@node("alert_level", "forex_pressure")  # transport / fuel subsidy
def subsidy(e, alert, fx): return (e.random() < sig(-2.5 + .8 * alert + 1.5 * fx)).astype(int)

@node("alert_level")
def ngo_relief(e, alert): return (e.random() < .05 + .4 * (alert >= 1)).astype(int)

@node("extortion", "panic", "tcb", "unrest")
def hoarding(e, ext, panic, tcb, u): return sig(-2 + 1.6 * ext + 1.8 * panic + .6 * u - tcb + e.normal(0, .3))


# --- transport state at dispatch --------------------------------------------------
@node("flood_index", "strike", "festival_demand")
def bridge_congestion(e, flood, s, fest): return sig(-1.2 + 1.2 * flood + 1.2 * s + 1.5 * (fest - 1) + e.normal(0, .4))

@node("ferry_dep", "flood_index", "strike", "curfew", "bridge_congestion")
def speed_kmh(e, ferry, flood, s, c, cong):
    return np.clip(BASE_SPEED[ferry] * (1 - .5 * flood - .35 * s - .5 * c - .2 * cong) + e.normal(0, 2), 5, 80)

@node("strike", "curfew", "fuel_bdt", "subsidy")
def truck_avail(e, s, c, fuel, sub): return sig(1.4 - 2.2 * s - 2.5 * c - .035 * (fuel - 95) + .5 * sub + e.normal(0, .3))

@node("urban_pct", "outage_h", "flood_index")
def cold_chain(e, urban, out, flood): return sig(1.2 + .015 * (urban - 50) - .3 * out - flood + e.normal(0, .3))


# --- realised during transit ------------------------------------------------------
@node("extortion", "strike", "curfew")
def checkpost_h(e, ext, s, c): return np.clip(e.gamma(1.6, .5) * (1 + 1.5 * ext + .8 * s + c), 0, 24)

@node("ferry_dep", "flood_index", "bridge_congestion")
def ferry_wait_h(e, ferry, flood, cong):
    wait = e.gamma(1.6, .9) * (1 + 1.7 * flood + .8 * cong)
    return np.clip(np.where(ferry == 1, wait, .05 * wait), 0, 48)

@node("route_km", "speed_kmh", "ferry_wait_h")
def travel_h(e, km, v, wait): return km / v + wait

@node("travel_h", "route_km", "ferry_dep", "checkpost_h", "route_cut_prob", "truck_avail", "internet_outage", "hoarding")
def delay_hours(e, travel, km, ferry, post, cut, trucks, net, hoard):
    slack = travel - km / BASE_SPEED[ferry]  # slowdown and ferry queues beyond free-flow time
    return np.clip(slack + post + 14 * cut + 6 * (1 - trucks) + 1.5 * net + 3 * hoard + e.normal(0, 1), 0, 120)

@node("perishability", "cold_chain", "route_km", "ferry_dep", "delay_hours", "temp_c")
def spoilage_pct(e, perish, cold, km, ferry, delay, temp):
    hours = km / BASE_SPEED[ferry] + delay
    return np.clip(perish * (1 + 8 * (1 - cold)) * (1 + hours / 24) * (1 + .04 * (temp - 25)) + e.normal(0, .3), 0, 60)

@node("hoarding", "input_shock", "labor_shortage", "truck_avail", "demand_surge", "delay_hours", "tcb", "ngo_relief")
def shortfall(e, hoard, shock, labor, trucks, demand, delay, tcb, ngo):
    z = -3.2 + 2 * hoard + 3 * shock + 1.2 * labor + 1.4 * (1 - trucks) + 1.5 * demand + .03 * delay
    return sig(z - .6 * tcb - .4 * ngo + e.normal(0, .3))

@node("delay_hours", "spoilage_pct", "shortfall")
def condition_score(e, delay, spoil, short): return np.clip(105 - 2 * delay - spoil - 35 * short + e.normal(0, 2), 0, 100)

@node("condition_score")
def condition_class(e, score): return np.select([score >= 75, score >= 50], [0, 1], 2)


def simulate(days=3000, seed=42, start="2016-01-01", do=None):
    """Draw the SCM. `do` maps node -> constant or fn(natural values); descendants respond."""
    do = do or {}
    dates = pd.DatetimeIndex(np.tile(pd.date_range(start, periods=days).to_numpy(), D))
    cols = {}
    for i, (name, (parents, fn)) in enumerate(SCM.items()):
        e = Env(np.random.default_rng([seed, i]), days, dates)
        x = np.asarray(fn(e, *(cols[p] for p in parents)))
        if name in do:
            x = do[name](x) if callable(do[name]) else np.full(e.n, do[name])
        cols[name] = x
    df = pd.DataFrame({"timestamp": dates, **cols})
    return df.sort_values(["timestamp", "district"], ignore_index=True)


def generate(days, out, seed, start):
    df = simulate(days, seed, start)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{out}: {len(df):,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simulate the Bangladesh supply-route SCM.")
    p.add_argument("--days", type=int, default=3000, help="days per district (rows = days x 31)")
    p.add_argument("--out", default="data/routes.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", default="2016-01-01")
    a = p.parse_args()
    generate(a.days, a.out, a.seed, a.start)
