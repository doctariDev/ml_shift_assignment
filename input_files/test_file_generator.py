import json
from datetime import datetime, timedelta, date
import random

# ------------- Config -------------
RANDOM_SEED = 42
HISTORY_WEEKS = 16
TARGET_WEEKS = 10
START_MONDAY = "2025-12-08"  # YYYY-MM-DD, should be a Monday
UNITS = (10, 20, 30)
DAILY_MIN = 10
DAILY_MAX = 12
CUSTOMER = {"id": 200, "name": "General Hospital", "zone_id": "Europe/Berlin"}
# ----------------------------------

random.seed(RANDOM_SEED)

def dt_local(d: date, hour: int) -> str:
    # Europe/Berlin DST switch: last Sunday in March -> +02:00 after 02:00 local time
    def last_sunday_of_march(year):
        d = datetime(year, 3, 31)
        while d.weekday() != 6:
            d -= timedelta(days=1)
        return d.date()
    offset = 1
    march_last_sun = last_sunday_of_march(d.year)
    if d >= march_last_sun:
        offset = 2
    sign = "+" if offset >= 0 else "-"
    return f"{d.isoformat()}T{hour:02d}:00:00{sign}{abs(offset):02d}:00"

EMPLOYEES = [
    {"uuid": "E0001", "name": "Alice", "qualifications": [{"id": 1}], "timed_properties": [{"weekly_hours": 40}]},
    {"uuid": "E0002", "name": "Bob", "qualifications": [{"id": 1}], "timed_properties": [{"weekly_hours": 40}]},
    {"uuid": "E0003", "name": "Carol", "qualifications": [{"id": 1}, {"id": 2}], "timed_properties": [{"weekly_hours": 32}]},
    {"uuid": "E0004", "name": "Dave", "qualifications": [{"id": 2}], "timed_properties": [{"weekly_hours": 20}]},
    {"uuid": "E0005", "name": "Eve", "qualifications": [{"id": 1}], "timed_properties": [{"weekly_hours": 30}]},
    {"uuid": "E0006", "name": "Frank", "qualifications": [{"id": 2}], "timed_properties": [{"weekly_hours": 38}]},
    {"uuid": "E0007", "name": "Grace", "qualifications": [{"id": 1}], "timed_properties": [{"weekly_hours": 24}]},
    {"uuid": "E0008", "name": "Heidi", "qualifications": [{"id": 1}, {"id": 2}], "timed_properties": [{"weekly_hours": 40}]},
    {"uuid": "E0009", "name": "Ivan", "qualifications": [{"id": 2}], "timed_properties": [{"weekly_hours": 36}]},
    {"uuid": "E0010", "name": "Judy", "qualifications": [{"id": 1}], "timed_properties": [{"weekly_hours": 28}]},
]

SHIFT_DEFS = {
    "DAY":   {"start": 8,  "end": 16, "quals": [1]},
    "EVE":   {"start": 16, "end": 22, "quals": [2]},
    "NIGHT": {"start": 22, "end": 6,  "quals": [1, 2]},  # crosses midnight
}

def choose_employee(qual_required):
    cands = []
    for e in EMPLOYEES:
        have = {q["id"] for q in e["qualifications"]}
        if set(qual_required).issubset(have):
            cands.append(e["uuid"])
    return random.choice(cands or [e["uuid"] for e in EMPLOYEES])

def recurring_rules(sh, weekday):
    # Monday DAY at unit 10 -> E0001
    if weekday == 0 and sh["shift_card_id"] == "DAY" and sh["workplace_id"] == 10:
        return "E0001"
    # Thursday DAY at unit 10 -> E0002
    if weekday == 3 and sh["shift_card_id"] == "DAY" and sh["workplace_id"] == 10:
        return "E0002"
    # Friday NIGHT at unit 20 -> E0008
    if weekday == 4 and sh["shift_card_id"] == "NIGHT" and sh["workplace_id"] == 20:
        return "E0008"
    return None

def generate_unique_day_shifts(day: date, units, n_shifts, max_attempts=200):
    # Prevent duplicates per (unit, shift_type, start_hour) for a given day
    keys = set()
    shifts = []
    attempts = 0

    # target mix weights; you can tweak these
    types = ["DAY", "EVE", "NIGHT"]
    weights = [4, 4, 2]  # ~40% day, 40% eve, 20% night

    while len(shifts) < n_shifts and attempts < max_attempts:
        attempts += 1
        t = random.choices(types, weights=weights, k=1)[0]
        unit = random.choice(units)
        start_h = SHIFT_DEFS[t]["start"]
        key = (unit, t, start_h)
        if key in keys:
            continue
        keys.add(key)

        start_iso = dt_local(day, start_h)
        end_h = SHIFT_DEFS[t]["end"]
        end_day = day if end_h > start_h else day + timedelta(days=1)
        end_iso = dt_local(end_day, end_h)

        shifts.append({
            "id": None,
            "workplace_id": unit,
            "shift_card_id": t,
            "start_date_time": start_iso,
            "end_date_time": end_iso,
            "qualification_ids": SHIFT_DEFS[t]["quals"]
        })
    return shifts

def generate_week(day0: date, units, daily_min, daily_max):
    shifts = []
    for i in range(7):
        d = day0 + timedelta(days=i)
        n = random.randint(daily_min, daily_max)
        day_shifts = generate_unique_day_shifts(d, units, n)
        shifts.extend(day_shifts)
    return shifts

def assign_history(shifts):
    assigns = []
    for sh in shifts:
        dt = datetime.fromisoformat(sh["start_date_time"].replace("Z", "+00:00"))
        weekday = dt.weekday()
        fixed = recurring_rules(sh, weekday)
        uuid = fixed if fixed else choose_employee(sh["qualification_ids"])
        assigns.append({"shift_id": sh["id"], "employee_uuid": uuid})
    return assigns

def generate_dataset():
    # History
    start_hist = datetime.fromisoformat(START_MONDAY).date()
    past_plans = []
    shift_id_counter = 9_000_000
    plan_id_counter = 9_100

    for w in range(HISTORY_WEEKS):
        week_start = start_hist + timedelta(days=7*w)
        week_shifts = generate_week(week_start, UNITS, DAILY_MIN, DAILY_MAX)
        for sh in week_shifts:
            shift_id_counter += 1
            sh["id"] = shift_id_counter
        assignments = assign_history(week_shifts)
        past_plans.append({
            "id": plan_id_counter + w + 1,
            "customer": {"zone_id": CUSTOMER["zone_id"]},
            "shifts": week_shifts,
            "shift_assignments": assignments
        })

    # Target
    start_target = start_hist + timedelta(days=7*HISTORY_WEEKS)
    target_days = 7 * TARGET_WEEKS
    target_shifts = []
    target_shift_id_counter = 300_000
    for d in range(target_days):
        day = start_target + timedelta(days=d)
        n = random.randint(DAILY_MIN, DAILY_MAX)
        day_shifts = generate_unique_day_shifts(day, UNITS, n)
        for sh in day_shifts:
            target_shift_id_counter += 1
            sh["id"] = target_shift_id_counter
        target_shifts.extend(day_shifts)

    data = {
        "shift_plan": {
            "customer": CUSTOMER,
            "employees": EMPLOYEES,
            "shifts": target_shifts,
            "shift_assignments": []
        },
        "past_shift_plans": past_plans
    }
    return data

if __name__ == "__main__":
    data = generate_dataset()
    with open("generated_example_2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Written json")