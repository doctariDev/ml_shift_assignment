import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date as date_cls, time as time_cls
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# =========================
# Timezone helper (optional)
# =========================
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

def dt_parse_iso(s: str, tz: Optional[str]) -> datetime:
    """
    Parse ISO string with possible Z/offset; convert to customer tz if provided.
    Returns timezone-aware datetime when tz is available and ZoneInfo is present,
    otherwise a naive datetime.
    """
    dt = pd.to_datetime(s)
    if tz and ZoneInfo:
        try:
            # If dt has tzinfo (offset or Z), convert; else assume UTC then convert to tz
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            dt = dt.tz_convert(ZoneInfo(tz))
        except Exception:
            try:
                dt = pd.to_datetime(s).to_pydatetime()
            except Exception:
                pass
    return dt.to_pydatetime()

# =========================
# Holidays: day-based matching
# =========================

def parse_holiday_days(holiday_list: List[Dict[str, Any]]) -> set:
    """
    Build a set of holiday dates (YYYY-MM-DD) from public_holidays entries.
    Only considers the 'date' field as the holiday day identifier.
    """
    days = set()
    for h in holiday_list or []:
        dstr = h.get("date")
        if not dstr:
            continue
        try:
            day = pd.to_datetime(dstr).date()
            days.add(day)
        except Exception:
            continue
    return days

def is_holiday_day(shift_start_dt: datetime, holiday_days: set) -> bool:
    """
    Returns True if the local date of shift_start_dt is in holiday_days.
    Assumes shift_start_dt was already localized to customer tz if applicable.
    """
    try:
        local_day = shift_start_dt.date()
        return local_day in holiday_days
    except Exception:
        return False

# =========================
# Visualization utilities
# =========================


# =========================
# Core utilities
# =========================

def get_feature_importance_dict(model, features):
    imp_gain = model.feature_importance(importance_type="gain")
    return sorted([{ "feature": f, "gain": float(g)} for f, g in zip(features, imp_gain)], key=lambda x: -x["gain"])

def parse_time_simple(tstr: str) -> time_cls:
    parts = list(map(int, tstr.split(":")))
    if len(parts) == 2:
        return time_cls(parts[0], parts[1])
    return time_cls(parts[0], parts[1], parts[2])

def to_date(obj) -> Optional[date_cls]:
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj.replace("Z", "+00:00")).date()
        except Exception:
            return pd.to_datetime(obj).date()
    if isinstance(obj, datetime):
        return obj.date()
    if hasattr(obj, "to_pydatetime"):
        return obj.to_pydatetime().date()
    if hasattr(obj, "date"):
        try:
            return obj.date()
        except Exception:
            pass
    if isinstance(obj, date_cls):
        return obj
    raise TypeError(f"Unsupported date type: {type(obj)}")

def in_date_range(date_obj, start: Optional[str], end: Optional[str]) -> bool:
    d = to_date(date_obj)
    if d is None:
        return True
    if start:
        if d < to_date(start):
            return False
    if end:
        if d > to_date(end):
            return False
    return True

# =========================
# Feasibility checks (doctari schema)
# =========================

def user_qualified(user: Dict[str, Any], required_quals: List[int]) -> bool:
    uq = {q["id"] for q in user.get("qualifications", [])}
    return set(required_quals or []).issubset(uq)

def user_active(user: Dict[str, Any], date_obj) -> bool:
    return True

def user_available_for_shift(user: Dict[str, Any], shift: Dict[str, Any], customer_tz: Optional[str]) -> bool:
    return True

def conflicts_with_parallel(shift: Dict[str, Any], user_id: str, assigned_by_user: Dict[str, List[int]], shift_index: Dict[int, Dict[str, Any]]) -> bool:
    for sid in assigned_by_user.get(user_id, []):
        other = shift_index.get(sid)
        if not other:
            continue
        if shift["date"] == other["date"]:
            s_start = parse_time_simple(shift["start"])
            s_end   = parse_time_simple(shift["end"])
            o_start = parse_time_simple(other["start"])
            o_end   = parse_time_simple(other["end"])
            if not (s_end <= o_start or o_end <= s_start):
                return True
    return False

def conflicts_with_exceptions(shift: Dict[str, Any], user_id: str, all_assignments_by_user: Dict[str, List[Dict[str, Any]]]) -> bool:
    return False

def negative_wish(shift: Dict[str, Any], user_id: str) -> bool:
    return False

def positive_wishers(shift: Dict[str, Any]) -> List[str]:
    return []

# =========================
# Adapters: doctari JSON -> unified frames (holiday by day)
# =========================

def adapt_past_plans_to_frames(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    past_plans = data.get("past_shift_plans", []) or []
    default_tz = ((data.get("shift_plan") or {}).get("customer") or {}).get("zone_id")

    shift_rows = []
    asg_rows = []

    for p in past_plans:
        tz = (p.get("customer") or {}).get("zone_id") or default_tz
        holiday_days = parse_holiday_days(p.get("public_holidays") or [])

        for sh in p.get("shifts", []):
            sid = int(sh["id"])
            start_dt_local = dt_parse_iso(sh["start_date_time"], tz)
            end_dt_local = dt_parse_iso(sh["end_date_time"], tz)
            is_hol = is_holiday_day(start_dt_local, holiday_days)
            shift_rows.append({
                "id": sid,
                "unit": str(sh.get("workplace_id")),
                "shiftType": str(sh.get("shift_card_id") or "GEN"),
                "weekday": int(start_dt_local.weekday()),
                "date": start_dt_local.date().isoformat(),
                "start": start_dt_local.strftime("%H:%M"),
                "end": end_dt_local.strftime("%H:%M"),
                "isHoliday": int(is_hol),
                "requiredQualifications": [int(q) for q in sh.get("qualification_ids", [])]
            })
        for a in p.get("shift_assignments", []):
            asg_rows.append({"shiftId": int(a["shift_id"]), "userId": a["employee_uuid"]})

    hist_shifts_df = pd.DataFrame(shift_rows)
    assignments_df = pd.DataFrame(asg_rows)
    return hist_shifts_df, assignments_df

def adapt_target_plan_to_frames(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]], Optional[str]]:
    sp = data["shift_plan"]
    customer_tz = (sp.get("customer") or {}).get("zone_id")
    holiday_days = parse_holiday_days(sp.get("public_holidays") or [])

    target_shifts = []
    for sh in sp.get("shifts", []):
        sid = int(sh["id"])
        start_dt_local = dt_parse_iso(sh["start_date_time"], customer_tz)
        end_dt_local = dt_parse_iso(sh["end_date_time"], customer_tz)
        is_hol = is_holiday_day(start_dt_local, holiday_days)
        target_shifts.append({
            "id": sid,
            "unit": str(sh.get("workplace_id")),
            "shiftType": str(sh.get("shift_card_id") or "GEN"),
            "weekday": int(start_dt_local.weekday()),
            "date": start_dt_local.date().isoformat(),
            "start": start_dt_local.strftime("%H:%M"),
            "end": end_dt_local.strftime("%H:%M"),
            "isHoliday": int(is_hol),
            "requiredQualifications": [int(q) for q in sh.get("qualification_ids", [])],
            "preplannedUserId": None,
            "requests": []
        })

    shift_index = {s["id"]: s for s in target_shifts}
    users_by_id = {e["uuid"]: e for e in sp.get("employees", [])}
    return target_shifts, shift_index, users_by_id, customer_tz

# =========================
# Stats, training data
# =========================

def recency_weight(ts: datetime, now: datetime, lam=0.85, unit="week") -> float:
    delta_days = (now - ts).days
    k = max(delta_days / 7.0, 0) if unit == "week" else max(delta_days, 0)
    return lam ** k

def collect_history_stats(hist_shifts_df: pd.DataFrame,
                          assignments_df: pd.DataFrame,
                          users_df: pd.DataFrame,
                          lam: float = 0.85) -> Dict[str, Dict[Tuple, Dict[str, float]]]:
    """
    Build per-user, per-context recency-weighted stats, including holiday-only rates.
    Context = (unit, shiftType, weekday).
    For each (user, context):
      - rw_assign_rate: overall recency-weighted assignment rate proxy
      - rw_assign_rate_holiday: recency-weighted rate on holiday shifts only
      - count_assigned / count_occurrences: totals
      - count_assigned_holiday / count_occurrences_holiday: holiday-only totals
      - last_assigned_days: recency marker (overall)
    """
    if hist_shifts_df.empty:
        return {}
    df = hist_shifts_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    now = df["date"].max()
    now_dt = pd.to_datetime(now) if pd.notnull(now) else pd.Timestamp(datetime.utcnow())

    stats = defaultdict(lambda: defaultdict(lambda: {
        "rw_denom": 0.0, "rw_num": 0.0,
        "count_occurrences": 0, "count_assigned": 0,
        "last_assigned_days": 9999.0,
        # holiday-specific
        "rw_denom_holiday": 0.0, "rw_num_holiday": 0.0,
        "count_occurrences_holiday": 0, "count_assigned_holiday": 0
    }))

    df["context"] = df.apply(lambda r: (r["unit"], r["shiftType"], r["weekday"]), axis=1)
    context_by_shift = dict(zip(df["id"], df["context"]))
    isHoliday_by_shift = dict(zip(df["id"], df["isHoliday"]))

    if assignments_df.empty:
        # Compute occurrence counts without assignments if needed
        return stats

    merged = assignments_df.merge(df[["id", "unit", "shiftType", "weekday", "date", "isHoliday"]],
                                  left_on="shiftId", right_on="id", how="left")
    merged["date"] = pd.to_datetime(merged["date"])

    # We also want to count occurrences per context for users we sample as negatives.
    # For simplicity, we use assignment-driven stats plus recency/occurrence counters keyed off assignments.
    # If you want full occurrence counts independent of assignments, extend with per-context shift frequencies.

    for _, row in merged.iterrows():
        uid = row["userId"]
        sid = row["shiftId"]
        if sid not in context_by_shift:
            continue
        ctx = context_by_shift[sid]
        date_dt = row["date"]
        if pd.isnull(date_dt):
            continue
        w = recency_weight(date_dt, now_dt, lam=lam, unit="week")
        sh_is_hol = int(row.get("isHoliday", 0)) == 1
        s = stats[uid][ctx]

        # overall
        s["rw_num"] += w
        s["rw_denom"] += w
        s["count_assigned"] += 1
        s["count_occurrences"] += 1  # using assignments as occurrences proxy here
        s["last_assigned_days"] = min(s["last_assigned_days"], (now_dt - date_dt).days)

        # holiday-only
        if sh_is_hol:
            s["rw_num_holiday"] += w
            s["rw_denom_holiday"] += w
            s["count_assigned_holiday"] += 1
            s["count_occurrences_holiday"] += 1

    for uid, ctxs in stats.items():
        for ctx, s in ctxs.items():
            s["rw_assign_rate"] = (s["rw_num"] / max(s["rw_denom"], 1e-6))
            s["rw_assign_rate_holiday"] = (s["rw_num_holiday"] / max(s["rw_denom_holiday"], 1e-6)) if s["rw_denom_holiday"] > 0 else 0.0

    return stats

def build_training_data(hist_shifts_df: pd.DataFrame,
                        assignments_df: pd.DataFrame,
                        users_df: pd.DataFrame,
                        stats_by_user_ctx: Dict[str, Dict[Tuple, Dict[str, float]]],
                        k_neg_per_pos: int = 5) -> pd.DataFrame:
    if assignments_df.empty:
        raise ValueError("assignments_df is empty; need historical assignments to train.")
    shift_ids = set(assignments_df["shiftId"].unique())
    hist_shifts_df = hist_shifts_df[hist_shifts_df["id"].isin(shift_ids)].copy()
    hist_shifts_df = hist_shifts_df.sort_values(["date", "start"]).drop_duplicates(subset=["id"], keep="first")
    shift_meta = hist_shifts_df.set_index("id").to_dict(orient="index")
    users_by_id = users_df.set_index("uuid").to_dict(orient="index")

    rows = []
    rng = np.random.default_rng(42)

    for sid, g in assignments_df.groupby("shiftId"):
        if sid not in shift_meta:
            continue
        sm = shift_meta[sid]
        ctx = (sm["unit"], sm["shiftType"], sm["weekday"])
        start_s = sm["start"]
        end_s = sm["end"]
        try:
            hour_val = int(str(start_s)[:2])
        except Exception:
            hour_val = 8
        try:
            s_date = to_date(sm["date"])
            duration_val = (datetime.combine(s_date, parse_time_simple(end_s)) -
                            datetime.combine(s_date, parse_time_simple(start_s))).seconds / 3600.0
        except Exception:
            duration_val = 8.0

        req = sm.get("requiredQualifications", [])

        compatible = []
        for uid, u in users_by_id.items():
            if not set(req or []).issubset({q["id"] for q in u.get("qualifications", [])}):
                continue
            compatible.append(uid)

        pos_users = list(g["userId"].values)
        for uid in pos_users:
            sstats = stats_by_user_ctx.get(uid, {}).get(ctx, {})
            urec = users_by_id.get(uid, {})
            rows.append({
                "shiftId": sid, "userId": uid, "y": 1,
                "unit": sm["unit"], "shiftType": sm["shiftType"], "weekday": sm["weekday"],
                "hour": hour_val, "duration": duration_val, "isHoliday": int(sm.get("isHoliday", 0)),
                "rw_assign_rate": sstats.get("rw_assign_rate", 0.0),
                "count_assigned": sstats.get("count_assigned", 0),
                "last_assigned_days": sstats.get("last_assigned_days", 9999.0),
                "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0
            })
        neg_pool = [u for u in compatible if u not in pos_users]
        if len(neg_pool) > 0:
            sample_size = min(k_neg_per_pos * len(pos_users), len(neg_pool))
            neg_sample = list(rng.choice(neg_pool, size=sample_size, replace=False))
            for uid in neg_sample:
                sstats = stats_by_user_ctx.get(uid, {}).get(ctx, {})
                urec = users_by_id.get(uid, {})
                rows.append({
                    "shiftId": sid, "userId": uid, "y": 0,
                    "unit": sm["unit"], "shiftType": sm["shiftType"], "weekday": sm["weekday"],
                    "hour": hour_val, "duration": duration_val, "isHoliday": int(sm.get("isHoliday", 0)),
                    "rw_assign_rate": sstats.get("rw_assign_rate", 0.0),
                    "count_assigned": sstats.get("count_assigned", 0),
                    "last_assigned_days": sstats.get("last_assigned_days", 9999.0),
                    "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Training data ended up empty. Check past_shift_plans.shift_assignments and ids.")
    df["unit"] = df["unit"].astype("category")
    df["shiftType"] = df["shiftType"].astype("category")
    return df

# =========================
# Model training
# =========================

def train_and_calibrate_with_val(df: pd.DataFrame) -> Tuple[Any, Any, List[str], Tuple[pd.DataFrame, pd.Series]]:
    features = ["unit", "shiftType", "weekday", "hour", "duration", "isHoliday",
                "rw_assign_rate", "count_assigned", "last_assigned_days", "userFTE"]
    X = df[features]
    y = df["y"]
    cat_features = ["unit", "shiftType"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features, free_raw_data=False)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "seed": 42
    }
    model = lgb.train(params, lgb_train, num_boost_round=500)
    p_raw_val = model.predict(X_val)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw_val, y_val)
    auc_val = roc_auc_score(y_val, p_raw_val)
    ap_val = average_precision_score(y_val, p_raw_val)
    brier = brier_score_loss(y_val, iso.predict(p_raw_val))
    print(f"[Validation] AUC={auc_val:.3f} AP={ap_val:.3f} Brier={brier:.3f}")
    return model, iso, features, (X_val, y_val)

# =========================
# Scoring and assignment
# =========================

def score_candidates_for_shift(shift: Dict[str, Any],
                               candidate_ids: List[str],
                               users_by_id: Dict[str, Dict[str, Any]],
                               stats_by_user_ctx: Dict[str, Dict[Tuple, Dict[str, float]]],
                               model, iso_calibrator, features: List[str]) -> List[Tuple[str, float]]:
    ctx = (shift["unit"], shift["shiftType"], shift["weekday"])
    try:
        hour = int(str(shift["start"])[:2])
    except Exception:
        hour = 8
    try:
        s_date = to_date(shift["date"])
        duration = (datetime.combine(s_date, parse_time_simple(shift["end"])) -
                    datetime.combine(s_date, parse_time_simple(shift["start"]))).seconds / 3600.0
    except Exception:
        duration = 8.0
    isHoliday = int(shift.get("isHoliday", 0))
    rows, idx = [], []
    for uid in candidate_ids:
        s = stats_by_user_ctx.get(uid, {}).get(ctx, {})
        urec = users_by_id.get(uid, {})
        rows.append({
            "unit": shift["unit"], "shiftType": shift["shiftType"], "weekday": shift["weekday"],
            "hour": hour, "duration": duration, "isHoliday": isHoliday,
            "rw_assign_rate": s.get("rw_assign_rate", 0.0),
            "count_assigned": s.get("count_assigned", 0),
            "last_assigned_days": s.get("last_assigned_days", 9999.0),
            "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0
        })
        idx.append(uid)
    if not rows:
        return []
    X = pd.DataFrame(rows)
    X["unit"] = X["unit"].astype("category")
    X["shiftType"] = X["shiftType"].astype("category")
    p_raw = model.predict(X[features])
    p = iso_calibrator.predict(p_raw)
    return list(zip(idx, p))  # do not sort here; we'll sort after blending

def assign_target_period(target_shifts: List[Dict[str, Any]],
                         users_by_id: Dict[str, Dict[str, Any]],
                         shift_index: Dict[int, Dict[str, Any]],
                         model, iso_calibrator, features,
                         stats_by_user_ctx: Dict[str, Dict[Tuple, Dict[str, float]]],
                         fairness_weekly_cap_hours: Optional[float] = None,
                         customer_tz: Optional[str] = None,
                         top_k: int = 5) -> Tuple[Dict[int, str], Dict[str, Any]]:
    assigned_by_user: Dict[str, List[int]] = defaultdict(list)
    assigned_hours_by_user: Dict[str, float] = defaultdict(float)
    all_assignments_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    result = {}
    report = {
        "shifts": [],
        "usersSummary": {},
        "modelSummary": {},
        "runConfig": {"fairness_weekly_cap_hours": fairness_weekly_cap_hours, "top_k": top_k}
    }

    def shift_hours(s):
        try:
            start = parse_time_simple(s["start"])
            end = parse_time_simple(s["end"])
            dt = datetime.combine(to_date(s["date"]), end) - datetime.combine(to_date(s["date"]), start)
            return dt.seconds / 3600.0
        except Exception:
            return 8.0

    # Process in chronological order
    target_shifts = sorted(target_shifts, key=lambda x: (x["date"], x["start"]))
    for s in target_shifts:
        sid = s["id"]
        explanation = {
            "shiftId": sid,
            "meta": {
                "unit": s["unit"], "shiftType": s["shiftType"], "weekday": s["weekday"],
                "date": s["date"], "start": s["start"], "end": s["end"],
                "requiredQualifications": s.get("requiredQualifications", []),
                "isHoliday": int(s.get("isHoliday", 0))
            },
            "preplannedUserId": s.get("preplannedUserId"),
            "positiveWishers": positive_wishers(s),
            "candidatesBeforeFilters": [],
            "filteredOut": [],
            "candidatesAfterFilters": [],
            "topCandidates": [],
            "chosen": None,
            "decisionPath": [],
            "notes": []
        }

        # Candidate generation
        explanation["decisionPath"].append("candidate_generation")
        all_user_ids = list(users_by_id.keys())
        for uid in all_user_ids:
            reasons = []
            user = users_by_id[uid]
            if not user_qualified(user, s.get("requiredQualifications", [])):
                reasons.append("not_qualified")
            if not user_available_for_shift(user, s, customer_tz):
                reasons.append("not_available")
            if conflicts_with_parallel(s, uid, assigned_by_user, shift_index):
                reasons.append("parallel_conflict")
            if negative_wish(s, uid):
                reasons.append("negative_wish")
            if reasons:
                explanation["filteredOut"].append({"userId": uid, "reasons": reasons})
            else:
                explanation["candidatesAfterFilters"].append(uid)
        explanation["candidatesBeforeFilters"] = len(all_user_ids)

        if not explanation["candidatesAfterFilters"]:
            explanation["notes"].append("no feasible candidates")
            report["shifts"].append(explanation)
            continue

        # Scoring
        explanation["decisionPath"].append("scoring")
        ranked = score_candidates_for_shift(s, explanation["candidatesAfterFilters"], users_by_id, stats_by_user_ctx, model, iso_calibrator, features)

        # Prepare stats for blending
        ctx = (s["unit"], s["shiftType"], s["weekday"])
        stats_for_uid = {uid: (stats_by_user_ctx.get(uid, {}) or {}).get(ctx, {}) for uid, _ in ranked}

        # Top-k details (report before blending)
        top_list_report = sorted(ranked, key=lambda t: t[1], reverse=True)[:top_k]
        enriched_top = []
        for uid, p in top_list_report:
            sstats = stats_for_uid.get(uid, {}) or {}
            urec = users_by_id.get(uid, {})
            enriched_top.append({
                "userId": uid,
                "p_calibrated": float(p),
                "rw_assign_rate": sstats.get("rw_assign_rate", 0.0),
                "rw_assign_rate_holiday": sstats.get("rw_assign_rate_holiday", 0.0),
                "count_assigned": int(sstats.get("count_assigned", 0)),
                "count_assigned_holiday": int(sstats.get("count_assigned_holiday", 0)),
                "last_assigned_days": float(sstats.get("last_assigned_days", 9999.0)),
                "userFTE": (urec.get("timed_properties", [{}])[0].get("weekly_hours", 40.0))/40.0
            })
        explanation["topCandidates"] = enriched_top

        if not ranked:
            explanation["notes"].append("scoring produced no results")
            report["shifts"].append(explanation)
            continue

        # Blend score: p_cal + bonuses from history to break ties
        beta1, beta2, beta3 = 0.10, 0.05, 0.05  # weights
        C, L = 5.0, 60.0  # caps for count and recency window (days)
        is_holiday = int(s.get("isHoliday", 0)) == 1
        holiday_weight = 0.08  # gentle extra nudge for holiday specialists; tune as needed

        def blended_score(uid: str, p: float) -> float:
            st = stats_for_uid.get(uid, {}) or {}
            rw = float(st.get("rw_assign_rate", 0.0))
            rw_hol = float(st.get("rw_assign_rate_holiday", 0.0))
            cnt = float(st.get("count_assigned", 0.0))
            last_days = float(st.get("last_assigned_days", 9999.0))
            cnt_norm = min(cnt / C, 1.0)
            recency_bonus = max(0.0, 1.0 - min(last_days, L) / L)
            holiday_bonus = holiday_weight * rw_hol if is_holiday else 0.0
            return float(p) + beta1 * rw + beta2 * cnt_norm + beta3 * recency_bonus + holiday_bonus

        ranked_blended = sorted(ranked, key=lambda t: blended_score(t[0], t[1]), reverse=True)

        # Greedy pick with fairness cap
        explanation["decisionPath"].append("greedy_pick_with_fairness")
        hrs = shift_hours(s)
        chosen = None
        chosen_p = None
        fairness_skips = []
        for uid, p in ranked_blended:
            if fairness_weekly_cap_hours and assigned_hours_by_user[uid] + hrs > fairness_weekly_cap_hours:
                fairness_skips.append({"userId": uid, "p": float(p), "reason": "fairness_cap"})
                continue
            chosen = uid
            chosen_p = float(p)
            break

        if chosen:
            result[sid] = chosen
            assigned_by_user[chosen].append(sid)
            assigned_hours_by_user[chosen] += hrs
            all_assignments_by_user[chosen].append(s)
            explanation["chosen"] = {"userId": chosen, "p_calibrated": chosen_p}
            if fairness_skips:
                explanation["notes"].append({"fairness_skipped": fairness_skips})
        else:
            explanation["notes"].append("no candidate satisfied fairness cap")
        report["shifts"].append(explanation)

    for uid in users_by_id.keys():
        report["usersSummary"][uid] = {
            "totalHours": float(assigned_hours_by_user.get(uid, 0.0)),
            "assignedCount": len(assigned_by_user.get(uid, []))
        }
    return result, report

# =========================
# Pipeline for doctari JSON
# =========================

def run_pipeline_on_doctari_json(job_json_path: str,
                                 fairness_weekly_cap_hours: Optional[float] = None,
                                 visualization_mode: bool = False,
                                 output_dir: str = "viz_out"):
    with open(job_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Historical
    hist_shifts_df, assignments_df = adapt_past_plans_to_frames(data)
    if hist_shifts_df.empty or assignments_df.empty:
        raise ValueError("No historical data found. Ensure past_shift_plans contain shifts and shift_assignments.")

    # Target
    target_shifts, shift_index, users_by_id, customer_tz = adapt_target_plan_to_frames(data)
    if not target_shifts:
        raise ValueError("No target shifts found in shift_plan.shifts.")

    users_df = pd.DataFrame(list(users_by_id.values()))

    # Stats and training
    stats_by_user_ctx = collect_history_stats(hist_shifts_df, assignments_df, users_df, lam=0.85)
    train_df = build_training_data(hist_shifts_df, assignments_df, users_df, stats_by_user_ctx, k_neg_per_pos=5)
    model, iso_cal, feats, (X_val, y_val) = train_and_calibrate_with_val(train_df)

    # Assign target
    result, report = assign_target_period(
        target_shifts=target_shifts,
        users_by_id=users_by_id,
        shift_index=shift_index,
        model=model,
        iso_calibrator=iso_cal,
        features=feats,
        stats_by_user_ctx=stats_by_user_ctx,
        fairness_weekly_cap_hours=fairness_weekly_cap_hours,
        customer_tz=customer_tz,
        top_k=5
    )
    report["modelSummary"] = {"featureImportance": get_feature_importance_dict(model, feats)}

    print("Hist shifts:", hist_shifts_df.shape, "Assignments:", assignments_df.shape)
    print("Distinct shiftIds with assignments:", assignments_df["shiftId"].nunique())
    print("Missing shiftIds in history:", len(set(assignments_df["shiftId"]) - set(hist_shifts_df["id"])))
    print(hist_shifts_df[["unit", "shiftType", "weekday"]].drop_duplicates().shape)

    # Write back assignments into shift_plan.shift_assignments
    target_shift_ids = {int(s["id"]) for s in data["shift_plan"].get("shifts", [])}
    existing = data["shift_plan"].get("shift_assignments") or []
    kept = [a for a in existing if int(a["shift_id"]) not in target_shift_ids]
    for sid, uid in result.items():
        kept.append({"shift_id": int(sid), "employee_uuid": uid, "source": "MODEL"})
    data["shift_plan"]["shift_assignments"] = kept

    # Visualization
    if visualization_mode:
        ensure_dir(output_dir)
        viz = render_visualizations(model, iso_cal, feats, X_val, y_val, report,
                                    target_shifts_count=len(target_shifts),
                                    output_dir=output_dir, sample_topk_shifts=6)
        data["reportHtml"] = viz["html"]
        print(f"Visualization written to: {viz['html']}")

    # Attach report
    data["assignmentReport"] = report
    return data

# =========================
# CLI
# =========================

if __name__ == "__main__":
    output = run_pipeline_on_doctari_json(
        job_json_path="input_files/generated_example_3.json",
        fairness_weekly_cap_hours=40.0,
        visualization_mode=True,
        output_dir="viz_out"
    )
    with open("output_job.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    assigned_pairs = [(a["shift_id"], a["employee_uuid"]) for a in output["shift_plan"]["shift_assignments"] if a.get("source") == "MODEL"]
    print(f"Assigned {len(assigned_pairs)} shifts by model, e.g.:", assigned_pairs[:10])
