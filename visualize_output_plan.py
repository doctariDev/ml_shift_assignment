#!/usr/bin/env python3
from datetime import datetime, timedelta, date
from collections import defaultdict, OrderedDict
import pandas as pd
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import shutil
from pathlib import Path
import base64

INPUT_JSON = "output/output_job.json"
OUT_DIR = "output/calendar_visualization"
DPI = 150

# Extra output dirs to clean
VIZ_OUT_DIR = "viz_out"
CALENDER_OUT_DIR = "calender_out"  # keeping your spelling

# Colors
EMP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#a55194",
    "#6b6ecf", "#9c9ede", "#cedb9c", "#e7cb94"
]
UNASSIGNED_COLOR = "#B0B0B0"
HOLIDAY_BORDER_COLOR = "#800080"  # purple

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def empty_dir(path: str):
    """
    Delete all files and subdirectories inside 'path' without deleting 'path' itself.
    If the directory does not exist, it will be created.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)
        except Exception as e:
            print(f"Warning: could not remove {fp}: {e}")

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()

def plot_feature_importance(model, out_path: str, max_num_features=20):
    try:
        gains = model.feature_importance(importance_type="gain")
        names = model.feature_name()
        if gains is None or len(gains) == 0 or np.all(np.array(gains) == 0):
            plt.figure(figsize=(6,2))
            plt.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
            plt.axis("off")
            save_fig(out_path)
            return
        imp = sorted(zip(names, gains), key=lambda x: x[1], reverse=True)[:max_num_features]
        labels = [i[0] for i in imp]
        values = [i[1] for i in imp]
        plt.figure(figsize=(8, max(3, 0.3*len(labels)+1)))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel("Gain")
        plt.title("Feature Importance (gain)")
        save_fig(out_path)
    except Exception as e:
        plt.figure(figsize=(6,2))
        plt.text(0.5, 0.5, f"Feature importance unavailable: {e}", ha="center", va="center", wrap=True)
        plt.axis("off")
        save_fig(out_path)

def plot_calibration(y_val, p_raw, p_cal, out_path: str):
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(6,5))
    for name, p in [("Raw", p_raw), ("Calibrated", p_cal)]:
        frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker='o', label=name)
    plt.plot([0,1],[0,1], 'k--', alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    save_fig(out_path)

def plot_pr_roc(y_val, p_cal, out_pr_path: str, out_roc_path: str):
    prec, rec, _ = precision_recall_curve(y_val, p_cal)
    ap = average_precision_score(y_val, p_cal)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f})")
    save_fig(out_pr_path)
    fpr, tpr, _ = roc_curve(y_val, p_cal)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC={roc_auc:.3f})")
    save_fig(out_roc_path)

def plot_hours_per_user(assigned_hours_by_user: dict, out_path: str):
    if not assigned_hours_by_user:
        return
    users = list(assigned_hours_by_user.keys())
    hours = [assigned_hours_by_user[u] for u in users]
    plt.figure(figsize=(8,4))
    plt.bar(users, hours)
    plt.ylabel("Assigned hours"); plt.title("Workload distribution")
    plt.xticks(rotation=45, ha='right')
    save_fig(out_path)

def plot_topk_for_shift(shift_id: str, top_list: list, out_path: str, k: int = 5):
    if not top_list:
        return
    data = top_list[:k]
    users = [d["userId"] for d in data][::-1]
    probs = [d["p_calibrated"] for d in data][::-1]
    plt.figure(figsize=(6,4))
    plt.barh(users, probs)
    plt.xlabel("Calibrated probability")
    plt.title(f"Top-{k} candidates for {shift_id}")
    save_fig(out_path)

def _img_data_uri(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    suffix = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def write_html_report(out_dir: str, run_title: str, summary: dict, images: dict, assignment_report_path: str):
    html = []
    html.append(f"<html><head><meta charset='utf-8'><title>{run_title}</title></head><body>")
    html.append(f"<h1>{run_title}</h1>")
    html.append("<h2>Run Summary</h2>")
    html.append("<ul>")
    html.append(f"<li>Total shifts (target): {summary.get('target_shifts_total')}</li>")
    html.append(f"<li>Assigned shifts: {summary.get('assigned_shifts')}</li>")
    html.append(f"<li>Unassigned shifts: {summary.get('unassigned_shifts')}</li>")
    html.append(f"<li>Users with assignments: {summary.get('users_with_assignments')}</li>")
    html.append("</ul>")
    # Keep JSON as a file link; will work in a browser even if Jupyter blocks file://
    html.append(f"<p>Assignment report JSON: <a href='{os.path.basename(assignment_report_path)}'>{os.path.basename(assignment_report_path)}</a></p>")

    def img_tag(path, title):
        if not path or not os.path.exists(path):
            return ""
        data_uri = _img_data_uri(path)
        return f"<div><h3>{title}</h3><img src='{data_uri}' style='max-width:900px;'/></div>"

    html.append("<h2>Model Diagnostics</h2>")
    html.append(img_tag(images.get("feat_importance"), "Feature Importance (gain)"))
    html.append(img_tag(images.get("calibration"), "Calibration curve"))
    html.append(img_tag(images.get("pr"), "Precision–Recall"))
    html.append(img_tag(images.get("roc"), "ROC"))

    html.append("<h2>Plan Insights</h2>")
    html.append(img_tag(images.get("workload"), "Workload distribution (hours per user)"))
    topk_imgs = images.get("topk", [])
    if topk_imgs:
        html.append("<h2>Top‑k leaderboards (sampled shifts)</h2>")
        for p in topk_imgs:
            title = f"Top‑k: {os.path.splitext(os.path.basename(p))[0]}"
            html.append(img_tag(p, title))

    html.append("</body></html>")
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path

def render_visualizations(model, iso_cal, feats, X_val, y_val, report, target_shifts_count, output_dir: str, sample_topk_shifts: int = 5):
    ensure_dir(output_dir)
    images = {}
    images["feat_importance"] = os.path.join(output_dir, "feature_importance.png")
    plot_feature_importance(model, images["feat_importance"])
    p_raw = model.predict(X_val)
    p_cal = iso_cal.predict(p_raw)
    images["calibration"] = os.path.join(output_dir, "calibration.png")
    plot_calibration(y_val, p_raw, p_cal, images["calibration"])
    images["pr"] = os.path.join(output_dir, "pr_curve.png")
    images["roc"] = os.path.join(output_dir, "roc_curve.png")
    plot_pr_roc(y_val, p_cal, images["pr"], images["roc"])
    hours_by_user = {uid: v.get("totalHours", 0.0) for uid, v in report.get("usersSummary", {}).items()}
    images["workload"] = os.path.join(output_dir, "workload.png")
    plot_hours_per_user(hours_by_user, images["workload"])
    images["topk"] = []
    shifts_with_top = [s for s in report.get("shifts", []) if s.get("topCandidates")]
    for i, s in enumerate(shifts_with_top[:sample_topk_shifts]):
        sid = s["shiftId"]
        topk_img_path = os.path.join(output_dir, f"topk_{sid}.png")
        plot_topk_for_shift(sid, s["topCandidates"], topk_img_path, k=5)
        images["topk"].append(topk_img_path)
    summary = {
        "target_shifts_total": target_shifts_count,
        "assigned_shifts": sum(1 for s in report.get("shifts", []) if s.get("chosen")),
        "unassigned_shifts": sum(1 for s in report.get("shifts", []) if not s.get("chosen")),
        "users_with_assignments": sum(1 for u, v in report.get("usersSummary", {}).items() if v.get("assignedCount", 0) > 0)
    }
    assignment_report_path = os.path.join(output_dir, "assignment_report.json")
    with open(assignment_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    html_path = write_html_report(output_dir, run_title="Shift Planner ML – Report", summary=summary,
                                  images=images, assignment_report_path=assignment_report_path)
    return {"images": images, "html": html_path, "summary": summary}

def to_dt(s: str) -> datetime:
    return pd.to_datetime(s).to_pydatetime()

def fmt_hm(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def month_key(d: date) -> str:
    return d.strftime("%Y-%m")

def daterange(start_date: date, end_date: date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)

def first_day_of_month(d: date) -> date:
    return date(d.year, d.month, 1)

def last_day_of_month(d: date) -> date:
    nd = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return nd - timedelta(days=1)

def build_employee_maps(employees):
    id2name = {}
    for e in employees:
        id2name[e["uuid"]] = e.get("name") or e["uuid"]
    # stable color per employee
    emp_ids = sorted(id2name.keys())
    id2color = {uid: EMP_COLORS[i % len(EMP_COLORS)] for i, uid in enumerate(emp_ids)}
    id2color[None] = UNASSIGNED_COLOR
    return id2name, id2color

def load_output(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_shift_records(shifts, assignments, id2name):
    """
    Build a DataFrame with shift records used for visualization.
    No holiday propagation here; holidays are read directly from JSON public_holidays.
    """
    # index assignments by shift_id
    asg_by_shift = {}
    for a in assignments or []:
        sid = int(a["shift_id"])
        asg_by_shift[sid] = {
            "employee_uuid": a["employee_uuid"],
            "source": a.get("source")
        }
    rows = []
    for sh in shifts or []:
        sid = int(sh["id"])
        start = to_dt(sh["start_date_time"])
        end = to_dt(sh["end_date_time"])
        asg = asg_by_shift.get(sid, {})
        emp = asg.get("employee_uuid")
        rows.append({
            "shift_id": sid,
            "unit": str(sh.get("workplace_id")),
            "shiftType": str(sh.get("shift_card_id") or "GEN"),
            "start": start,
            "end": end,
            "date": start.date(),
            "weekday": start.weekday(),  # Monday=0
            "assigned": emp is not None,
            "employee_uuid": emp,
            "employee_name": id2name.get(emp, "Unassigned")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["start", "unit"]).reset_index(drop=True)
    return df

def group_by_month(df):
    # returns OrderedDict of {YYYY-MM: df_month}
    if df.empty:
        return OrderedDict()
    df = df.copy()
    df["month"] = df["date"].apply(month_key)
    keys = sorted(df["month"].unique())
    return OrderedDict((k, df[df["month"] == k].copy()) for k in keys)

def month_calendar_grid(dt_month: date):
    # Build a 6x7 grid covering the month with Monday as first column (0)
    first = first_day_of_month(dt_month)
    last = last_day_of_month(dt_month)
    # start from Monday on/before first
    start = first - timedelta(days=(first.weekday() - 0) % 7)
    # end at Sunday on/after last
    end = last + timedelta(days=(6 - last.weekday()) % 7)
    # ensure 6 weeks displayed
    days = list(daterange(start, end))
    while len(days) < 42:
        end += timedelta(days=7)
        days = list(daterange(start, end))
    return days[:42], start, end

def draw_month_calendar(ax, month_date: date, day_to_items, id2color, title, holidays_set=None):
    """
    Draws a month calendar. If holidays_set contains a date, that day's cell gets a purple border.
    """
    holidays_set = holidays_set or set()

    ax.axis("off")
    days, grid_start, grid_end = month_calendar_grid(month_date)
    rows, cols = 6, 7

    # Base grid (light gray)
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((c, rows-1-r), 1, 1, fill=False, edgecolor="#999999", linewidth=1.0))

    # Weekday headers
    weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for c, wd in enumerate(weekdays):
        ax.text(c+0.02, rows+0.2, wd, fontsize=10, ha="left", va="bottom", transform=ax.transData)

    # Month title
    ax.text(0, rows+0.6, title, fontsize=14, weight="bold", ha="left", va="bottom", transform=ax.transData)

    # Cells
    for idx, d in enumerate(days):
        r = idx // cols
        c = idx % cols
        x0, y0 = c, rows-1-r

        # Overlay purple border for holidays
        if d in holidays_set:
            ax.add_patch(plt.Rectangle((x0, y0), 1, 1, fill=False, edgecolor=HOLIDAY_BORDER_COLOR, linewidth=2.0))

        # Day number
        in_month = (d.month == month_date.month and d.year == month_date.year)
        day_color = "black" if in_month else "#AAAAAA"
        ax.text(x0+0.03, y0+0.85, str(d.day), fontsize=9, color=day_color, ha="left", va="top", transform=ax.transData)

        items = day_to_items.get(d, [])
        if not items:
            continue

        # Sort items by start time
        items = sorted(items, key=lambda it: it["start"])
        usable_height = 0.80
        max_blocks = max(1, len(items))
        block_h = min(0.18, usable_height / max_blocks)
        y_cursor = y0 + 0.80 - block_h

        for it in items:
            color = id2color.get(it["employee_uuid"], id2color[None])
            ax.add_patch(plt.Rectangle((x0+0.02, y_cursor), 0.96, block_h, color=color, alpha=0.9, ec="black", lw=0.3))
            label = f"{it['shiftType']} · U{it['unit']} · {fmt_hm(it['start'])}-{fmt_hm(it['end'])}"
            if it["assigned"]:
                label += f" · {it['employee_name']}"
            else:
                label += " · Unassigned"
            ax.text(x0+0.04, y_cursor + block_h/2, label, fontsize=7, color="white",
                    ha="left", va="center", transform=ax.transData)
            y_cursor -= (block_h + 0.01)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows+1)

def build_day_map(df):
    # Group items by date
    day_map = defaultdict(list)
    for _, row in df.iterrows():
        item = {
            "unit": row["unit"],
            "shiftType": row["shiftType"],
            "start": row["start"],
            "end": row["end"],
            "assigned": bool(row["assigned"]),
            "employee_uuid": row.get("employee_uuid"),
            "employee_name": row.get("employee_name", "Unassigned")
        }
        day_map[row["date"]].append(item)
    return day_map

def parse_holiday_days(holiday_list):
    """
    From JSON 'public_holidays' entries, parse and return a set of Python date objects.
    """
    days = set()
    for h in holiday_list or []:
        dstr = h.get("date")
        if not dstr:
            # Some feeds might not have 'date' but have 'start'/'end'; fallback to 'start'
            dstr = h.get("start")
        if not dstr:
            continue
        try:
            day = pd.to_datetime(dstr).date()
            days.add(day)
        except Exception:
            continue
    return days

def render_calendars_for_plan(plan_name, df, id2color, out_dir, holidays_set=None):
    if df.empty:
        return []
    holidays_set = holidays_set or set()
    months = group_by_month(df)
    outputs = []
    for mk, mdf in months.items():
        year, mon = map(int, mk.split("-"))
        month_date = date(year, mon, 1)
        day_map = build_day_map(mdf)

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        draw_month_calendar(ax, month_date, day_map, id2color,
                            title=f"{plan_name} – {month_date.strftime('%B %Y')}",
                            holidays_set=holidays_set)
        fname = f"{plan_name.replace(' ', '_').lower()}_{mk}.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)
        outputs.append((f"{plan_name} {month_date.strftime('%B %Y')}", out_path))
    return outputs

def write_index(pairs, out_dir):
    # Include <base> here so the links work if served via HTTP
    base_href = Path(out_dir).resolve().as_uri().rstrip("/") + "/"
    lines = []
    lines.append("<html><head><meta charset='utf-8'>"
                 f"<base href='{base_href}'>"
                 "<title>Shift Calendars</title></head><body>")
    lines.append("<h1>Shift Calendars</h1>")
    lines.append("<ul>")
    for title, path in pairs:
        lines.append(f"<li><a href='{os.path.basename(path)}'>{title}</a></li>")
    lines.append("</ul>")
    lines.append("</body></html>")
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

def visualize_plans():
    ensure_dir(OUT_DIR)

    # Clean the requested folders before generating new visualizations
    empty_dir(VIZ_OUT_DIR)
    empty_dir(CALENDER_OUT_DIR)

    data = load_output(INPUT_JSON)

    # Target plan
    sp = data.get("shift_plan", {}) or {}
    employees = sp.get("employees", []) or []
    id2name, id2color = build_employee_maps(employees)

    # Build target DF
    target_df = build_shift_records(sp.get("shifts", []) or [], sp.get("shift_assignments", []) or [], id2name)

    # Holidays from target plan JSON
    target_holidays = parse_holiday_days(sp.get("public_holidays") or [])

    # Past plans (each rendered separately with their own holidays)
    past = data.get("past_shift_plans", []) or []
    past_outputs = []
    for i, plan in enumerate(past):
        plan_label = f"Past Plan {plan.get('id', i+1)}"
        hist_df = build_shift_records(plan.get("shifts", []) or [], plan.get("shift_assignments", []) or [], id2name)
        plan_holidays = parse_holiday_days(plan.get("public_holidays") or [])
        past_outputs += render_calendars_for_plan(plan_label, hist_df, id2color, OUT_DIR, holidays_set=plan_holidays)

    # Target calendars by month (with target holidays)
    target_outputs = render_calendars_for_plan("Target Plan", target_df, id2color, OUT_DIR, holidays_set=target_holidays)

    all_outputs = target_outputs + past_outputs
    idx = write_index(all_outputs, OUT_DIR)
    print(f"Rendered {len(all_outputs)} calendar images.")
    print(f"Open: {idx}")

if __name__ == "__main__":
    visualize_plans()
