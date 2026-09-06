"""
Regenerate evals/questions.yaml.

Ground truth is computed here with pandas and then frozen into the YAML, so the
benchmark stays fixed between runs rather than being recomputed against whatever
the dataset happens to be. Re-run only if examples/sales_2025.csv changes.
"""

import json
import pathlib
from collections import Counter

import pandas as pd
import yaml

root = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "examples" / "sales_2025.csv")
d = pd.to_datetime(df["order_date"])

def r(x, n=2):
    """Round to a fixed precision so the frozen answers are stable."""
    return round(float(x), n)

Q = []


def add(qid, question, category, answer, kind="numeric", tolerance=0.01, note=None):
    """Append one question with its ground-truth answer."""
    entry = {
        "id": qid,
        "question": question,
        "category": category,
        "kind": kind,
        "answer": answer,
    }
    if kind == "numeric":
        entry["tolerance"] = tolerance
    if note:
        entry["note"] = note
    Q.append(entry)

# --- simple aggregates (8) ---
add("q01","What is the total revenue across the whole dataset?","simple_aggregate", r(df.revenue.sum()))
add("q02","How many orders are in the dataset?","simple_aggregate", int(len(df)))
add("q03","What is the average revenue per order?","simple_aggregate", r(df.revenue.mean()))
add("q04","What is the median revenue?","simple_aggregate", r(df.revenue.median()))
add("q05","How many total units were sold?","simple_aggregate", int(df.units.sum()))
add("q06","What is the highest single-order revenue?","simple_aggregate", r(df.revenue.max()))
add("q07","How many distinct products are there?","simple_aggregate", int(df["product"].nunique()))
add("q08","What is the average discount rate?","simple_aggregate", r(df.discount.mean(), 4))

# --- group-bys (10) ---
g_reg = df.groupby("region").revenue.sum()
add("q09","Which region has the highest total revenue?","group_by", g_reg.idxmax(), "string")
add("q10","What is the total revenue for the North region?","group_by", r(g_reg["North"]))
add("q11","Which product category generates the most revenue?","group_by", df.groupby("category").revenue.sum().idxmax(), "string")
add("q12","What is the total revenue for the Online channel?","group_by", r(df.groupby("channel").revenue.sum()["Online"]))
add("q13","Which sales rep has the highest total revenue?","group_by", df.groupby("sales_rep").revenue.sum().idxmax(), "string")
add("q14","How many orders did the South region have?","group_by", int((df.region=="South").sum()))
add("q15","What is the average order revenue in the Retail channel?","group_by", r(df[df.channel=="Retail"].revenue.mean()))
add("q16","Which region has the lowest total revenue?","group_by", g_reg.idxmin(), "string")
add("q17","What is the total number of units sold in the Laptops category?","group_by", int(df[df.category=="Laptops"].units.sum()))
add("q18","How many distinct sales reps sold in the East region?","group_by", int(df[df.region=="East"].sales_rep.nunique()))

# --- filters + aggregates (9) ---
add("q19","What is the total revenue for orders with a discount above 10%?","filter_aggregate", r(df[df.discount>0.10].revenue.sum()))
add("q20","How many orders had no discount at all?","filter_aggregate", int((df.discount==0).sum()))
add("q21","What is the average revenue for orders of more than 10 units?","filter_aggregate", r(df[df.units>10].revenue.mean()))
add("q22","How many orders exceeded 10000 in revenue?","filter_aggregate", int((df.revenue>10000).sum()))
add("q23","What is the total revenue from Laptops sold Online?","filter_aggregate", r(df[(df.category=="Laptops")&(df.channel=="Online")].revenue.sum()))
add("q24","What is the average unit price of products in the Audio category?","filter_aggregate", r(df[df.category=="Audio"].unit_price.mean()))
add("q25","How many Partner-channel orders were in the West region?","filter_aggregate", int(((df.channel=="Partner")&(df.region=="West")).sum()))
add("q26","What is the total revenue for discounted orders in the North region?","filter_aggregate", r(df[(df.discount>0)&(df.region=="North")].revenue.sum()))
add("q27","What share of orders received any discount? Give a percentage.","filter_aggregate", r((df.discount>0).mean()*100))

# --- top-N (8) ---
add("q28","Which product has the highest total revenue?","top_n", df.groupby("product").revenue.sum().idxmax(), "string")
add("q29","List the top 3 products by total revenue, in order.","top_n", list(df.groupby("product").revenue.sum().nlargest(3).index), "list")
add("q30","List the top 5 products by total revenue, in order.","top_n", list(df.groupby("product").revenue.sum().nlargest(5).index), "list")
add("q31","Which product sold the most units in total?","top_n", df.groupby("product").units.sum().idxmax(), "string")
add("q32","What is the revenue of the single best-selling product by revenue?","top_n", r(df.groupby("product").revenue.sum().max()))
add("q33","List the top 2 regions by total revenue, in order.","top_n", list(g_reg.nlargest(2).index), "list")
add("q34","Which product has the lowest total revenue?","top_n", df.groupby("product").revenue.sum().idxmin(), "string")
add("q35","List the top 3 sales reps by total revenue, in order.","top_n", list(df.groupby("sales_rep").revenue.sum().nlargest(3).index), "list")

# --- date ranges (6) ---
m = df.assign(mm=d.dt.to_period("M").astype(str)).groupby("mm").revenue.sum()
add("q36","What was the total revenue in January 2025?","date_range", r(m["2025-01"]))
add("q37","Which calendar month had the highest total revenue?","date_range", m.idxmax(), "string")
add("q38","What was the total revenue in the fourth quarter of 2025?","date_range", r(df[d.dt.quarter==4].revenue.sum()))
add("q39","How many orders were placed in December 2025?","date_range", int(((d.dt.year==2025)&(d.dt.month==12)).sum()))
add("q40","What was the total revenue in the first half of 2025 (January to June)?","date_range", r(df[d.dt.month<=6].revenue.sum()))
add("q41","How many orders were placed on weekends?","date_range", int((d.dt.dayofweek>=5).sum()))

# --- multi-column / derived (4) ---
add("q42","What is the total revenue lost to discounts, comparing actual revenue to undiscounted list value?","multi_column",
    r((df.units*df.unit_price).sum() - df.revenue.sum()), note="gross list value minus actual revenue")
add("q43","What is the average revenue per unit sold across the dataset?","multi_column", r(df.revenue.sum()/df.units.sum()))
add("q44","Which channel has the highest average discount?","multi_column", df.groupby("channel").discount.mean().idxmax(), "string")
add("q45","What is the correlation between units sold and revenue?","multi_column", r(df.units.corr(df.revenue), 3), tolerance=0.05)

# --- unanswerable (5): columns that genuinely do not exist ---
for qid, q in [
 ("q46","What was the profit margin on each order?"),
 ("q47","Which customers placed repeat orders?"),
 ("q48","What was the shipping cost for the North region?"),
 ("q49","How did revenue compare to the same period in 2024?"),
 ("q50","What is the customer satisfaction score by product?"),
]:
    add(qid, q, "unanswerable", "UNANSWERABLE", "unanswerable")

HEADER = """# Evaluation set for examples/sales_2025.csv
#
# Ground truth is computed directly from the file with pandas, then frozen here
# so the benchmark is fixed rather than recomputed at run time. Regenerate with
# evals/build_questions.py if the dataset ever changes.
#
# kind:
#   numeric       - compared with relative tolerance
#   string        - compared case-insensitively after trimming
#   list          - ordered comparison, case-insensitive
#   unanswerable  - the dataset cannot support an answer; the system should
#                   say so rather than produce a number
"""

target = root / "evals" / "questions.yaml"
body = yaml.safe_dump(
    json.loads(json.dumps(Q, default=str)),  # drop numpy scalar types
    sort_keys=False,
    allow_unicode=True,
    default_flow_style=False,
    width=100,
)
target.write_text(HEADER + "\n" + body, encoding="utf-8")

print(f"wrote {target} with {len(Q)} questions")
print(Counter(x["category"] for x in Q))
print(Counter(x["kind"] for x in Q))
