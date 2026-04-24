import os
import re
import io
import base64
import logging

import praw
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from urllib.parse import unquote
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Clients (env vars injected by Render) ────────────────────────────────────
reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent=os.environ.get("REDDIT_USER_AGENT", "sentivityb2c"),
    check_for_async=False,
)
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
analyzer = SentimentIntensityAnalyzer()


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search_reddit(query, limit, sort, time_filter):
    rows = []
    for post in reddit.subreddit("all").search(
        query, sort=sort, time_filter=time_filter, limit=limit
    ):
        rows.append(
            {
                "id": post.id,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "created_utc": post.created_utc,
                "full_text": clean_text(post.title + " " + post.selftext),
            }
        )
    return pd.DataFrame(rows)


def optimized_collection(target, competitors):
    results = {}
    logger.info(f"Deep diving target: {target}")
    results[target] = search_reddit(target, limit=500, sort="hot", time_filter="month")
    for company in competitors:
        logger.info(f"Quick scan competitor: {company}")
        results[company] = search_reddit(
            company, limit=150, sort="hot", time_filter="month"
        )
    return results


def compute_weighted_sentiment(texts, scores):
    if not texts:
        return 0.0
    sentiments = [analyzer.polarity_scores(t)["compound"] for t in texts]
    scores = np.array(scores, dtype=float)
    return (
        float(np.dot(sentiments, scores) / scores.sum())
        if scores.sum() != 0
        else float(np.mean(sentiments))
    )


def impute_missing_days(df):
    out = []
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    for company, g in tmp.groupby("company", sort=False):
        g = g.sort_values("date").set_index("date")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g2 = g.reindex(full_idx)
        g2["company"] = company
        g2["sentiment_score"] = (
            g2["sentiment_score"]
            .interpolate(method="time", limit_area="inside")
            .ffill()
            .bfill()
        )
        g2 = g2.reset_index().rename(columns={"index": "date"})
        out.append(g2)
    result = pd.concat(out, ignore_index=True)
    result["date"] = result["date"].dt.date
    return result


def trend_stats(g):
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < 2:
        return None
    end = g["date"].iloc[-1]
    start = end - pd.Timedelta(days=35)
    h = g[g["date"] >= start].copy()
    if len(h) < 2:
        return None
    last = float(h["smoothed"].iloc[-1])
    prev_7 = h["date"] <= (end - pd.Timedelta(days=7))
    delta_7 = last - float(h.loc[prev_7, "smoothed"].iloc[-1]) if prev_7.any() else float("nan")
    prev_30 = h["date"] <= (end - pd.Timedelta(days=30))
    delta_30 = (
        last - float(h.loc[prev_30, "smoothed"].iloc[-1])
        if prev_30.any()
        else last - float(h["smoothed"].iloc[0])
    )
    diff_series = h["smoothed"].diff()
    biggest_idx = diff_series.abs().idxmax()
    biggest_jump = float(diff_series.iloc[biggest_idx]) if pd.notna(diff_series.iloc[biggest_idx]) else 0.0
    biggest_date = h.loc[biggest_idx, "date"].date().isoformat()
    return {
        "last_smoothed": round(last, 4),
        "delta_7": round(delta_7, 4) if not np.isnan(delta_7) else None,
        "delta_30": round(delta_30, 4),
        "biggest_jump_date": biggest_date,
        "biggest_jump": round(biggest_jump, 4),
    }


def build_plot_base64(df_plot, target, all_companies):
    df_plot = df_plot.copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])
    df_plot = df_plot.sort_values(["company", "date"])
    df_plot["smoothed_sentiment"] = df_plot.groupby("company")["sentiment_score"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for company, group in df_plot.groupby("company"):
        lw = 2.5 if company == target else 1.5
        ax.plot(group["date"], group["smoothed_sentiment"], label=company, linewidth=lw)

    ax.set_xlabel("Date")
    ax.set_ylabel("Smoothed Sentiment Score (-1 to +1)")
    ax.set_title(f"Public Sentiment Trend – {target} vs Competitors")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def pick_examples(company, exploded_df, is_target):
    k = 10 if is_target else 5
    subset = exploded_df[exploded_df["company"] == company].copy()
    if subset.empty:
        return []
    combined = pd.concat(
        [subset.nlargest(k, "score"), subset.nlargest(k, "sentiment"), subset.nsmallest(k, "sentiment")]
    ).drop_duplicates("id")
    return combined.to_dict("records")


def format_stats(stats_dict, target, competitors):
    lines = []
    for c in [target] + competitors:
        s = stats_dict.get(c)
        if s:
            lines.append(
                f"{c}: current={s['last_smoothed']:.3f}, "
                f"7d={s.get('delta_7') or 0:+.3f}, "
                f"30d={s['delta_30']:+.3f}, "
                f"biggest spike {s['biggest_jump']:+.3f} on {s['biggest_jump_date']}"
            )
    return "\n".join(lines)


def format_examples(examples_dict):
    output = []
    for company, posts in examples_dict.items():
        output.append(f"--- {company} ---")
        for p in posts:
            s = p.get("sentiment", 0)
            label = "POS" if s > 0.1 else "NEG" if s < -0.1 else "ENG"
            text = str(p.get("full_text", ""))[:400].replace("\n", " ").strip()
            output.append(f"[{label}] {text}")
    return "\n".join(output)


def generate_report(target, competitors, stats_dict, examples_dict):
    prompt = f"""
You are a forensic industry analyst. Your goal is to find the "Ground Truth" behind the sentiment spikes.

DATA KEY:
- [ENG]: High-engagement (Top scores)
- [POS]: High positive sentiment
- [NEG]: High negative sentiment

TREND STATS:
{format_stats(stats_dict, target, competitors)}

REPRESENTATIVE POSTS:
{format_examples(examples_dict)}

Write EXACTLY this output format. No direct quotes.

INDUSTRY OUTLOOK
[2-3 short sentences. Identify the specific external event driving the current market momentum.]

TARGET BRAND ANALYSIS: {target}
• {target} POSITIVE DRIVERS: [Identify the specific event, location, or product feature mentioned.]
• {target} FRICTION POINTS: [Identify the specific grievance with concrete detail.]
• KEY SIGNAL: One sentence connecting a specific [NEG] thread to a long-term brand risk.
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()


@app.route("/<path:company_path>", methods=["GET"])
def sentiment_report(company_path):
    """
    GET /target_company/competitor1/competitor2/competitor3/
    
    Handles both URL-encoded spaces (%20) and plus-encoded spaces (+).
    
    Examples:
      https://sentimentreport.onrender.com/Apple/Microsoft/Google/Samsung/
      https://sentimentreport.onrender.com/Tesla%20Inc/Ford%20Motor/General%20Motors/
      https://sentimentreport.onrender.com/Tesla Inc/Ford Motor/General Motors/
    """
    # Split by '/' and decode each company name
    parts = [unquote(p).replace('+', ' ').strip() for p in company_path.split('/') if p.strip()]
    
    if len(parts) < 3:
        return jsonify({"error": "URL format: /target/competitor1/competitor2/... (minimum 3 total)"}), 400
    
    target = parts[0]
    competitors = parts[1:]

    if not (2 <= len(competitors) <= 5):
        return jsonify({"error": f"competitors must be 2–5 companies (you provided {len(competitors)})"}), 400

    logger.info(f"Starting report: target={target}, competitors={competitors}")
 
    # 1. Collect data
    company_dfs = optimized_collection(target, competitors)
 
    combined_df = pd.concat(
        [
            df.assign(company=company)[["company", "id", "score", "full_text", "created_utc"]]
            for company, df in company_dfs.items()
            if not df.empty
        ],
        ignore_index=True,
    )
 
    if combined_df.empty:
        return jsonify({"error": "No data retrieved from Reddit"}), 502
 
    combined_df["date"] = pd.to_datetime(combined_df["created_utc"], unit="s").dt.date
 
    # 2. Daily weighted sentiment
    daily_grouped = (
        combined_df.groupby(["company", "date"])
        .agg({"id": list, "score": list, "full_text": list})
        .reset_index()
    )
    daily_grouped["sentiment_score"] = daily_grouped.apply(
        lambda row: compute_weighted_sentiment(row["full_text"], row["score"]), axis=1
    )
 
    daily_filled = impute_missing_days(daily_grouped[["company", "date", "sentiment_score"]])
 
    # 3. Build plot
    plot_b64 = build_plot_base64(daily_filled, target, [target] + competitors)
 
    # 4. Trend stats
    df_ts = daily_filled.copy()
    df_ts["date"] = pd.to_datetime(df_ts["date"])
    df_ts["smoothed"] = df_ts.groupby("company")["sentiment_score"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    stats_dict = {
        company: trend_stats(group) for company, group in df_ts.groupby("company")
    }
 
    # 5. Pick examples + enrich with top comments
    exploded_df = daily_grouped.explode(["id", "score", "full_text"]).copy()
    exploded_df = exploded_df.rename(columns={"sentiment_score": "sentiment"})
    exploded_df["score"] = pd.to_numeric(exploded_df["score"], errors="coerce")
    exploded_df["sentiment"] = pd.to_numeric(exploded_df["sentiment"], errors="coerce")
    exploded_df = exploded_df.dropna(subset=["score", "sentiment"])
 
    all_companies = [target] + competitors
    examples = {c: pick_examples(c, exploded_df, c == target) for c in all_companies}
 
    # Batch fetch top comments
    all_ids = [f"t3_{p['id']}" for posts in examples.values() for p in posts]
    if all_ids:
        submissions = list(reddit.info(fullnames=all_ids))
        sub_map = {s.id: s for s in submissions}
        for company in examples:
            for post in examples[company]:
                sub = sub_map.get(post["id"])
                if sub:
                    sub.comment_sort = "top"
                    sub.comments.replace_more(limit=0)
                    if sub.comments:
                        post["full_text"] += f" [TOP COMMENT]: {sub.comments[0].body[:200]}"
 
    # 6. GPT report
    report_text = generate_report(target, competitors, stats_dict, examples)
 
    # 7. Build daily sentiment table for response
    sentiment_table = (
        daily_filled.groupby("company")
        .apply(lambda g: g.set_index("date")["sentiment_score"].to_dict())
        .to_dict()
    )
 
    return jsonify(
        {
            "target": target,
            "competitors": competitors,
            "trend_stats": stats_dict,
            "report": report_text,
            "sentiment_by_company": {
                c: {str(k): round(v, 4) for k, v in vals.items()}
                for c, vals in sentiment_table.items()
            },
            "chart_png_base64": plot_b64,
        }
    )
 
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
