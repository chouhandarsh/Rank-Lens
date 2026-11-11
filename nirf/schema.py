"""Canonical field names and analytical metadata."""

PARAMETERS = ["TLR", "RPC", "GO", "OI", "PR"]
PARAMETER_LABELS = {
    "TLR": "Teaching, Learning and Resources",
    "RPC": "Research and Professional Practice",
    "GO": "Graduation Outcome",
    "OI": "Outreach and Inclusivity",
    "PR": "Perception",
}
NUMERIC_FIELDS = ["score", "rank_numeric", *[x.lower() for x in PARAMETERS]]
DISPLAY_FIELDS = {
    "score": "Overall score",
    "rank_numeric": "Rank",
    "tlr": "TLR",
    "rpc": "RPC",
    "go": "GO",
    "oi": "OI",
    "pr": "PR",
}
CATEGORY_LABELS = {
    "University": "University",
    "College": "College",
    "Research": "Research",
    "Architecture": "Architecture",
}
RANK_BANDS = ["Top 10", "11–50", "51–100", "101+", "Unranked / not numeric"]
