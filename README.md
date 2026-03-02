# Collaborative Filtering Movie Recommendation System

A movie recommendation app built with collaborative filtering and Streamlit. This version includes a lightweight **A/B testing workflow** inspired by Netflix/YouTube experimentation: deterministic user bucketing, event logging, and experiment analytics.

## Features

- User-user collaborative filtering recommendations
- Treatment ranking variant that blends CF + popularity priors
- Deterministic A/B assignment (`control` vs `treatment`)
- Event logging for impressions, clicks, and likes
- In-app experiment analytics table (CTR and like rate)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the app at `http://localhost:8501`.

## A/B Testing Flow

1. Select a user.
2. The app deterministically assigns that user to a variant.
3. Recommendations are shown from that variant’s strategy.
4. Click **Log impressions** once per recommendation view.
5. Use **Clicked** / **Liked** buttons to log engagement events.
6. View aggregate metrics in the **Experiment analytics** section.

Event logs are stored in `logs/events.jsonl`.

## Files

- `app.py`: Streamlit UI and event instrumentation
- `recommender.py`: candidate generation + ranking + variant-based recommendation
- `experiments.py`: deterministic hash-based assignment
- `analytics.py`: event aggregation and experiment metrics

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud and create a new app from the repo.
3. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10+ recommended
4. Deploy.

### Deployment note

`logs/events.jsonl` is local ephemeral storage on hosted instances. For production-like experimentation, replace file logging with a durable store (Postgres, BigQuery, S3, etc.).
