# 🏏 CRICKET INTELLIGENCE
### Ball-by-Ball Player Performance & Scouting Intelligence Platform

Cricket Intelligence is an advanced, high-performance cricket scouting and performance analysis platform built on top of 200,999 individual ball-by-ball deliveries across 883 matches, 138 tournaments, and 1,217 players (covering 2022 to 2026).

---

## 🚀 Key Highlights & Architecture

- **Sub-Second OLAP Performance**: Uses Apache PyArrow Parquet caching and DuckDB in-memory querying, loading 200,000+ deliveries in ~0.06 seconds.
- **Relational Integrity**: Strict relational keys (`player_id`, `bowler_id`, `team_id`, `comp_id`, `match_id`) preserved across all datasets.
- **Official Cricket Rules**:
  - Wides excluded from balls faced by batsman.
  - No-balls (Oversteps) counted as balls faced by batter, but excluded from bowler's legal deliveries.
  - Bowler charges properly isolated (byes and legbyes are not charged to bowler).
  - Bowler credited wickets strictly exclude Run-Outs, Obstructing Field, and Retire Outs.
  - Correct handling of not-out innings for Batting Average calculation.
- **Transparent Data Integrity**: Audited encoded columns (`extras1_runs` identified as categorical code, not numeric run count). Ground spatial limitations transparently disclosed without fabricating artificial zone coordinates.

---

## 📁 Project Structure

```
cricket_intelligence/
│
├── data/
│   ├── raw/                  # Original 6 CSV files
│   └── processed/            # High-performance Parquet analytical files
│
├── src/
│   ├── data_cleaning.py      # Cleans raw data and generates Parquet cache
│   ├── data_loader.py        # Streamlit-cached data loading and ID maps
│   ├── database.py           # DuckDB in-memory OLAP query engine
│   ├── batting_analysis.py   # Batting KPIs, over-by-over, delivery inspector
│   ├── bowling_analysis.py   # Bowling KPIs, phases, handedness, delivery styles
│   ├── matchup_analysis.py   # Batsman vs Bowler matrix & 1-on-1 duel analyzer
│   ├── shot_analysis.py      # Stroke productivity, pace vs spin, delivery styles
│   ├── wagon_wheel.py        # Cricket ground geometry, 17 zones, vector simulator
│   ├── team_analysis.py      # Team records, phase run rates, squad contributions
│   ├── match_analysis.py     # Playing XI, scorecards, fall of wickets, partnerships
│   └── ui_components.py      # Dark sports analytics theme and reusable cards
│
├── pages/
│   ├── 1_Player_Database.py       # Searchable player scouting directory
│   ├── 2_Batsman_Analysis.py      # Detailed batsman profile & delivery inspector
│   ├── 3_Bowler_Analysis.py       # Dedicated bowler profile & phase spells
│   ├── 4_Batsman_vs_Bowler.py     # Head-to-head matchup matrix & duels
│   ├── 5_Shot_Analysis.py         # Stroke productivity & delivery styles
│   ├── 6_Wagon_Wheel.py           # Interactive cricket ground & 17 zones
│   ├── 7_Pace_vs_Spin.py          # Dedicated Pace vs Spin comparative matrix
│   ├── 8_Team_Analysis.py         # Team standings and phase efficiency
│   ├── 9_Match_Explorer.py        # Full match scorecards & ball logs
│   ├── 10_Competition_Analysis.py # Tournament leaderboards & records
│   ├── 11_Player_Comparison.py    # Multi-player radar & KPI benchmark
│   └── 12_Data_Quality.py         # Data audit, formulas, and schema report
│
├── app.py                     # Main dashboard & executive overview
├── requirements.txt           # Project dependencies
└── README.md                  # Documentation and deployment guide
```

---

## 🛠️ Installation & Local Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.13, 3.14 supported)
- pip package manager

### 2. Clone / Navigate to Directory
```bash
cd cricket_intelligence
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Run Data Cleaning Pipeline
*The analytical Parquet cache is already pre-generated. To re-process from raw CSVs:*
```bash
python src/data_cleaning.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Online Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended & Free)
1. Push this project repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py` (or `cricket_intelligence/app.py`).
5. Click **"Deploy"**. Streamlit Cloud installs `requirements.txt` and hosts the live website automatically.

### Option 2: Docker Container
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t cricket-intelligence .
docker run -p 8501:8501 cricket-intelligence
```

### Option 3: Hugging Face Spaces / Render / AWS ECS
- On **Hugging Face Spaces**: Select **Streamlit** SDK and choose your repo.
- On **Render**: Choose **Web Service**, set Build Command to `pip install -r requirements.txt` and Start Command to `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.

---

## 🏏 Key Analytical Capabilities

1. **Individual Batsman Profile**:
   - Executive KPIs: Runs, Balls, Average, Strike Rate, Highest Score (handling not-outs), 30s, 50s, 100s, 4s, 6s, Dot %, Boundary %.
   - **Over-by-Over Table**: Inspect scoring rate per over with drill-down into every delivery.
   - **Phase Analysis**: Powerplay (1-6), Middle (7-15), Death (16-20).
   - **Ball-by-Ball Inspector**: Multi-criteria filters for bowler, pace/spin, shot, and outcome.

2. **Bowler Profile**:
   - Wickets, Legal Balls, Overs, Conceded Runs, Economy, Bowling Average, Strike Rate, Dot %.
   - Spells across Powerplay, Middle, and Death.
   - Handedness splits (vs RHB and vs LHB).
   - Delivery styles (Yorker, Bouncer, Inswing, Outswing, Leg Break, Googly, etc.).

3. **Batsman vs Bowler Matchups**:
   - Comprehensive matrix of every bowler faced.
   - Dedicated 1-on-1 duel analyzer with head-to-head ball history.

4. **Pace vs Spin**:
   - Side-by-side comparative table and charts for pace vs spin responses.

5. **Match Explorer**:
   - Reconstructed scorecards, fall of wickets, partnerships, and over worm.

6. **Wagon Wheel & Ground Visualization**:
   - Standard 2D cricket ground with 17 fielding zones and transparent vector trajectory simulation.

