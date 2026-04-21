import requests
import time
import pandas as pd
import numpy as np

#Jikan API url
base_url = "https://api.jikan.moe/v4/"


def get_anime_relations(mal_id):
    url = f"{base_url}anime/{mal_id}/relations"
    response = requests.get(url)
    
    if response.status_code == 429:
        time.sleep(3)
        response = requests.get(url)
    
    time.sleep(0.4)
    data = response.json()
    
    for relation in data.get("data", []):
        if relation["relation"] in ["Prequel", "Parent story"]:
            return 1
    return 0


def get_season_info(season, year):

    url = f"{base_url}seasons/{year}/{season}"
    filter = {"filter": "tv", "sfw": "true"}
    response = requests.get(url, params=filter)

    if response.status_code == 429:
        print("Rate limited! Sleeping for 3 seconds...")
        time.sleep(3)
        response = requests.get(url, params=filter)

    total_pages = response.json()["pagination"]["last_visible_page"]

    mal_ids = []
    image_urls = []
    titles = []
    synopsis = []
    sources = []
    genres = []
    demographics = []
    is_franchise = []
    members = []
    episodes = []
    year_lst = []
    season_lst = []
    producer_id = []
    studio_id = []
    scores = []

    for curr_page in range(1, total_pages + 1):
        print(f"Fetching page {curr_page} of {total_pages}")
        filter["page"] = curr_page

        response = requests.get(url, params=filter)

        if response.status_code == 429:
            print("Rate limited! Sleeping for 3 seconds...")
            time.sleep(3)
            response = requests.get(url, params=filter)

        data = response.json()

        for anime in data["data"]:
            mal_ids.append(anime["mal_id"])
            image_urls.append(anime["images"]["webp"]["image_url"])
            titles.append(anime["title"])
            synopsis.append(anime["synopsis"] if anime["synopsis"] is not None else "")
            sources.append(anime["source"])
            genres.append([genre["name"] for genre in anime["genres"]])
            demographics.append([d["name"] for d in anime["demographics"]])
            is_franchise.append(get_anime_relations(anime["mal_id"]))
            members.append(anime["members"])
            episodes.append(anime["episodes"] if anime["episodes"] is not None else 0)
            year_lst.append(year)
            season_lst.append(season)
            producer_id.append([producer["mal_id"] for producer in anime["producers"]])
            studio_id.append([studio["mal_id"] for studio in anime["studios"]])
            scores.append(anime["score"])

    info = {"mal_id": mal_ids,
            "image_url": image_urls,
            "title": titles,
            "synopsis": synopsis,
            "source": sources,
            "genres": genres,
            "demographics": demographics,
            "is_franchise": is_franchise,
            "members": members,
            "episodes": episodes,
            "year": year_lst,
            "season": season_lst,
            "producer_id": producer_id,
            "studio_id": studio_id,
            "score": scores}

    df = pd.DataFrame(info)
    print(f"Scraped {season}, {year}")
    return df


# ── Scraping ──────────────────────────────────────────────────────────────────

train_dfs = []
train_dfs.append(get_season_info("winter", 2018))
train_dfs.append(get_season_info("spring", 2018))
train_dfs.append(get_season_info("summer", 2018))
train_dfs.append(get_season_info("fall", 2018))
train_dfs.append(get_season_info("winter", 2019))
train_dfs.append(get_season_info("spring", 2019))
train_dfs.append(get_season_info("summer", 2019))
train_dfs.append(get_season_info("fall", 2019))
train_dfs.append(get_season_info("winter", 2020))
train_dfs.append(get_season_info("spring", 2020))
train_dfs.append(get_season_info("summer", 2020))
train_dfs.append(get_season_info("fall", 2020))
train_dfs.append(get_season_info("winter", 2021))
train_dfs.append(get_season_info("spring", 2021))
train_dfs.append(get_season_info("summer", 2021))
train_dfs.append(get_season_info("fall", 2021))
train_dfs.append(get_season_info("winter", 2022))
train_dfs.append(get_season_info("spring", 2022))
train_dfs.append(get_season_info("summer", 2022))
train_dfs.append(get_season_info("fall", 2022))
train_dfs.append(get_season_info("winter", 2023))
train_dfs.append(get_season_info("spring", 2023))
train_dfs.append(get_season_info("summer", 2023))
train_dfs.append(get_season_info("fall", 2023))
train_dfs.append(get_season_info("winter", 2024))
train_dfs.append(get_season_info("spring", 2024))
train_dfs.append(get_season_info("summer", 2024))
train_dfs.append(get_season_info("fall", 2024))
train_dfs.append(get_season_info("winter", 2025))
train_dfs.append(get_season_info("spring", 2025))

summer_2025_df = get_season_info("summer", 2025)
fall_2025_df = get_season_info("fall", 2025)


# ── Raw data ─────────────────────────────────────────────────────────────────

train_df = pd.concat(train_dfs, ignore_index=True)
train_df = train_df.drop_duplicates(subset=["mal_id"])
train_df.to_csv("train_raw.csv", index=False)
train_df = train_df.dropna(subset=['score'])
train_df = train_df.reset_index(drop=True)

summer_2025_df = summer_2025_df.drop_duplicates(subset=["mal_id"])
summer_2025_df.to_csv("summer25_raw.csv", index=False)
summer_2025_df = summer_2025_df.dropna(subset=['score'])
summer_2025_df = summer_2025_df.reset_index(drop=True)

fall_2025_df = fall_2025_df.drop_duplicates(subset=["mal_id"])
fall_2025_df.to_csv("fall25_raw.csv", index=False)
fall_2025_df = fall_2025_df.dropna(subset=['score'])
fall_2025_df = fall_2025_df.reset_index(drop=True)

print(f"Train: {len(train_df)} anime")
print(f"Summer 2025: {len(summer_2025_df)} anime")
print(f"Fall 2025: {len(fall_2025_df)} anime")


# ── Cleaning data function ───────────────────────────────────────────────────

def clean_df(df, source_map, all_genres, zero_var_genres, all_demographics,
             zero_var_demos, train_season_cols, reputation_maps, is_train=False):

    # Source dummies
    df['source_new'] = df['source'].map(source_map).fillna("Other")
    source_dummies = pd.get_dummies(df['source_new'], prefix='source', drop_first=True, dtype=int)
    df = pd.concat([df, source_dummies], axis=1)

    # Genre dummies
    for genre in all_genres:
        df[f"genre_{genre}"] = df['genres'].apply(lambda lst: 1 if genre in lst else 0)
    df = df.drop(columns=zero_var_genres)

    # Demographic dummies
    for demo in all_demographics:
        df[f"demo_{demo}"] = df['demographics'].apply(lambda lst: 1 if demo in lst else 0)
    df = df.drop(columns=zero_var_demos)

    # Season dummies
    season_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=True, dtype=int)
    df = pd.concat([df, season_dummies], axis=1)

    # Reputation features — from training maps
    studio_quality_map, producer_quality_map, studio_hype_map, studio_experience_map, train_score_mean, train_members_mean = reputation_maps
    df['studio_rep_quality'] = df['studio_id'].apply(lambda x: get_avg_reputation(x, studio_quality_map, train_score_mean))
    df['producer_rep_quality'] = df['producer_id'].apply(lambda x: get_avg_reputation(x, producer_quality_map, train_score_mean))
    df['studio_rep_hype'] = df['studio_id'].apply(lambda x: get_avg_reputation(x, studio_hype_map, train_members_mean))
    df['studio_rep_hype'] = np.log1p(df['studio_rep_hype'])
    df['studio_experience'] = df['studio_id'].apply(lambda x: get_avg_reputation(x, studio_experience_map, 0))

    # Episodes log
    df['episodes_log'] = np.log1p(df['episodes'])

    # Drop raw columns
    df = df.drop(columns=['source', 'source_new', 'genres', 'demographics',
                           'members', 'episodes', 'season', 'studio_id', 'producer_id'])

    # Align to training columns for test sets
    if not is_train:
        df = df.reindex(columns=train_season_cols, fill_value=0)

    return df


# ── Fit on training data ─────────────────────────────────────────────────────

source_map = {"Manga": "Manga", "Web manga": "Manga", "4-koma manga": "Manga",
              "Light novel": "LN", "Novel": "LN",
              "Original": "Original",
              "Visual novel": "VN",
              "Game": "Game"}

all_genres = ["Action", "Adventure", "AvantGarde", "AwardWinning", "BL", "Comedy", "Drama",
              "Fantasy", "GL", "Gourmet", "Horror", "Mystery", "Romance", "Sci-Fi",
              "SliceOfLife", "Sports", "Supernatural", "Suspense"]

all_demographics = ["Shounen", "Seinen", "Shoujo", "Josei", "Kids"]

# Determine zero-variance columns from training data
genre_cols = [f"genre_{g}" for g in all_genres]
demo_cols = [f"demo_{d}" for d in all_demographics]

temp_genre = pd.DataFrame({c: train_df['genres'].apply(lambda lst: 1 if c.replace('genre_', '') in lst else 0) for c in genre_cols})
temp_demo = pd.DataFrame({c: train_df['demographics'].apply(lambda lst: 1 if c.replace('demo_', '') in lst else 0) for c in demo_cols})
zero_var_genres = [c for c in genre_cols if temp_genre[c].sum() == 0]
zero_var_demos = [c for c in demo_cols if temp_demo[c].sum() == 0]

# Compute reputation maps from training data
def calculate_reputation(df, id_column, target_column):
    exploded_df = df.explode(id_column)
    exploded_df[id_column] = pd.to_numeric(exploded_df[id_column], errors='coerce')
    exploded_df = exploded_df.dropna(subset=[id_column])
    return exploded_df.groupby(id_column)[target_column].mean().to_dict()

def calculate_experience(df, id_column):
    exploded_df = df.explode(id_column)
    exploded_df[id_column] = pd.to_numeric(exploded_df[id_column], errors='coerce')
    exploded_df = exploded_df.dropna(subset=[id_column])
    return exploded_df.groupby(id_column)[id_column].count().to_dict()

def get_avg_reputation(id_list, rep_map, default_avg):
    if not id_list or not isinstance(id_list, list):
        return default_avg
    scores = [rep_map.get(idx) for idx in id_list if idx in rep_map]
    return sum(scores) / len(scores) if scores else default_avg

studio_quality_map = calculate_reputation(train_df, 'studio_id', 'score')
producer_quality_map = calculate_reputation(train_df, 'producer_id', 'score')
studio_hype_map = calculate_reputation(train_df, 'studio_id', 'members')
studio_experience_map = calculate_experience(train_df, 'studio_id')
train_score_mean = train_df['score'].mean()
train_members_mean = train_df['members'].mean()

reputation_maps = (studio_quality_map, producer_quality_map, studio_hype_map,
                   studio_experience_map, train_score_mean, train_members_mean)


# ── Clean data ───────────────────────────────────────────────────────────────

train_df = clean_df(train_df, source_map, all_genres, zero_var_genres,
                    all_demographics, zero_var_demos,
                    train_season_cols=None, reputation_maps=reputation_maps,
                    is_train=True)

# Capture final training columns to align test sets
train_season_cols = train_df.columns.tolist()

summer_2025_df = clean_df(summer_2025_df, source_map, all_genres, zero_var_genres,
                           all_demographics, zero_var_demos,
                           train_season_cols=train_season_cols,
                           reputation_maps=reputation_maps, is_train=False)

fall_2025_df = clean_df(fall_2025_df, source_map, all_genres, zero_var_genres,
                         all_demographics, zero_var_demos,
                         train_season_cols=train_season_cols,
                         reputation_maps=reputation_maps, is_train=False)


# ── Processed files ───────────────────────────────────────────────────────────

train_df.to_csv("train_processed.csv", index=False)
summer_2025_df.to_csv("summer25_processed.csv", index=False)
fall_2025_df.to_csv("fall25_processed.csv", index=False)

print("Training Dataframe Complete:")
print(f"Total Anime in Training Data: {len(train_df)}")
print(f"Summer 2025 Test: {len(summer_2025_df)} anime")
print(f"Fall 2025 Test: {len(fall_2025_df)} anime")
print(f"Feature columns: {len(train_df.columns)}")
