import git
import streamlit as st
import dvc.api
import pandas as pd


REPO = git.Repo(".")
MODELS_COMMITS = list(REPO.iter_commits(paths="dvc.lock"))

selected_commit = st.selectbox(
    "Choose your commit",
    [commit for commit in MODELS_COMMITS],
    format_func=lambda commit: f"{commit.hexsha[:6]} - {commit.message} - {commit.committed_datetime}",
)

st.write("Selected Commit", selected_commit)


@st.cache
def load_predictions(rev: str) -> pd.DataFrame:
    with dvc.api.open("data/predictions.csv", rev=rev) as f:
        return pd.read_csv(f)


predictions = load_predictions(rev=selected_commit.hexsha)

st.dataframe(predictions)

