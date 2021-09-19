import git
import streamlit as st

REPO = git.Repo(".")
MODELS_COMMITS = list(REPO.iter_commits(paths="dvc.lock"))

selected_commit = st.selectbox(
    "Choose your commit",
    [commit for commit in MODELS_COMMITS],
    format_func=lambda commit: f"{commit.hexsha[:6]} - {commit.message} - {commit.committed_datetime}",
)

st.write("Selected Commit", selected_commit)
