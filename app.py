import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic EDA", layout="wide")
st.title("Titanic Dataset: Data Cleaning & Exploratory Data Analysis")
st.write(
    "This dashboard cleans the Titanic training dataset and explores the "
    "relationships between passenger attributes and survival."
)


@st.cache_data
def load():
    return pd.read_csv("train.csv")


raw = load()

# ---------------------------------------------------------------
# 1. Data Cleaning
# ---------------------------------------------------------------
st.header("1. Data Cleaning")

st.subheader("Raw data preview")
st.dataframe(raw.head())

st.subheader("Missing values (before cleaning)")
st.write(raw.isnull().sum())

st.write(f"Duplicate rows: {raw.duplicated().sum()}")

df = raw.copy()
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["deck"] = df["Cabin"].fillna("M").str[0]
df = df.drop(columns=["Cabin"]).drop_duplicates()
df["family_size"] = df["SibSp"] + df["Parch"] + 1

st.markdown(
    """
**Cleaning steps performed**
- **Age:** missing values filled with the median age.
- **Embarked:** missing values filled with the mode (most common port).
- **Cabin:** mostly missing, so a `deck` column was created from the first letter (missing = 'M') and the original column was dropped.
- **Duplicates:** duplicate rows removed.
- **family_size:** new feature created as SibSp + Parch + 1.
"""
)

st.subheader("Missing values (after cleaning)")
st.write(df.isnull().sum())

# ---------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------
st.header("2. Exploratory Data Analysis")

st.subheader("Summary statistics")
st.write(df.describe())

c1, c2 = st.columns(2)
with c1:
    st.subheader("Survival by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)
with c2:
    st.subheader("Survival by Passenger Class")
    fig, ax = plt.subplots()
    sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Age Distribution by Survival")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x="Age", hue="Survived", common_norm=False, ax=ax)
    st.pyplot(fig)
with c4:
    st.subheader("Fare vs Survival")
    fig, ax = plt.subplots()
    sns.boxplot(x="Survived", y="Fare", data=df, ax=ax)
    st.pyplot(fig)

c5, c6 = st.columns(2)
with c5:
    st.subheader("Survival Rate by Family Size")
    fig, ax = plt.subplots()
    sns.barplot(x="family_size", y="Survived", data=df, ax=ax)
    st.pyplot(fig)
with c6:
    st.subheader("Survival Rate by Port of Embarkation")
    fig, ax = plt.subplots()
    sns.barplot(x="Embarked", y="Survived", data=df, ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

st.subheader("Survival Rate by Class and Gender")
st.write(df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack())

# ---------------------------------------------------------------
# 3. Key Insights
# ---------------------------------------------------------------
st.header("3. Key Insights")
st.markdown(
    """
- Female passengers had a much higher survival rate than male passengers.
- First-class passengers had the best chances of survival, and third-class the lowest.
- Children (lower ages) had a better survival rate than other age groups.
- Passengers who paid higher fares were more likely to survive.
- Passengers in small families survived more often than those travelling alone or in very large families.
"""
)