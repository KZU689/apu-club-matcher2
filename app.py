import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 1. Mock Dataset (In a real project, you can load this from a CSV file)
data = {
    "Club_Name": [
        "Google Developer Groups on Campus (GDGoC APU)",
        "APU Cyber Security Club (CTF)",
        "APU Game Development Club",
        "APU Data Science Club",
        "APU Cloud Computing Club",
        "APU Artificial Intelligence Club",
    ],
    "Category": [
        "Software & Google Tech",
        "Cybersecurity",
        "Gaming & AR/VR",
        "Data & Analytics",
        "Cloud & DevOps",
        "Artificial Intelligence",
    ],
    "Description": [
        "Build mobile and web apps, explore Google AI tools, machine learning, and web development through workshops and hackathons.",
        "Learn ethical hacking, penetration testing, network security, and participate in global Capture The Flag competitions.",
        "Design 2D and 3D games using Unity, Unreal Engine, C#, and C++ with fellow student developers.",
        "Learn data analysis, Python libraries like Pandas and NumPy, data visualization, and basic predictive modeling.",
        "Master cloud infrastructure platforms like AWS, Azure, Google Cloud Platform (GCP), and containerization tools like Docker.",
        "Explore deep learning, natural language processing (NLP), computer vision, and build real-world AI applications using Python.",
    ],
    "Tags": [
        "python web mobile ai google hackathon code",
        "cybersecurity hacking security network ctf linux",
        "game unity unreal 3d design c++ c#",
        "data pandas python numpy analytics statistics",
        "cloud aws azure gcp devops docker linux",
        "ai machine learning python nlp deep learning tensorflow",
    ],
}

df = pd.DataFrame(data)

# 2. Streamlit Web App Interface
st.set_page_config(
    page_title="APU Club & Tech Matcher", page_icon="🎓", layout="centered"
)

st.title("🎓 APU Smart Club & Tech Matcher")
st.write(
    "Not sure which club or tech track to join? Type your interests below and let our AI-powered recommendation system find the best match for you!"
)

# User Input Text Box
user_input = st.text_input(
    "What are you interested in learning or building?",
    placeholder="e.g., Python, machine learning, cybersecurity, game design...",
)

if st.button("Find My Club Match!"):
  if user_input.strip() == "":
    st.warning("Please enter at least one interest or keyword!")
  else:
    # 3. Recommendation Engine (TF-IDF & Cosine Similarity)
    # Combine Description and Tags for better matching
    corpus = df["Description"] + " " + df["Tags"]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Transform user input into vector space
    user_vector = vectorizer.transform([user_input])

    # Calculate Cosine Similarity between user input and all clubs
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Add scores to dataframe and sort
    df["Match_Score"] = similarity_scores
    sorted_df = df.sort_values(by="Match_Score", ascending=False)

    # Display Results
    st.success("Here are the best matches for your interests:")

    for index, row in sorted_df.iterrows():
      # Only show results with a match greater than 0
      if row["Match_Score"] > 0.05:
        with st.container():
          st.subheader(f"🏆 {row['Club_Name']}")
          st.write(f"**Category:** {row['Category']}")
          st.write(f"**About:** {row['Description']}")
          # Convert score to a neat percentage
          percentage = int(row["Match_Score"] * 100)
          st.progress(
              row["Match_Score"],
              text=f"Match Confidence Score: {percentage}%",
          )
          st.divider()
      else:
        # If top scores are too low
        pass

    # Fallback if no good match is found
    if sorted_df.iloc[0]["Match_Score"] <= 0.05:
      st.info(
          "No direct keyword matches found. Try searching for terms like"
          " 'Python', 'AI', 'Security', or 'Games'!"
      )
