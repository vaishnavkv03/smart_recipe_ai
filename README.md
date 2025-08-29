# SMART RECIPE AI

📖 About The Project
Ever stared into your fridge, full of random ingredients, with no idea what to cook? The AI Recipe Generator is designed to solve that exact problem. By simply providing a list of ingredients you have on hand, our intelligent system leverages advanced Machine Learning and Deep Learning models to craft a unique recipe just for you.

Our mission is to make cooking simpler, reduce food waste, and inspire you to try new dishes with the items you already own.

✨ Key Features
Personalized Recipes: Get cooking instructions tailored specifically to your ingredients.

Reduce Food Waste: Find creative uses for leftover food items.

Discover New Dishes: Explore new culinary possibilities you might not have thought of.

Powered by AI: Utilizes both classical ML and modern DL techniques for robust predictions.

🛠️ Technical Breakdown
This project follows a complete data science pipeline, from data acquisition and cleaning to model training and deployment.

1. Data Acquisition
The foundation of our recipe generator is a rich, diverse dataset of recipes. We scraped data from various online culinary sources using a combination of powerful Python libraries:

Beautiful Soup: For parsing HTML and XML documents to extract recipe information.

2. Data Preprocessing
Raw data is never clean. To ensure our models receive high-quality input, we applied a rigorous preprocessing pipeline:

Data Cleaning: Removed inconsistencies, handled missing values, and standardized recipe formats.

Feature Selection: Identified the most relevant textual features (ingredients, titles) that contribute to a successful recipe prediction.

Normalization: Standardized all text to lowercase and removed punctuation to maintain consistency.

3. Machine Learning Model
For our baseline approach, we selected a classic and highly effective text classification model.

Model: Multinomial Naive Bayes paired with TfidfVectorizer.

Why? This combination is a powerful standard for text-based feature extraction. TfidfVectorizer converts raw text into a meaningful numerical representation by weighing the importance of each ingredient. The MultinomialNB classifier is incredibly efficient and performs exceptionally well on text classification tasks, making it a perfect starting point.

4. Deep Learning Model
To capture more nuanced and semantic relationships between ingredients, we implemented a neural network.

Model: A custom-built Sequential Neural Network.

Why? Deep learning allows us to move beyond simple word frequencies. Our model architecture is designed to understand the context and relationships between ingredients.

Embedding Layer: This is the core of our model. It transforms each ingredient token into a dense vector, capturing its semantic meaning. Ingredients like "tomato" and "basil" will have closer vector representations than "tomato" and "flour."

Dense Layers: These fully connected layers process the embedded vectors to learn complex patterns and interactions.

Softmax Activation: The final layer uses a softmax function to output the probability distribution over all possible recipe categories, giving us the most likely recipe for the given ingredients.

🚀 Getting Started
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.8+

pip

Installation
Clone the repo

Bash

git clone https://github.com/vaishnavkv03/smart_recipe_ai.git
Install Python packages

Bash

pip install -r requirements.txt
