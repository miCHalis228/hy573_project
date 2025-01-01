from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_word_clouds(components, features):
    try:
        # Define the number of topics
        num_topics = len(components)
        print(num_topics)
        # Calculate the number of rows and columns for the subplots
        cols = 4  # Number of columns (adjust as needed)
        rows = (num_topics + cols - 1) // cols  # Calculate rows needed to fit all subplots

        # Create a larger figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2))  # Adjust figure size

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        for i in range(num_topics):
            # Generate a word cloud for the topic
            wordcloud = WordCloud(width=600, height=400).generate_from_frequencies(
                dict(zip(features, components[i]))
            )

            # Plot the word cloud in the corresponding subplot
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].axis('off')  # Turn off axis
            axes[i].set_title(f"Topic {i+1}")  # Set title for each subplot

        # Turn off any unused subplots
        for j in range(num_topics, len(axes)):
            axes[j].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the combined figure
        plt.show()
    except Exception:
        print(Exception)
