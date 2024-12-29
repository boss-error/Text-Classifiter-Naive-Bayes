from flask import Flask, render_template, request
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Example data with sentences and corresponding tags


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Predict text provided by the user (e.g., "A very close gamea")
        predict = request.form['predict'].lower()
        words_in_sentence = predict.split()
        sentences = request.form.getlist('sentence[]')
        labels = request.form.getlist('label[]')
        data = pd.DataFrame({
            'sentence': sentences,
            'label': labels
        })

# Class labels
        classes = data['label'].unique()

        # Word frequency calculation
        word_freq = {label: {} for label in classes}
        total_words = {label: 0 for label in classes}

# Tokenize sentences and count word frequencies
        for idx, row in data.iterrows():
            words = row['sentence'].lower().split()
            class_label = row['label']
            total_words[class_label] += len(words)
            
            for word in words:
                word_freq[class_label][word] = word_freq[class_label].get(word, 0) + 1

        # Unique words
        unique_words = set(word for freq_dict in word_freq.values() for word in freq_dict.keys())
        # Prior probabilities for each class
        class_priors = {label: len(data[data['label'] == label]) / len(data) for label in classes}

        # Probability calculations for each class
        prob = {label: class_priors[label] for label in classes}
        
        # Calculate probabilities of words in each class using Laplace smoothing
        for word in words_in_sentence:
            for label in classes:
                prob[label] *= (word_freq[label].get(word, 0) + 1) / (total_words[label] + len(unique_words))

        # Get the classification with the highest probability
        classification = max(prob, key=prob.get)

        # Generate LaTeX outputs for table, equations, and steps
        table_latex = generate_latex_table(word_freq, class_priors, prob, classification,data)
        equations_latex = generate_equations(words_in_sentence, word_freq, total_words)
        steps_latex = generate_steps(words_in_sentence, word_freq, class_priors, prob, total_words, unique_words)


        return render_template('solver.html', 
                               table_latex=table_latex, 
                               equations_latex=equations_latex,
                               steps_latex=steps_latex,
                               result_latex=f"Classification: {classification}")

    return render_template('index.html')

# Function to generate LaTeX table dynamically
# Function to generate LaTeX table dynamically
def generate_latex_table(table_data,word_freq, class_priors, prob,data):
    # Sample data with sentences and their tags
    table_data = list(zip(data['sentence'], data['label']))
    
    # Count the number of rows per class (for probability calculation)
    class_counts = defaultdict(int)
    for _, tag in table_data:
        class_counts[tag] += 1
    
    # Calculate the total number of rows
    total_rows = len(table_data)
    
    # Calculate the probability of each class
    class_probabilities = {cls: count / total_rows for cls, count in class_counts.items()}

    # Initialize a dictionary to store word counts for each unique class
    word_count = defaultdict(lambda: defaultdict(int))

    # Process the sentences and count word frequencies per class
    for sentence, tag in table_data:
        words = sentence.lower().split()  # Split sentence into words and convert to lowercase
        for word in words:
            word_count[tag][word] += 1

    # Generate LaTeX table for sentence and tag
    latex_table = """
    \\begin{array}{|l|l|} \\hline
    Text & Tag \\\\ \\hline
    """
    
    for sentence, tag in table_data:
        # Ensuring spaces between words in LaTeX table entries
        sentence = sentence.replace(" ", " \\ ")
        tag = tag.replace(" ", " \\ ")
        latex_table += f"{sentence} & {tag} \\\\ \\hline\n"
    
    latex_table += """
    \\end{array}
    """

    # Add class probabilities to the LaTeX table under each label
    latex_table += """
    \\begin{array}{|l|c|} \\hline
    & P(class) \\\\ \\hline
    """
    
    for cls, prob in class_probabilities.items():
        cls = cls.replace(" ", " \\ ")
        latex_table += f"{cls} & {prob:.3f} \\\\ \\hline\n"
    
    latex_table += """
    \\end{array}
    """

    # Generate LaTeX table for word frequency
    word_freq_table = """
    \\begin{array}{|l|""" + "c|" * len(word_count) + """} \\hline
    & Words """ + "".join([f"& {label} " for label in word_count.keys()]) + """ \\\\ \\hline
    """

# Get the unique words from the word count dictionaries and sort them alphabetically
    all_words = sorted(set(word for sentence, tag in table_data for word in sentence.lower().split()))

    # Initialize sums for each class label
    class_sums = {label: 0 for label in word_count}

    # Counter for the first column
    counter = 1

    # Populate word frequency table with sorted words and calculate sums
    for word in all_words:
        word_freq_table += f"{counter} & {word} "
        for label in word_count:
            word_freq_table += f"& {word_count[label].get(word, 0)} "
            class_sums[label] += word_count[label].get(word, 0)
        word_freq_table += "\\\\ \\hline\n"
        counter += 1  # Increment counter for the next row

    # Add the sums row at the bottom
    word_freq_table += "& Total "  # Add an extra column for the "Total" label
    for label in word_count:
        word_freq_table += f"& {class_sums[label]} "
    word_freq_table += "\\\\ \\hline\n"

    word_freq_table += """
    \\end{array}
"""

    # Merge both LaTeX tables
    merged_table = latex_table + "\n\n" + word_freq_table

    
    return merged_table



# Function to generate LaTeX equations dynamically
def generate_equations(words_in_sentence, word_freq, total_words):
    equations = """
    $$ P(Class) = \\frac{\\text{Count of Class}}{\\text{Total Sentences}} $$
    $$ P(Word|Class) = \\frac{\\text{Count of Word in Class} + 1}{\\text{Total Words in Class} + \\text{Unique Words}} $$
    """
    for label in word_freq:
        # Equation structure: P(Class|Sentence) = P(Word1|Class) * P(Word2|Class) * ... * P(Class)
        terms = [f"P({word}|{label})" for word in words_in_sentence]
        terms.append(f"P({label})")
        equation = f"P({label}|\\text{{sentence}}) = " + " \\cdot ".join(terms)
        
        equations += f"$$ {equation} $$\n"
   
    return equations
# Function to generate LaTeX equations dynamically


# Function to generate LaTeX steps dynamically
def generate_steps(words_in_sentence, word_freq, class_priors, prob, total_words, unique_words):
    steps = """
    <ul>
        <li>Step 1: Tokenize the sentence into words.</li>
        <ul>
    """
    for word in words_in_sentence:
        steps += f"<li>{word}</li>"
    steps += "</ul>"

    steps += """
        <li>Step 2: Calculate the probability for each word in the sentence for each class.</li>
        <ul>
    """
    for word in words_in_sentence:
        for label in word_freq:
            P_word_class = (word_freq[label].get(word, 0) + 1) / (total_words[label] + len(unique_words))
            steps += f"""
            $$ P({word}|{label}) = \\frac{{{word_freq[label].get(word, 0)} + 1}}{{{total_words[label]} + {len(unique_words)}}} = {P_word_class:.4f} $$
            """
    steps += "</ul>"

    # Step 3: Multiply probabilities for all words in the sentence for each class.
    steps += """
        <li>Step 3: Multiply probabilities for all words in the sentence for each class.</li>
        <ul>
    """

    # Calculate P(Class | Sentence) for each class
    for label in class_priors:
        class_prob = class_priors[label]  # P(Class)
        prob_class_sentence = class_prob

        word_probs = []  # To store word probabilities for product calculation
        cumulative_prob = class_prob  # To calculate the cumulative product of probabilities
        for word in words_in_sentence:
            P_word_class = (word_freq[label].get(word, 0) + 1) / (sum(word_freq[label].values()) + len(unique_words))
            word_probs.append(f"*{P_word_class:.4f}")
            cumulative_prob *= P_word_class  # Update the cumulative probability for the sentence

        # Format the product display with the probabilities of each word
        product_str = " ".join(word_probs)
        # Display the full cumulative multiplication
        steps += f"""
    <li>
        P({label}|sentence) = {class_prob}{product_str} = {cumulative_prob:.8f}
    </li>
    """

    steps += "</ul>"

    return steps




if __name__ == '__main__':
    app.run(debug=True)
