import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download and load NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def createWordCloud(text):
  
    # Remove stopwords
    filtered_words = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # Function to make all words black
    def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(0,100%, 0%)"

    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=800, background_color='white', 
                        stopwords=stop_words, min_font_size=10, max_words=30,
                        color_func=black_color_func).generate(filtered_words)


    # Display the generated image
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show()



if __name__ == '__main__':
    job_text = '''
        The Company  At Springboard, we’re on a mission to bridge the world’s skills gap, offering transformative online education in data science, UI/UX design, machine learning, and coding. Our courses may be tech-enabled, but we're ultimately human-centric: each student taps into a vast community throughout their time with us, engaging with fellow students, industry-expert mentors, student advisors, and career coaches, the goal of which is to successfully transition students into their dream job. Through this hybrid approach, we’ve helped thousands of learners revamp their careers and, by extension, their lives, with hundreds of top-notch job offers received every year and a near-perfect placement rate for our program graduates.\nAbout the course\nWe’re looking for mentors for our Machine Learning Engineering Career Track course. This 6-month course is primarily designed for students who want to become Machine Learning Engineers.\nAs part of the course, students start with an introduction to basic ML algorithms and quickly advance to topics like large language models and generative AI. Through hands-on projects and practical exercises, they will master the entire machine learning pipeline, from data preprocessing and feature engineering to model deployment and scaling. They will gain proficiency in popular frameworks and tools like TensorFlow, Scikit-Learn, and AWS, equipping you with the ability to develop and deploy machine learning models at scale. The course goes beyond just the technical aspects of machine learning. Students also explore ethical considerations surrounding AI and learn how to build models that are fair, transparent, and unbiased. We'll cover topics like interpretability, bias detection, and privacy, ensuring you have a well-rounded understanding of the field. \nThe Opportunity\nSpringboard runs an online Machine Learning Engineering Career Track Bootcamp in which participants learn with the help of a case-study-based curriculum and 1-1 guidance from an expert mentor.\nWe are currently focused on seeking Machine Learning Engineers with proficient skills in Artificial Intelligence Design, Python/Python Libraries,SQL, AWS, Model Evaluations and Model Deployment. These individuals must be passionate about mentoring as Machine Learning Engineering, and can give a few hours per week in return for an honorarium, we would love to hear from you.\nQuestions? Please write to us at mentorrecruiting@springboard.com\nThe Program:\nCompletely online and self-pacedStudents become proficient in Machine Learning Engineering with the help of a curated online curriculum and project-based deliverablesStudents are working professionals from all over the worldStudents have a biweekly 30-minute check in with their mentor to discuss questions, projects, and career advice!Students communicate with mentors outside of calls on an as-needed basis to support learning and career objectives\nYou:\nAre as passionate about teaching coding as about machine learning itself3+ years of industry experience in machine learning engineering or data scienceExperience in Python Data Science Stack; using python and its standard libraries; building visualizations with Matplotlib and Seaborn;Writing code in Python using PEP 8 StandardData Wrangling: Use Pandas to wrangle and clean data; working with different file formats, from plain text, to CSV, to JSON; relational and non-relational databases; SQLMachine Learning: Scikit-learn; ML techniques: linear and logistic regression, trees, and clustering;Deploying ML: Experience deploying ML solutions with a cloud service provider (at least one of AWS, GCP or Azure).Deep Learning Fundamentals: Familiarity with deep learning, computer vision, natural language processingExpertise in Python, libraries like TensorFlow and PyTorch, SQL, cloud ML, etc.Experience working with large language models and generative AI, such as training and fine-tuning language models for specific tasks.Knowledge of cloud-based deployment strategies for large AI models and practical experience with model versioning and reproducibility.Degree/Certifications in machine learning/AI/data science preferredHave experience critiquing work, in particular giving meaningful feedback on machine learning engineering and are able to think on your feet quicklyAre empathetic and have excellent communication skillsBonus points if you have experience hiring in the ML/AI field\nBenefits:\nReceive a monthly per-student honorarium between $14.85 - $27.00 USD per 30 minute call per student Membership in a rich community of expert mentors from great companies like AirBnB, Uber, Google, and PivotalChange the lives of students in our programHelp us revolutionize online education and inspire the next generation of software engineers!Work at your convenience\n We are an equal opportunity employer and value diversity at our company. We welcome applications from all backgrounds, and do not discriminate on the basis of race, religion, national origin, gender, sexual orientation, age, marital status, veteran status, or disability status. California Privacy Rights Notice for Job ApplicantsUnder the California Consumer Privacy Act (“CCPA”), Springboard is required to inform California residents who are job applicants about the categories of personal information we collect about you and the purposes for which we will use this information. This notice contains disclosures required by the CCPA and applies only to personal information that is subject to the CCPA.
    '''

    createWordCloud(job_text)