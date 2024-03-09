import os
import openai
from openai import OpenAI
import json
import requests
from time import sleep
import pandas as pd
import traceback

openai.organization = "<your org id>"

openai.api_key ="<your API key>"



client = OpenAI(
   
    api_key="<your API key>",
)

#openai functions 
tools = [
     {
        "type": "function",
        "function": {
            "name": "complete_job_features",
            "description": "Interpret and Complete job data.",
            "parameters":{ 

                            'type': 'object',
                            'properties': {
                                            'meta': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'job_title': {
                                                            'type': 'string',                                                            
                                                            'description': 'This is the title of the position in the most basic form.',
                                                        },
                                                        'job_category': {
                                                            'type': 'string',                                                            
                                                            'description': 'The common category of job type.',
                                                        },
                                                        'job_class': {
                                                            'type': 'string',                                                            
                                                            'description': 'The common class of the job.',
                                                        },
                                                        'sector': {
                                                            'type': 'string',
                                                            'description': 'The sector of the job: Software Engineering , Management, or Sales.',
                                                        },            
                                                        'salary': {
                                                            'type': 'integer',
                                                            'description': 'Estimate a yearly salary. required.',
                                                        },      
                                                        'remote': {
                                                            'type': 'boolean',
                                                            'description': 'True or False for if remote work is possible',
                                                        }                                                                                                 
                                                    }
                                                }
                                            },

                                            'skills': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'skill': {
                                                            'type': 'string',                                                            
                                                            'description': 'This is the name of the identified skill.',
                                                        },
                                                        'experience': {
                                                            'type': 'integer',
                                                            'description': 'The number of years experience for the skill.',
                                                        }                                                                                                                
                                                    }
                                                }
                                            },

                                            'duties_responsibilities': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'duty': {
                                                            'type': 'string',                                                            
                                                            'description': 'A short description of a single duty or responsibility for this role',
                                                        },                                                       
                                                    }
                                                }
                                            },

                                            'subjective_elements': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'soldier_general': {
                                                            'type': 'integer',                                                            
                                                            'description': 'A score between 0-255 as a measure of self driven leadership requirement.',
                                                        },
                                                        'student_teacher': {
                                                            'type': 'integer',
                                                            'description': 'A score between 0-255 as a measure of mentoring requirement.',
                                                        },
                                                        
                                                        'introvert_extrovert': {
                                                            'type': 'integer',
                                                            'description': 'A score between 0-255 as a measure of leadership requirement.',
                                                        },
                                                    }
                                                }
                                            },

                                            'education_certification': {
                                                'type': 'array',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'credential': {
                                                            'type': 'string',                                                            
                                                            'description': 'This the certification or credential required.',
                                                        },
                                                   
                                                       
                                                    }
                                                }
                                            }
                            },

                        }
            }                
     },  
    
]


json_example = '''
{
  "meta": [
    {
      "job_title": "Marketing Manager",
      "job_category": "Marketing",
      "job_class": "Manager",
      "sector": "Marketing",
      "salary": 85000,
      "remote": false
    }
  ],
  "skills": [
    {
      "skill": "Marketing Strategy",
      "experience": 5
    },
    {
      "skill": "Team Leadership",
      "experience": 5
    },
    {
      "skill": "Market Analysis",
      "experience": 5
    },
    {
      "skill": "Online Presence Management",
      "experience": 5
    },
    {
      "skill": "Event Planning",
      "experience": 5
    },
    {
      "skill": "Data Analysis",
      "experience": 5
    },
    {
      "skill": "Digital Marketing",
      "experience": 5
    },
    {
      "skill": "Communication Skills",
      "experience": 5
    },
    {
      "skill": "Organizational Skills",
      "experience": 5
    }
  ],
  "duties_responsibilities": [
    {
      "duty": "Develop and execute comprehensive marketing strategies to enhance brand visibility and promote freight and logistics services."
    },
    {
      "duty": "Lead a team of marketing specialists, fostering collaboration and guiding their efforts towards achieving set goals."
    },
    {
      "duty": "Analyze market trends, customer behavior, and competitor activity to identify growth opportunities and refine strategies."
    },
    {
      "duty": "Manage the company's online presence, optimizing the website and social media channels for maximum engagement and conversions."
    },
    {
      "duty": "Plan and oversee participation in industry events, conferences, and trade shows to showcase services and network with potential clients."
    },
    {
      "duty": "Measure and report on the performance of marketing campaigns, using data-driven insights to continuously improve strategies."
    }
  ],
  "subjective_elements": [
    {
      "soldier_general": 100,
      "student_teacher": 80,
      "introvert_extrovert": 120
    }
  ],
  "education_certification": [
    {
      "credential": "Bachelor's degree in Marketing, Business, or a related field (Master's preferred)"
    }
  ]
}
'''





def intepretJobData(job_title, job_text):

    #print(recipe_json)
    #print('running function')
    global json_example

    messages = []
    messages.append({"role": "system", "content": "You are a master HR and Talent Hiring specialist. You know how capture the essence of a job and identifiy it's elements.  You are also specialized in job data, and salary information.  You know all the distinct jobs typ classes and industries."})
    messages.append({"role": "user", "content": "Please interpret and isolate elements of this job posting from plain text to json, and make sure to give a realistic salary for the role.  When presented with functions as tool , you provide back all parameters.: \n\n" + job_text})
    messages.append({"role": "user", "content": "Make sure to return your response in this json format only : \n\n" + json_example })

    chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-1106",
            tools=tools,
        )

    #print(chat_completion)
    #print(chat_completion.choices[0].message.tool_calls[0].function.arguments)
    features = json.loads(chat_completion.choices[0].message.tool_calls[0].function.arguments)
    #output = json.dumps(features, indent=2)
    return features







if __name__ == "__main__":

    count=0

    job_text = '''
        The Company  At Springboard, we’re on a mission to bridge the world’s skills gap, offering transformative online education in data science, UI/UX design, machine learning, and coding. Our courses may be tech-enabled, but we're ultimately human-centric: each student taps into a vast community throughout their time with us, engaging with fellow students, industry-expert mentors, student advisors, and career coaches, the goal of which is to successfully transition students into their dream job. Through this hybrid approach, we’ve helped thousands of learners revamp their careers and, by extension, their lives, with hundreds of top-notch job offers received every year and a near-perfect placement rate for our program graduates.\nAbout the course\nWe’re looking for mentors for our Machine Learning Engineering Career Track course. This 6-month course is primarily designed for students who want to become Machine Learning Engineers.\nAs part of the course, students start with an introduction to basic ML algorithms and quickly advance to topics like large language models and generative AI. Through hands-on projects and practical exercises, they will master the entire machine learning pipeline, from data preprocessing and feature engineering to model deployment and scaling. They will gain proficiency in popular frameworks and tools like TensorFlow, Scikit-Learn, and AWS, equipping you with the ability to develop and deploy machine learning models at scale. The course goes beyond just the technical aspects of machine learning. Students also explore ethical considerations surrounding AI and learn how to build models that are fair, transparent, and unbiased. We'll cover topics like interpretability, bias detection, and privacy, ensuring you have a well-rounded understanding of the field. \nThe Opportunity\nSpringboard runs an online Machine Learning Engineering Career Track Bootcamp in which participants learn with the help of a case-study-based curriculum and 1-1 guidance from an expert mentor.\nWe are currently focused on seeking Machine Learning Engineers with proficient skills in Artificial Intelligence Design, Python/Python Libraries,SQL, AWS, Model Evaluations and Model Deployment. These individuals must be passionate about mentoring as Machine Learning Engineering, and can give a few hours per week in return for an honorarium, we would love to hear from you.\nQuestions? Please write to us at mentorrecruiting@springboard.com\nThe Program:\nCompletely online and self-pacedStudents become proficient in Machine Learning Engineering with the help of a curated online curriculum and project-based deliverablesStudents are working professionals from all over the worldStudents have a biweekly 30-minute check in with their mentor to discuss questions, projects, and career advice!Students communicate with mentors outside of calls on an as-needed basis to support learning and career objectives\nYou:\nAre as passionate about teaching coding as about machine learning itself3+ years of industry experience in machine learning engineering or data scienceExperience in Python Data Science Stack; using python and its standard libraries; building visualizations with Matplotlib and Seaborn;Writing code in Python using PEP 8 StandardData Wrangling: Use Pandas to wrangle and clean data; working with different file formats, from plain text, to CSV, to JSON; relational and non-relational databases; SQLMachine Learning: Scikit-learn; ML techniques: linear and logistic regression, trees, and clustering;Deploying ML: Experience deploying ML solutions with a cloud service provider (at least one of AWS, GCP or Azure).Deep Learning Fundamentals: Familiarity with deep learning, computer vision, natural language processingExpertise in Python, libraries like TensorFlow and PyTorch, SQL, cloud ML, etc.Experience working with large language models and generative AI, such as training and fine-tuning language models for specific tasks.Knowledge of cloud-based deployment strategies for large AI models and practical experience with model versioning and reproducibility.Degree/Certifications in machine learning/AI/data science preferredHave experience critiquing work, in particular giving meaningful feedback on machine learning engineering and are able to think on your feet quicklyAre empathetic and have excellent communication skillsBonus points if you have experience hiring in the ML/AI field\nBenefits:\nReceive a monthly per-student honorarium between $14.85 - $27.00 USD per 30 minute call per student Membership in a rich community of expert mentors from great companies like AirBnB, Uber, Google, and PivotalChange the lives of students in our programHelp us revolutionize online education and inspire the next generation of software engineers!Work at your convenience\n We are an equal opportunity employer and value diversity at our company. We welcome applications from all backgrounds, and do not discriminate on the basis of race, religion, national origin, gender, sexual orientation, age, marital status, veteran status, or disability status. California Privacy Rights Notice for Job ApplicantsUnder the California Consumer Privacy Act (“CCPA”), Springboard is required to inform California residents who are job applicants about the categories of personal information we collect about you and the purposes for which we will use this information. This notice contains disclosures required by the CCPA and applies only to personal information that is subject to the CCPA.
    '''

    jobs = pd.read_csv("../big_job_data.csv")

    #jobs = jobs.sample(10)

    def llm(jobrow):

        try:
            global count
            count=count+1
            print(count)
            result = intepretJobData(jobrow['title'], jobrow['description_x'])
            #print(json.dumps(result))
            #print(result['meta'])
            with open("../jobjsonfiles/"+str(jobrow['job_id'])+'-'+(result['meta'][0]['job_title']).replace(' ', '-').replace('/', '-')+'.json', 'w') as jobfile:
                jobfile.write(json.dumps(result, indent=2))
        except Exception as e:
            print(traceback.print_exc())

    jobs.apply(llm, axis=1)
