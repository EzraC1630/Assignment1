# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import re
import os
import sys
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

def task1():
    with open(datafilepath) as f:
        Data = json.load(f)
    code_l = Data['teams_codes']
    code_l.sort()
    return(code_l)
    
    
def task2():
    with open(datafilepath) as f:
        Data = json.load(f)
    clubs_l = Data['clubs']

    # create dataframe and sort the values
    df = pd.DataFrame(clubs_l).loc[:,['club_code', 'goals_scored', 'goals_conceded']]
    df = df.sort_values(by = 'club_code')

    # edit rownames
    df = df.set_axis(['Team code', 'Goals scored by team', 'Goals scored against team'], axis = 'columns')
    df_edited = df.set_axis(list(range(20)), axis = 'index')

    #create the csv file
    df.to_csv(r'task2.csv', index = False, header=True)
    
    return(df)
   
    
def task3():
    total_goals = [0]*265
    file_n = [0]*265
    for i in range(265):
        a = i+1
        if a < 10: 
            filen = 'data/football/00' + str(a) + '.txt'
            file_n[i] = '00' + str(a) + '.txt'
        if 10 <= a < 100:
            filen = 'data/football/0' + str(a) + '.txt'
            file_n[i] = '0' + str(a) + '.txt'
        if 100 <= a :
            filen = 'data/football/' + str(a) + '.txt'
            file_n[i] = str(a) + '.txt'

        # open file
        with open(filen) as f:
            string = f.read()

        pattern = r'(\d+)[-](\d+)'

        if re.search(pattern, string) :
            re.findall(pattern, string)
            score = re.findall(pattern, string)
            sum_t = [0]*(len(score))

            for b in range(len(score)):
                sum1 = sum(tuple(map(int, score[b])))
                if sum1 < 101:
                    sum_t[b] = sum1

            total_goals[i] = max(sum_t)

    df1 = pd.DataFrame({'Filename': file_n, 'Total_goals': total_goals})
    df1.to_csv(r'task3.csv', index = False, header=True)
    
    return(df1)


def task4():
    df1 = task3()
    t_g = df1.iloc[:,1]
    names = df1.iloc[:, 0]
    t_g.index = names

    %matplotlib inline
    import matplotlib.pyplot as plt

    plt.boxplot(t_g)
    plt.title('Boxplot for Total_Goals')
    plt.savefig('task4.png')
    
    return

    
def task5():
    with open(datafilepath) as f:
        Data = json.load(f)
    club_n = Data['participating_clubs']
    club_n.sort()
    n_mentions = [0]*20

    for i in range(265):
        a = i+1
        if a < 10: 
            filen = 'data/football/00' + str(a) + '.txt'
        if 10 <= a < 100:
            filen = 'data/football/0' + str(a) + '.txt'
        if 100 <= a :
            filen = 'data/football/' + str(a) + '.txt'

            # open file
        with open(filen) as f:
            string = f.read()

        pattern1 = r'Arsenal'
        pattern2 = r'Bournemouth'
        pattern3 = r'Brighton'
        pattern4 = r'Burnley'
        pattern5 = r'Cardiff'
        pattern6 = r'Chelsea'
        pattern7 = r'Crystal Palace'
        pattern8 = r'Everton'
        pattern9 = r'Fulham'
        pattern10 = r'Huddersfield'
        pattern11 = r'Leicester City'
        pattern12 = r'Liverpool'
        pattern13 = r'Manchester City'
        pattern14 = r'Manchester United'
        pattern15 = r'Newcastle United'
        pattern16 = r'Southampton'
        pattern17 = r'Tottenham'
        pattern18 = r'Watford'
        pattern19 = r'West Ham United'
        pattern20 = r'Wolverhampton'

        if re.search(pattern1, string) :
            n_mentions[0] = n_mentions[0] + 1 
        if re.search(pattern2, string) :
            n_mentions[1] = n_mentions[1] + 1       
        if re.search(pattern3, string) :
            n_mentions[2] = n_mentions[2] + 1        
        if re.search(pattern4, string) :
            n_mentions[3] = n_mentions[3] + 1
        if re.search(pattern5, string) :
            n_mentions[4] = n_mentions[4] + 1
        if re.search(pattern6, string) :
            n_mentions[5] = n_mentions[5] + 1
        if re.search(pattern7, string) :
            n_mentions[6] = n_mentions[6] + 1
        if re.search(pattern8, string) :
            n_mentions[7] = n_mentions[7] + 1
        if re.search(pattern9, string) :
            n_mentions[8] = n_mentions[8] + 1
        if re.search(pattern10, string) :
            n_mentions[9] = n_mentions[9] + 1
        if re.search(pattern11, string) :
            n_mentions[10] = n_mentions[10] + 1
        if re.search(pattern12, string) :
            n_mentions[11] = n_mentions[11] + 1
        if re.search(pattern13, string) :
            n_mentions[12] = n_mentions[12] + 1
        if re.search(pattern14, string) :
            n_mentions[13] = n_mentions[13] + 1
        if re.search(pattern15, string) :
            n_mentions[14] = n_mentions[14] + 1
        if re.search(pattern16, string) :
            n_mentions[15] = n_mentions[15] + 1
        if re.search(pattern17, string) :
            n_mentions[16] = n_mentions[16] + 1
        if re.search(pattern18, string) :
            n_mentions[17] = n_mentions[17] + 1
        if re.search(pattern19, string) :
            n_mentions[18] = n_mentions[18] + 1
        if re.search(pattern20, string) :
            n_mentions[19] = n_mentions[19] + 1


    df = pd.DataFrame({'club_name': club_n, 'number_of_mentions': n_mentions})
    df.to_csv(r'task5.csv', index = False, header=True)
    
    %matplotlib inline
    import matplotlib.pyplot as plt
    import calendar
    from numpy import arange
    
    n_men = df.iloc[:,1]
    clu_n = df.iloc[:,0]
    plt.bar(arange(len(n_men)),n_men)
    plt.xticks( arange(len(clu_n)),clu_n, rotation=60)
    
    plt.title('Barplot for Number of Mentions of Clubs')
    plt.ylabel('Number of Mentions')
    plt.xlabel('Clubs')
    
    plt.savefig('task5.png')
    
    return(df)


    
def task6():
    with open(datafilepath) as f:
        Data = json.load(f)
        
    club_n = Data['participating_clubs']
    df = task5()

    pattern1 = r'Arsenal'
    pattern2 = r'Bournemouth'
    pattern3 = r'Brighton'
    pattern4 = r'Burnley'
    pattern5 = r'Cardiff'
    pattern6 = r'Chelsea'
    pattern7 = r'Crystal Palace'
    pattern8 = r'Everton'
    pattern9 = r'Fulham'
    pattern10 = r'Huddersfield'
    pattern11 = r'Leicester City'
    pattern12 = r'Liverpool'
    pattern13 = r'Manchester City'
    pattern14 = r'Manchester United'
    pattern15 = r'Newcastle United'
    pattern16 = r'Southampton'
    pattern17 = r'Tottenham'
    pattern18 = r'Watford'
    pattern19 = r'West Ham United'
    pattern20 = r'Wolverhampton'

    pattern = [0]*20
    pattern[0] = pattern1
    pattern[1] = pattern2
    pattern[2] = pattern3
    pattern[3] = pattern4
    pattern[4] = pattern5
    pattern[5] = pattern6
    pattern[6] = pattern7
    pattern[7] = pattern8
    pattern[8] = pattern9
    pattern[9] = pattern10
    pattern[10] = pattern11
    pattern[11] = pattern12
    pattern[12] = pattern13
    pattern[13] = pattern14
    pattern[14] = pattern15
    pattern[15] = pattern16
    pattern[16] = pattern17
    pattern[17] = pattern18
    pattern[18] = pattern19
    pattern[19] = pattern20

    sim_12 = []


    for i in list(range(20)):
        for a in list(range(20)):
            n1 = df.iloc[i,1]
            n2 = df.iloc[a,1]
            n12 = 0

            for c in range(265):
                d = c+1
                if d < 10: 
                    filen = 'data/football/00' + str(d) + '.txt'
                if 10 <= d < 100:
                    filen = 'data/football/0' + str(d) + '.txt'
                if 100 <= d :
                    filen = 'data/football/' + str(d) + '.txt'

                    # open file
                with open(filen) as f:
                    string = f.read()

                if (re.search(pattern[i], string)!= None) & (re.search(pattern[a], string)!= None):
                    n12 =  n12 + 1

            similarity = 2*n12 / (n1 + n2)
            sim_12.append(similarity)


    sim_12 = pd.Series(sim_12)
    club_n = [
        pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1,
        pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, pattern1, 
        pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2,
        pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, pattern2, 
        pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3,
        pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, pattern3, 
        pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, 
        pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, pattern4, 
        pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5,
        pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5, pattern5,
        pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6,
        pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6, pattern6,
        pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, 
        pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, pattern7, 
        pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8,
        pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8, pattern8,
        pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9,
        pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9, pattern9,
        pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, 
        pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, pattern10, 
        pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11,
        pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11, pattern11,
        pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12,
        pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12, pattern12,
        pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13,
        pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13, pattern13,
        pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14,
        pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14, pattern14,
        pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, 
        pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15, pattern15,
        pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16,
        pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16, pattern16,
        pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, 
        pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, pattern17, 
        pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, 
        pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18, pattern18,
        pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, 
        pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, pattern19, 
        pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20,
        pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20, pattern20
    ]
    
    sim_12.index = club_n
    sim_12 = pd.DataFrame(sim_12, columns=['Similarity Score'])

    import seaborn as sns

    import matplotlib.pyplot as plt

    sns.heatmap(sim_12,cmap='viridis',xticklabels=True)

    plt.title('Heat Map of Similarity Score for each pair of clubs ')
    plt.ylabel('Clubs')
    plt.savefig('task6.png')
    
    return(sim_12)


    
def task7():
    %matplotlib inline
    import matplotlib.pyplot as plt

    df1 = task5()
    df1.reindex([0,2,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    df2 = task2()

    for i in list(range(20)):
        plt.scatter(df2.loc[[i],['Goals scored by team']],df1.loc[[i],['number_of_mentions']])

    plt.xlim(1,12)
    plt.ylim(-10,105)
    plt.ylabel("Number of Mentions")
    plt.xlabel("Total Number of Goals")
    plt.title("Scatterplot for Number of Article Mentions vs Scored Goals")
    plt.grid(True)


    plt.savefig('task7.png')
    
    return
    
    
    
def task8(filename):
    with open(filename) as f:
        s = f.read()

    # lower case
    s = s.lower()

    # Remove all non-alphabetic characters
    pattern1 = r'[^a-zA-Z\s]'
    revised_s = re.sub(pattern1, r' ', s)

    # Convert all spacing characters
    pattern2 = r' *\n+ *| +'
    revised_s = re.sub(pattern2, r' ', revised_s)

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenization
    word_list = nltk.word_tokenize(revised_s)

    # Stopwords
    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('english'))

    filtered_list = [w for w in word_list if not w in stopWords]

    # Remove single characters
    for word in filtered_list:
        if len(word) == 1:
            filtered_list.remove(word)

 
    return(filtered_list)



def task9():

# Not Done


    list1 = task8('data/football/001.txt')
    list2 = task8('data/football/002.txt')

    # Calculate the term counts
    count_12 = [0]*(len(list1)+len(list2))
    count1 = [1]*len(list1)
    overlap = 0

    for a in range(len(list1)):
        for b in range(len(list2)):
            if list1[a] == list2[b]:
                count_12[a] = 1
                overlap = overlap + 1

    n = len(count_12) - overlap
    count_12 = count_12[:n]

    for a in range(n-len(list1)):
        count1.append(0)
        count_12[-a] = 1

    count_12

    count_n = [count1, count_12]


    # Get TF-IDF Vector
    from sklearn.feature_extraction.text import TfidfTransformer

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count_n)

    q_unit = [x/(math.sqrt(3)) for x in count_12]



    # Calcuate the consine similarity
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(v1, v2):
        return dot(v1, v2)/(norm(v1)*norm(v2))

    doc_tfidf = tfidf.toarray()
    sims = [cosine_sim(q_unit, doc_tfidf[d_id]) for d_id in range(doc_tfidf.shape[0])]

    sims

    return