from itertools import count
import operator
import math
from ast import pattern
from glob import glob
from operator import index
import os
import numpy
from spacy.pipeline import EntityRuler
import spacy
from spacy import displacy
from scipy import spatial
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import operator
import customtkinter as ctk
import pandas as pd
from tkinter import *
import pickle
import tkinter as tk
from tkinter import Text, END
import pandas as pd

Pstem=PorterStemmer()
i=-1
j=-1
nlp=spacy.load("en_core_web_sm")
V_soft=[]
V_soft_stem=[]
V_tech=[]
V_tech_stem=[]
index_tech={}
index_dup_tech={}
index_soft={}
index_dup_soft={}

def index():
    print("Running")
    index_dup = {}
    DIR = "C:\\Users\\Hp\\Desktop\\6th Semester\\IR\\Resume-Matching-with-JD-master\\cvs_txt\\"
    print("Directory:", DIR)
    
    try:
        sourcepath = os.listdir(DIR)
    except Exception as e:
        print("Error accessing directory:", e)
        return {}

    replace_chars = ['\t', '\n', '.', '. ', ',', '-', ')', '(', '%', '@', ' +', ':', '/', '_', '--', '[', ']', ' #', '&', '"', '\uf0b7', '–', '“', '”', '•', '\ufeff', '’', '|', '\x01', '\uf0a7']
    
    for file in sourcepath:
        inputfile = os.path.join(DIR, file)
        print("Processing file:", inputfile)
        
        try:
            with open(inputfile, encoding="utf8") as f1:
                file_con = f1.read().lower()
        except Exception as e:
            print("Error reading file:", e)
            continue
        
        for char in replace_chars:
            file_con = file_con.replace(char, ' ')
        
        try:
            with open("C:\\Users\\Hp\\Desktop\\6th Semester\\IR\\Resume-Matching-with-JD-master\\Stopword-List.txt", 'r') as f:
                for line in f:
                    for st in line.split():
                        file_con = file_con.replace(' ' + st + ' ', ' ')
        except Exception as e:
            print("Error reading stopword list:", e)
            continue
        
        doc = word_tokenize(file_con)
        index_dup[file] = doc

    return index_dup


def queryIndex(filename):
    replace_chars = ['\t', '\n', '.', '. ', ',', '-', ')', '(', '%', '@', ' +', ':', '/', '_', '--', '[', ']', ' #', '&', '"', '\uf0b7', '–', '“', '”', '•', '\ufeff', '’', '|', '\x01', '\uf0a7']
    f2=open(filename,'r')
    file_con=f2.read().lower()
    for char in replace_chars:
        file_con = file_con.lower().replace(char, ' ')
    f = open("C:\\Users\\Hp\\Desktop\\6th Semester\\IR\\Resume-Matching-with-JD-master\\Stopword-List.txt", 'r')
    for line in f:
        for st in line.split():
            file_con=file_con.lower().replace(' '+st+' ',' ')
    f.close()
    doc=word_tokenize(file_con)
    stemming=[]
    for w in doc:
        stemming.append(Pstem.stem(w))
    doc=stemming
    return doc

#Reads in all created patterns from a file and adds it to the pipeline
def addPattern(): 
    with open("C:\\Users\\Hp\\Desktop\\6th Semester\\IR\\Resume-Matching-with-JD-master\\tech.pickle",'rb') as f:
        pattern=pickle.load(f)
        new_ruler=nlp.add_pipe("entity_ruler")
        new_ruler.add_patterns(pattern)

#Visualize the Skill entities of a doc
# def visualize_entity_ruler(entity_list, doc):    
#     options = {"ents": entity_list}
#     displacy.render(doc, style='ent', options=options)
    
#Create dictionary for softskills of candidates
def makeSoftskills(resume_names,doc):
    '''Create a set of the extracted skill entities of a doc'''
    global j
    j=j+1
    soft_sk=set([ent.label_.upper()[7:] for ent in doc.ents if 'softsk' in ent.label_.lower()])
    index_soft[resume_names[j]]=list(soft_sk)
    for w in soft_sk:
        V_soft.append(w)
    t1=[]
    for w in soft_sk:
        if w not in t1:
            t1.append(w)
    index_dup_soft[resume_names[j]]=list(t1)
    return soft_sk
         
def makeSoftskillsJob(doc):
    '''Create a set of the extracted skill entities of a doc'''
    return set([ent.label_.upper()[7:] for ent in doc.ents if 'softsk' in ent.label_.lower()])
  
def makeTechskillsJob(doc):
    '''Create a set of the extracted skill entities of a doc'''
    return set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()])

#Create dictionary for technical skills of candidates 
def makeTechskills(resume_names,doc):
    global i
    i=i+1
    '''Create a set of the extracted skill entities of a doc'''
    tech_sk=set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()])
    index_tech[resume_names[i]]=list(tech_sk)
    for w in tech_sk:
        V_tech.append(w)
    t1=[]
    for w in tech_sk:
        if w not in t1:
            t1.append(w)
    index_dup_tech[resume_names[i]]=list(t1)
    return tech_sk

def makeEduSet(doc):
    return set([ent.label_.upper()[10:] for ent in doc.ents if 'education' in ent.label_.lower()])

def makeSkillset(resume_names, resume_texts):
    '''Create a dictionary containing a set of the extracted skills. Name is key, matching skillset is value'''
    softskillsets = [makeSoftskills(resume_names,resume_text) for resume_text in resume_texts]
    technicalskillsets = [makeTechskills(resume_names,resume_text) for resume_text in resume_texts]
    educationset = [makeEduSet(resume_text) for resume_text in resume_texts]

    return dict(zip(resume_names, softskillsets)), dict(zip(resume_names, technicalskillsets)), dict(zip(resume_names, educationset)) 
  
def tokenizeText():
    '''Create two lists, one with the names of the candidate and one with the tokenized 
       resume texts extracted from either a .pdf or .doc'''
    replace_chars = ['\t', '\n', '.', '. ', ', ', '-', ')', '(', '%', '@', ' +', ':', '/', '_', '--', '[', ']', ' #', '&', '"', '\uf0b7', '–', '“', '”', '•', '\ufeff', '’', '|']
    resume_texts, resume_names = [], []
    DIR = "C:\\Users\\Hp\\Desktop\\6th Semester\\IR\\Resume-Matching-with-JD-master\\cvs_txt\\"
    sourcepath = os.listdir(DIR)
    for file in sourcepath:
        inputfile = DIR + file
        f1=open(inputfile,encoding="utf8")
        fus=f1.read().lower()
        for char in replace_chars:
            fus = fus.lower().replace(char, ' ')
        resume_names.append(file)
        resume_texts.append(nlp(fus))
    return resume_texts, resume_names
    
def assessEdu(education_dict):
    ed_high={}
    for key in education_dict:
        ed_high.setdefault(key, 0)
        if 'PHD' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.1
        if 'POSTGRADUATE' or 'MASTERS' or 'GRADUATE' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.08
        if 'BACHELORS' or 'UNDERGRADUATE' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.06
        if 'INTERMEDIATE' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.04
        if 'MATRICULATION' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.02
        else:
            ed_high[key]=ed_high[key]+0
    return ed_high

    # education_levels = {
    #     'PHD': 0.1,
    #     'POSTGRADUATE': 0.08,
    #     'MASTERS': 0.08,
    #     'GRADUATE': 0.08,
    #     'BACHELORS': 0.06,
    #     'UNDERGRADUATE': 0.06,
    #     'INTERMEDIATE': 0.04,
    #     'MATRICULATION': 0.02
    # }

    # ed_high = {}
    # for key in education_dict:
    #     ed_high.setdefault(key, 0)
    #     for level, increment in education_levels.items():
    #         if level in education_dict[key]:
    #             ed_high[key] += increment

    # return ed_high    
       
def tfIdfTech(cont):
    DIR = "C:\\Users\\Hp\\Desktop\\6th Semester\\IR\\Resume-Matching-with-JD-master\\cvs_txt\\"
    sourcepath = os.listdir(DIR)
    num_cvs=len(sourcepath)
    wgt={}
    df={}
    idf={}
    tf={}
    dict={}
    index_tech_stem={}

    for key in index_tech:
        index_tech_stem.setdefault(key,[])
        for w in index_tech[key]:
            index_tech_stem[key].append(Pstem.stem(str(j)))
    # print(index_tech)
    #calculating df
    for w in V_tech:
        df.setdefault(w,0)
        for key in index_tech:
            if(w in index_tech[key]):
                df[w]+=1

    #calculating idf
    for key in df:
        idf.setdefault(key,0)
        # if(df[key]<=0):
        #     print(key)
        #     print(df[key])
        idf[key]=numpy.log2(num_cvs/df[key])
    
    #calculating tf
    for key in cont:
        dict.setdefault(key,{})
        for w in V_tech_stem:
            dict[key][w]=cont[key].count(w.lower())
        if(sourcepath[len(sourcepath)-1]==key):
            tf=dict
    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])

    #calculating tf*idf for each document
    for key in tf:
        wgt.setdefault(key,{})
        for w in V_tech_stem:
            wgt[key][w]=format(tf[key][w]*idf_stem[w],'.3F')
    return idf, wgt

def techJobProc(vacature_technicalskillset, idf, doc):
    wgt={}
    tf={}
    stemmer=[]
    for w in list(vacature_technicalskillset):
        stemmer.append(Pstem.stem(w))
    for w in V_tech_stem:
        tf.setdefault(w,0)
        if w in list(stemmer):
            tf[w]=doc.count(w)
    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])
    for w in V_tech_stem:
        wgt.setdefault(w,0)
        wgt[w]=format(tf[w]*idf_stem[w],'.3F')
    return(wgt)

def cosTech(d_wgt, q_wgt):
    cos={}
    for key in d_wgt:
        temp=[]
        temp1=[]
        for w in q_wgt:
            t=float(d_wgt[key][w])
            t1=float(q_wgt[w])
            temp.append(t)
            temp1.append(t1)           
        cos.setdefault(key,0)
        sum=0
        sumsqt=sumsqt1=0
        for w in range(len(temp)-1):
            sum=sum+temp[w]*temp1[w]
        sqt=[]
        sqt1=[]
        for w in range(len(temp)-1):
            sqt.append(temp[w]*temp[w])
            sqt1.append(temp1[w]*temp1[w])
        for w in range(len(temp)-1):
            sumsqt=sumsqt+sqt[w]
            sumsqt1=sumsqt1+sqt1[w]
        if math.sqrt(sumsqt)*math.sqrt(sumsqt1) ==0:
            cos[key]=0
        else:
            cos[key]=format(sum/(math.sqrt(sumsqt)*math.sqrt(sumsqt1)),'.3F')
    return cos

def tfIdfSoftskills(cont):
    DIR = "C:\\Users\\Hp\\Desktop\\6th Semester\\IR\\Resume-Matching-with-JD-master\\cvs_txt\\"
    sourcepath = os.listdir(DIR)
    num_cvs=len(sourcepath)
    wgt={}
    df={}
    idf={}
    tf={}
    dict={}
    index_soft_stem={}

    for key in index_soft:
        index_soft_stem.setdefault(key,[])
        for w in index_soft[key]:
            index_soft_stem[key].append(Pstem.stem(str(j)))

    #calculating df
    for w in V_soft:
        df.setdefault(w,0)
        for key in index_soft:
            if(w in index_soft[key]):
                df[w]+=1
    
    #calculating idf
    for key in df:
        idf.setdefault(key,0)
        if(df[key]<=0):
            print(key)
            print(df[key])
        idf[key]=numpy.log2(num_cvs/df[key])

    #calculating tf
    for key in cont:
        dict.setdefault(key,{})
        for w in V_soft_stem:
            dict[key][w]=cont[key].count(w)
        
        if(sourcepath[len(sourcepath)-1]==key):
            tf=dict

    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])
    # print(tf)
    #calculating tf*idf for each document
    for key in tf:
        wgt.setdefault(key,{})
        for w in V_soft_stem:
            wgt[key][w]=format(tf[key][w]*idf_stem[w],'.3F')
    # print(wgt)
    return idf, wgt

def cosSoftskills(d_wgt, q_wgt):
    cos={}
    for key in d_wgt:
        temp=[]
        temp1=[]
        for w in q_wgt:
            t=float(d_wgt[key][w])
            t1=float(q_wgt[w])
            # if(t1==0):
                # t1=0.000000001
            # if(t==0):
            #     t=0.000000001
            temp.append(t)
            temp1.append(t1)      
        # print('temp',temp)    
        # print('temp1',temp1)    
        cos.setdefault(key,0)
        sum=0
        sumsqt=sumsqt1=0
        for w in range(len(temp)-1):
            sum=sum+temp[w]*temp1[w]
        # print('mul',sum)
        sqt=[]
        sqt1=[]
        for w in range(len(temp)-1):
            sqt.append(temp[w]*temp[w])
            sqt1.append(temp1[w]*temp1[w])
        for w in range(len(temp)-1):
            sumsqt=sumsqt+sqt[w]
            sumsqt1=sumsqt1+sqt1[w]
        # print('sumsqt',sumsqt)
        # print('sumsqt1',sumsqt1)
        if math.sqrt(sumsqt)*math.sqrt(sumsqt1) ==0:
            cos[key]=0
        else:
            cos[key]=format(sum/(math.sqrt(sumsqt)*math.sqrt(sumsqt1)),'.3F')
        # cos[key]=format(1-spatial.distance.cosine(temp,temp1),'.3F')
    # print(cos)
    return cos

def softkillsJobProc(vacature_softskillset, idf, doc):
    wgt={}
    tf={}
    stemmer=[]
    for w in list(vacature_softskillset):
        stemmer.append(Pstem.stem(w))
    for w in V_soft_stem:
        tf.setdefault(w,0)
        if w in list(stemmer):
            tf[w]=doc.count(w)
    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])
    for w in V_soft_stem:
        wgt.setdefault(w,0)
        wgt[w]=format(tf[w]*idf_stem[w],'.3F')
    return wgt

def combineScore(cos_tech, cos_soft, ed_high):
    total={}
    for key in cos_tech:
        total.setdefault(key,0)
        total[key]=float(cos_soft[key])+float(cos_tech[key])
    #######################
    for key in ed_high:
        total[key]=total[key]+ed_high[key]
    #######################
    return total

def displayResult(total):
    final = {}
    rejected = {}
    alpha = 0.3

    # Assuming total is a dictionary containing CV scores

    for key in total:
        if total[key] > alpha:
            final[key] = total[key]
        else:
            rejected[key] = total[key]

    # Sorting final dictionary based on values
    final = sorted(final.items(), key=operator.itemgetter(1), reverse=True)

    print('ACCEPTED CVs')
    for key, value in final:
        print(f'\t{key}\t{value:.3f}')

    print('REJECTED CVs')
    for key, value in rejected.items():
        print(f'\t{key}\t{value:.3f}')

    acc = [[key, value] for key, value in final]
    rej = [[key, value] for key, value in rejected.items()]

    # Setting up GUI
    # ctk.set_appearance_mode("System")
    # ctk.set_default_color_theme("green")
    # root = ctk.CTk()
    # root.geometry('880x550')
    # root.title("JOB DESCRIPTION AND RESUME MATCHING")

    # # Printing accepted CVs
    # label = ['Accepted Resumes', 'Scores']
    # acc_df = pd.DataFrame(acc, columns=label)
    # acc_text = acc_df.to_string(index=False)
    # acc_heading = ctk.CTkLabel(root, text="ACCEPTED CVs", font=("Helvetica", 16, "bold"))
    # acc_heading.pack()
    # acc_text_widget = ctk.CTkLabel(root, text=acc_text)
    # acc_text_widget.pack()

    # # Printing rejected CVs
    # label = ['Rejected Resumes', 'Scores']
    # rej_df = pd.DataFrame(rej, columns=label)
    # rej_text = rej_df.to_string(index=False)
    # rej_heading = ctk.CTkLabel(root, text="REJECTED CVs", font=("Helvetica", 16, "bold"))
    # rej_heading.pack()
    # rej_text_widget = ctk.CTkLabel(root, text=rej_text)
    # rej_text_widget.pack()

    # root.mainloop()
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("green")
    root = ctk.CTk()
    root.title("Resume Matching Results")

    frame = ctk.CTkFrame(root)
    frame.pack(padx=10, pady=10)

    accepted_label =ctk.CTkLabel(frame, text="Accepted Resumes", font=('Arial', 14))
    accepted_label.grid(row=0, column=0, padx=10, pady=5)
    rejected_label = ctk.CTkLabel(frame, text="Rejected Resumes", font=('Arial', 14))
    rejected_label.grid(row=0, column=1, padx=10, pady=5)

    accepted_text = Text(frame, height=20, width=70)
    accepted_text.grid(row=1, column=0, padx=10, pady=5)
    rejected_text = Text(frame, height=20, width=70)
    rejected_text.grid(row=1, column=1, padx=10, pady=5)

    # for resume, score in final_rank.items():
    #     if score >= 0.2:  # Adjust the threshold as needed
    #         accepted_text.insert(END, f"{resume}: {score}\n")
    #     else:
    #         rejected_text.insert(END, f"{resume}: {score}\n")
    for resume, score in acc:
        accepted_text.insert(END, f"{resume}: {score}\n")
    for resume, score in rej:
        rejected_text.insert(END, f"{resume}: {score}\n")

    root.mainloop()

def proc():
    cont=index()                 #Preprocessing of resumes
    addPattern()          #Read training dataset
    Tk().withdraw()                     #we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  
    doc=queryIndex(filename)
    return cont,filename,doc

def main():
    cont,filename,doc=proc()
    print(filename)
    print(doc)
    resume_texts, resume_names = tokenizeText()          #Return Resumes content with their names
    f2=open(filename,'r')
    vacature_text=f2.read().lower()
    softskillset_dict, technicalskillset_dict, education_dict = makeSkillset(resume_names, resume_texts)

    global V_soft
    temp=[]
    for w in V_soft:
        if w not in temp:
            temp.append(w)
    V_soft=temp
    global V_soft_stem
    for w in V_soft:
        V_soft_stem.append(Pstem.stem(w))

    global V_tech
    temp=[]
    for w in V_tech:
        if w not in temp:
            temp.append(w)
    V_tech=temp
    global V_tech_stem
    for w in V_tech:
        V_tech_stem.append(Pstem.stem(w))

    idf_tech, wgt_tech=tfIdfTech(cont)

    idf_soft, wgt_soft=tfIdfSoftskills(cont)

    vacature_softskillset = makeSoftskillsJob(nlp(vacature_text))

    vacature_technicalskillset = makeTechskillsJob(nlp(vacature_text))

    # vacature_education = makeEduSet(nlp(vacature_text))

    tech_JD_wgt=techJobProc(vacature_technicalskillset, idf_tech, doc)

    cos_tech=cosTech(wgt_tech, tech_JD_wgt)

    soft_JD_wgt=softkillsJobProc(vacature_softskillset, idf_soft, doc)

    cos_soft=cosSoftskills(wgt_soft, soft_JD_wgt)

    ed_high=assessEdu(education_dict)
    
    total=combineScore(cos_tech, cos_soft, ed_high)    

    displayResult(total)

main()