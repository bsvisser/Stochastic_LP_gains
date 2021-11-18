# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:29:13 2021
@author: Brent Visser
"""
import streamlit as st
import numpy as np
import pandas as pd
import random
import altair as alt
from matplotlib import pyplot as plt

apex = False

st.set_option('deprecation.showPyplotGlobalUse', False)
ranks = ["Iron","Bronze", "Silver", "Gold", "Platinum", "Diamond", "Apex"]
divs = ["IV","III", "II", "I"] 

colors = ["#5c5653", "#945639", "#4a5a5b", "#b5732c", "#4e7875", "#54519d", "#B71DC9"]

st.title('Monte-Carlo simulation for climbing')
settings_expander = st.beta_expander("Settings for simulation")

left_column, right_column = st.beta_columns(2)
startl = left_column.add_selectbox = st.selectbox(
    "Starting league",
    (ranks)
)
startd = right_column.add_selectbox = st.selectbox(
    "Starting div",
    (divs)
)

"Ending point"
left_column, right_column = st.beta_columns(2)
endl = left_column.add_selectbox = st.selectbox(
    "Goal league",
    (ranks)
)
if endl == "Apex":
    endd = float(st.text_input("End LP", 100))
    apex = True
    
else:
    endd = right_column.add_selectbox = st.selectbox(
        "Goal div",
        (divs),
    )
with settings_expander:
    buff, col, buff2 = st.beta_columns([1,1,1])
    gain = buff.text_input("LP gain per win", 20)
    loss = col.text_input("LP decrease per loss", 15)
    wr = buff2.text_input("Est. winrate (i.e. 0.5 for a 50% wr)", 0.52)
    n = st.slider(
        "Number of trials", 1, 5000, 750
    )

try:
    gain, loss, wr = [float(i) for i in [gain, loss, wr]]
except:
    st.error("Only numbers please") 
    st.stop() 

if wr <= 0 or gain <= 0:
    st.markdown("<b>Even you can't be this bad.</b>", unsafe_allow_html=True)
    st.stop() 
else:
    wr= float(wr)

if wr > 1:
    st.markdown("<b>Please fill in a value smaller than 1</b>", unsafe_allow_html=True)
    st.stop()
    
if np.sign(loss) == -1:
    loss = -1*loss

if (startl, startd) == (endl, endd):
    st.markdown("<b>Your goal can't be where you're starting</b>", unsafe_allow_html=True)
    st.stop() 

if ranks.index(startl) > ranks.index(endl):
    st.markdown("<b>Your goal can't be lower than where you're starting</b>", unsafe_allow_html=True)
    st.stop() 
    
if apex:
    if type(startd) == int and startd < endd:
        st.markdown("<b>Your goal can't be lower than where you're starting</b>", unsafe_allow_html=True)
        st.stop()
    
else:
    if divs.index(startd) > divs.index(endd) and startl==endl:
        st.markdown("<b>Your goal can't be lower than where you're starting</b>", unsafe_allow_html=True)
        st.stop()
    
def stochastic_sim(startl, startd, endl, endd, gain, loss, n, wr):
    apex = False
    df_list = []
    wr = float(wr)
    for i in range(n):
        progress_bar.progress(i/(n-1))
        belowRank = True
        lp = 0
        curRank = [startl,startd]
        goalRank = [endl, endd]
        ext_lp_hist = [ranks.index(curRank[0])*400+divs.index(curRank[1])*100]
        games = 0
        while belowRank:
            #print(games, lp, curRank[0], curRank[1])
            games+=1
            result = random.choices([True, False], weights = [wr, 1-wr])
            if result[0] == False: #loss
                lp-=loss
                if lp < 0: #demote
                    if divs.index(curRank[1]) > 0: #division de
                        curRank[1] = divs[divs.index(curRank[1])-1]
                        lp = 100+lp
                    else: #league de
                        if curRank == ["Iron", "IV"]:
                            lp=0
                           
                        else:
                            if random.choices([True, False], weights = [0.1, 0.9]):
                                curRank[1] = divs[-1]
                                curRank[0] = ranks[ranks.index(curRank[0])-1]
                                lp = 100+lp
                            else:
                                lp = 0
                            
                            
                else: #don't have to do anything, lp is already set
                    pass
            else:
                lp+=gain
                if curRank[0] == "Apex":
                    if  lp > endd:
                        belowRank = False
                if lp > 100:
                    if divs.index(curRank[1]) < (len(divs)-1): #division pro
                        lp = lp-100
                        curRank[1] = divs[divs.index(curRank[1])+1]
                        
                    else: #league promo
                        promo_results = random.choices([True, False], weights = [wr, 1-wr], k =5)
                        if (sum(promo_results) >=3): #won promo
                            games+=sum(promo_results)
                            if curRank[0] == "Diamond" and curRank[1] == 1:
                                curRank[0] = "Apex"
                                lp = 1
                                
                            if curRank[0] == "Apex":
                                pass
                            else:
                                curRank[0] = ranks[ranks.index(curRank[0])+1]
                                curRank[1] = divs[0]
                                lp = 0
                                
                            
                        else: #loss promo
                            lp -=loss
                        if sum(promo_results[0:3]) ==3: #count games
                            games+=3
                        elif sum(promo_results[0:4]) ==3:
                            games+=4
                        else:
                            games+=5
                
                #check to see if goal reached:
                if curRank[0] == goalRank[0] and curRank[1] == goalRank[1]:
                    belowRank = False
                    
            ext_lp = lp + ranks.index(curRank[0])*400+divs.index(curRank[1])*100
            ext_lp_hist.append(ext_lp)
        df_list.append([games,ext_lp_hist])
        df= pd.DataFrame(df_list,columns = ["Games", "ext_LP"])
    return df

alphaline = np.e**-(n/800)
if alphaline < 0.05:
    alphaline = 0.05
    
if st.button('Run it down'):

    progress_bar = st.progress(0)
    f"Trying not to int {n} times in a row"
    df= stochastic_sim(startl, startd, endl, endd, gain, loss, n , wr)
    average = round(np.average(df["Games"]))
    one, two, three = st.beta_columns([1,1,1])
    one.markdown(f"<center>Average # games <b>{average}</b></center>", unsafe_allow_html=True)
    two.markdown(f"<center>Max # games <b>{df['Games'].max()}</b></center>", unsafe_allow_html = True)
    three.markdown(f"<center>Min # games <b>{df['Games'].min()}</b></center>", unsafe_allow_html = True)
    hist = alt.Chart(df).mark_bar().encode(x = 'Games', 
                                                 y = 'count()').properties(height=250, width=670,).interactive()
    hist
    
    st.markdown("<center>Each trial shown as a line</center>", unsafe_allow_html=True)
    progress_bar2 = st.progress(0)
    df['ext_LP'] = df['ext_LP'].apply(np.array)
    df_a = pd.DataFrame([pd.Series(x) for x in df.ext_LP]).transpose()
    plt.plot(df_a.iloc[0:len(df_a)], alpha = alphaline, color= "black", linewidth=0.4)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlim(left=0)
    plt.ylim([ymin-20, None])
    round_ymin = np.round(ymin/400)*400
    round_ymax = np.ceil(ymax/400)*400
    if np.sign(round_ymin) == -1:
        round_ymin = 0
    num_regions = int((round_ymax-round_ymin)/400)
    
    for i in range(num_regions):
        plt.hlines(y=round_ymin+i*400, xmin=0, xmax=xmax, colors=colors[int((round_ymin+i*400)//400)], linestyles='--', lw=1, label=ranks[int((round_ymin+i*400)//400)], alpha = 0.9)
        plt.text(xmax*0.99, round_ymin+i*400+400*0.05, f"{ranks[int((round_ymin+i*400)//400)]}", color = colors[int((round_ymin+i*400)//400)], horizontalalignment='right',)
        plt.axhspan(round_ymin+i*400, round_ymin+i*400+400, facecolor = colors[int((round_ymin+i*400)//400)], alpha=0.2)
        
        
        if i > 2400:
            plt.axhspan(round_ymin+i*400, ymax, facecolor = colors[int((round_ymin+i*400)//400)], alpha=0.2)
            
        progress_bar2.progress((i*0.9)/(num_regions))

    plt.xlabel("Games played")
    plt.ylabel("Total LP")
    st.pyplot(dpi=1200)
    progress_bar2.progress(100)
   
    # line = alt.Chart(source).mark_line().encode(
    # x='x',
    # y='Average').interactive()
    
    
    # df_a = df_a.fillna(df_a.max().max())
    # average = df_a.mean().to_numpy()
    # std = df_a.std()
    
    # x = range(len(average))
    # source = pd.DataFrame({'Average':average,'x':x})
    
    # band = pd.DataFrame({'std':std,'x':x})
    
    # line = alt.Chart(source).mark_line().encode(
    # x='x',
    # y='Average').interactive()

    # band = alt.Chart(source).mark_errorband(extent='ci').encode(
    # x='x',
    # y=alt.Y('Average', title='Average'),)

    # band
    
    # # x = range(len(df["ext_LP"][0]))
    # line= plt.plot(x, df["ext_LP"][0])
    # y_axis = np.arange(min(df["ext_LP"] [0]), max(df["ext_LP"][0]), (max(df["ext_LP"][0])-min(df["ext_LP"][0]))/11)
    # y_labels = []
    # for i in y_axis:    
    #     try:
    #         rank = ranks[int(np.ceil(i/400))]
    #     except:
    #         rank = "Diamond"
    #     y_labels.append(rank)
    # mpl_fig = plt.figure()
    # plt.yticks(y_axis, y_labels)
    # plt.xlabel("Games played")
    # st.pyplot(plt)
    




expander = st.beta_expander("Notes")
expander.write(f"Once a run loses at 0 lp and is at div IV, there is a 10% chance of demoting.")

# st.title('My first app')
# df=  pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# })

# df

# st.line_chart(df)

# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])

#     st.line_chart(chart_data)


# #streamlit run E:/streamlit/first_app.py
