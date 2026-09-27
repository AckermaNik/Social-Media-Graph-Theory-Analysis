import csv
from collections import *
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from networkx.algorithms.community import girvan_newman
import random


def bar_plotting(df):
    
    try:
       
        num_rows = len(df)
        print(f"Number of rows: {num_rows}")
        fig,ax = plt.subplots()
        
        value_counts = df['Platform'].value_counts()
        
        # Extract the top 3 occurrences
        top_3_counts = value_counts.head(3).tolist()
        names=value_counts.head(3).index # to get the names of the most 3 counted socials
       
        social=[names[0],names[1],names[2]]
        counts=[top_3_counts[0],top_3_counts[1],top_3_counts[2]] # number of occurances
        bar_colors=['red','orange','blue']
        bar_labels = ['tab:red', 'tab:orange', 'tab:blue']


        # bar plot
        ax.bar(social, counts, label=bar_labels, color=bar_colors)
        ax.set_ylabel('Usage')
        ax.set_title('Top 3 Dominant Social Media')


        #pie chart for ages
        bins = [ 15, 30, 50, float('inf')]
        labels = [ '15-30', '30-50', '50+']

        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df = df.dropna(subset=['Age'])
        df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        
        # counting the number of occurrences in each age group
        age_group_counts = df['Age'].value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Age Distribution')
           
        
        # pie chart for dominant emotions
        emotion_counts = df['Dominant_Emotion'].value_counts()
        
        explode = [0.1] * len(emotion_counts)
        explode[0] = 0.2
        
        plt.figure(figsize=(8, 8))
        plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, explode=explode)
        plt.title('Distribution of Dominant Emotions')
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
        
def data_analysis(df):
    
    min_age= df['Age'].min()
    max_age=df['Age'].max()
    min_comments= df['Comments_Received_Per_Day'].min()
    max_comments= df['Comments_Received_Per_Day'].max()
    min_msg= df['Messages_Sent_Per_Day'].min()
    max_msg= df['Messages_Sent_Per_Day'].max()
    min_likes=df['Likes_Received_Per_Day'].min()
    max_likes=df['Likes_Received_Per_Day'].max()
    min_posts=df['Posts_Per_Day'].min()
    max_posts=df['Posts_Per_Day'].max()
    min_usage=df['Daily_Usage_Time'].min()
    max_usage=df['Daily_Usage_Time'].max()
    
    print("min age: \n",min_age)
    print("max age: \n",max_age)
    print("min_comments: \n",min_comments)
    print("max_comments: \n",max_comments)
    print("min_msg: \n",min_msg)
    print("max_msg: \n",max_msg)
    print("min_likes: \n",min_likes)
    print("max_likes: \n",max_likes)
    print("min_posts: \n",min_posts)
    print("max_posts: \n",max_posts)
    print("min_usage: \n",min_usage)
    print("max_usage: \n",max_usage)
    
        # Mean
    print("Mean:")
    print(df.mean(numeric_only=True))

    # Variance
    print("\nVariance:")
    print(df.var(numeric_only=True))

    # Standard Deviation
    print("\nStandard Deviation:")
    print(df.std(numeric_only=True))
        
    try:

        G = nx.Graph()

        # add platforms as nodes
        platforms = df['Platform'].unique()
        G.add_nodes_from(platforms, node_type='platform')

        # add user nodes and connect them to their platform
        for _, row in df.iterrows():
            user = row['User_ID']
            platform = row['Platform']
            emotion = row['Dominant_Emotion']

            G.add_node(user, node_type='user')
            G.add_edge(user, platform, emotion=emotion)


        plt.figure(figsize=(12, 8))

        pos = nx.spring_layout(G)

        
        platform_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'platform']
        user_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'user']

        pos = {}
        # position platform nodes in a circle
        num_platforms = len(platforms)
        radius = 30  # Distance of the platforms from the center
        angle_step = 2 * math.pi / num_platforms

        platform_positions = {}
        for i, platform in enumerate(platforms):
            angle = i * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            platform_positions[platform] = (x, y)
            pos[platform] = (x, y)

        # Position user nodes around their respective platform in a smaller circle
        user_radius = 10  # Radius for users around each platform
        for platform, (px, py) in platform_positions.items():
            users = df[df['Platform'] == platform]['User_ID'].tolist()
            num_users = len(users)
            user_angle_step = 2 * math.pi / num_users if num_users > 0 else 0

            for j, user in enumerate(users):
                user_angle = j * user_angle_step
                ux = px + user_radius * math.cos(user_angle)
                uy = py + user_radius * math.sin(user_angle)
                pos[user] = (ux, uy)

        
        # Draw platform nodes
        nx.draw_networkx_nodes(G, pos, nodelist=platform_nodes, node_size=1000, node_color='skyblue', label='Platforms')

        # Draw user nodes
        nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_size=500, node_color='lightgreen', label='Users')

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.2)

        # Add labels for platforms and users
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Add edge labels for emotions
        edge_labels = nx.get_edge_attributes(G, 'emotion')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',label_pos=0.5, font_size=6)

        plt.title("Clustered Graph of Users and Platforms with Dominant Emotions", fontsize=14)
        plt.legend(scatterpoints=1, loc='best')
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    

def analyze_emotion_homophily_by_activity(df):
    
    try: 
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        behavior_features = ['Comments_Received_Per_Day', 'Likes_Received_Per_Day', 'Posts_Per_Day','Messages_Sent_Per_Day']
        data_scaled = scaler.fit_transform(df[behavior_features])

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(data_scaled)

        # Create a graph with a similarity threshold
        threshold = 0.9
        G = nx.Graph()
        for i, user1 in enumerate(df['User_ID']):
            for j, user2 in enumerate(df['User_ID']):
                if user1 != user2 and similarity_matrix[i, j] > threshold and not G.has_edge(user2,user1):
                    G.add_edge(user1, user2) # user1 = 528 (just the code)
                    
        #clustering = nx.clustering(G)
        avg_clustering = nx.average_clustering(G)
        print(f"Average Clustering Coefficient: {avg_clustering}")
        
        communities = next(girvan_newman(G))
        print("Detected Communities:", len( list(communities[1])))
        
        
        color_map = {'Happiness': 'yellow', 'Sadness': 'blue', 'Neutral': 'pink', 'Anger': 'red','Boredom': 'orange','Anxiety': 'brown'}
        node_colors = [color_map[df.loc[df['User_ID'] == node, 'Dominant_Emotion'].values[0]] for node in G.nodes]
        plt.figure(figsize=(10, 10))
        nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray")
        plt.show()
        
        for node in G.nodes:
            #loc -> to access specific rows or columns of a DataFrame.
            emotion_series = df.loc[df['User_ID'] == node, 'Dominant_Emotion']
            if not emotion_series.empty:  # make sure the node exists in the DataFrame
                emotion = emotion_series.values[0]
                G.nodes[node]['Dominant_Emotion'] = emotion 
                
        #for node in G.nodes:
            #print(node, G.nodes[node]['Dominant_Emotion'])
        
        # Calculate e_same
        same_emotion_edges = [
            edge for edge in G.edges
            if detect_same_emotions(G,edge)
        ]
        e_same = len(same_emotion_edges) / len(G.edges) if G.edges else 0

        # Calculate a_same
        emotion_counts = df['Dominant_Emotion'].value_counts(normalize=True)
        a_same = sum(emotion_counts**2)

        # Homophily index
        homophily_index = (e_same - a_same) / (1 - a_same) if (1 - a_same) != 0 else 0
        print(f"Homophily Index: {homophily_index:.2f}")
           
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
def detect_same_emotions(G,edge):
    
    emotion1= G.nodes[edge[0]]['Dominant_Emotion']
    emotion2= G.nodes[edge[1]]['Dominant_Emotion']
    
    if emotion1=='Happiness' or emotion1=='Neutral' or emotion1=='Boredom':
        if emotion2=='Happiness' or emotion2=='Neutral' or emotion2=='Boredom':
            return True
    elif  emotion1=='Sadness' or emotion1=='Anger' or emotion1=='Anxiety':
        if emotion2=='Sadness' or emotion2=='Anger' or emotion2=='Anxiety':
            return True
        
def find_shortest_path_to_emotion(selected_data,emotion):
    
    G = nx.DiGraph()

    G.add_node("Platform")
    
    if emotion=='Happiness':
        selected_data = df[df['Dominant_Emotion'] == 'Happiness']
    else:
        selected_data = df[df['Dominant_Emotion'] == 'Sadness']

    # Likes_Received nodes
    likes_nodes = selected_data['Likes_Received_Per_Day'].unique()
    for like in likes_nodes:
        G.add_node(f"Likes_{like}")
        G.add_edge("Platform", f"Likes_{like}", weight=like)

    # Messages_Sent_Per_Day nodes
    for like in likes_nodes:
        messages_nodes = selected_data['Messages_Sent_Per_Day'].unique()
        for message in messages_nodes:
            G.add_node(f"Messages_{message}")
            G.add_edge(f"Likes_{like}", f"Messages_{message}", weight=message)

    # Comments_Received_Per_Day nodes
    for message in selected_data['Messages_Sent_Per_Day'].unique():
        comments_nodes = selected_data['Comments_Received_Per_Day'].unique()
        for comment in comments_nodes:
            G.add_node(f"Comments_{comment}")
            G.add_edge(f"Messages_{message}", f"Comments_{comment}", weight=comment)

    # Posts_Per_Day nodes
    for comment in selected_data['Comments_Received_Per_Day'].unique():
        posts_nodes = selected_data['Posts_Per_Day'].unique()
        for post in posts_nodes:
            G.add_node(f"Posts_{post}")
            G.add_edge(f"Comments_{comment}", f"Posts_{post}", weight=post)

    # the final "Happiness" node
    if emotion=='Happiness':
        G.add_node("Happiness")
        for post in selected_data['Posts_Per_Day'].unique():
            G.add_edge(f"Posts_{post}", "Happiness", weight=0)
            
        # Find the shortest path using Dijkstra
        try:
            shortest_path = nx.shortest_path(G, source="Platform", target="Happiness", weight="weight")
            print("Shortest path:", shortest_path)
        except nx.NetworkXNoPath:
            print("No path exists between 'Platform' and 'Happiness'.")
    else: 
        G.add_node("Sadness")
        for post in selected_data['Posts_Per_Day'].unique():
            G.add_edge(f"Posts_{post}", "Sadness", weight=0)
            
        # Find the shortest path using Dijkstra
        try:
            shortest_path = nx.shortest_path(G, source="Platform", target="Sadness", weight="weight")
            print("Shortest path:", shortest_path)
        except nx.NetworkXNoPath:
            print("No path exists between 'Platform' and 'Happiness'.")
        

    return G, shortest_path if 'shortest_path' in locals() else None




def edit_dups(df):
    
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    
    df.loc[df['User_ID'] == 651, 'Age'] = 20
    df.loc[df['User_ID'] == 651, 'Gender'] = 'Male'
    
    platform_pool = ['Facebook', 'Twitter', 'Instagram', 'LinkedIn', 'Snapchat','Telegram','Whatsapp']
      
    duplicate_rows = df[df.duplicated(subset='User_ID', keep='first')]
    original_rows = df[~df.duplicated(subset='User_ID', keep='first')]

    for user_id in duplicate_rows['User_ID'].unique():
        
        existing_platforms = original_rows.loc[original_rows['User_ID'] == user_id, 'Platform'].tolist()

        # Pick a new platform from the pool avoiding existing ones
        available_platforms = [p for p in platform_pool if p not in existing_platforms]

        # If there are more duplicates than available platforms, cycle through the pool
        duplicate_indices = duplicate_rows[duplicate_rows['User_ID'] == user_id].index
        for i, index in enumerate(duplicate_indices):
            duplicate_rows.at[index, 'Platform'] = available_platforms[i % len(available_platforms)]
    
    # Combine the updated duplicates and originals back into a single DataFrame
    return pd.concat([original_rows, duplicate_rows]).sort_index()
    
    
def betweenness_centrality(df):
    
    G = nx.Graph()

    for _, row in df.iterrows():
        
        user_id = row['User_ID']
        
        
        if row['Likes_Received_Per_Day'] > 0:
            G.add_edge(user_id, f"Likes_{row['Likes_Received_Per_Day']}", weight=row['Likes_Received_Per_Day'])
            G.nodes[f"Likes_{row['Likes_Received_Per_Day']}"]['Emotion']=row['Dominant_Emotion']
            G.nodes[f"Likes_{row['Likes_Received_Per_Day']}"]['Activity']='Likes_Received_Per_Day'
            G.nodes[f"Likes_{row['Likes_Received_Per_Day']}"]['Score']=row['Likes_Received_Per_Day']
        if row['Comments_Received_Per_Day'] > 0:
            G.add_edge(user_id, f"Comments_{row['Comments_Received_Per_Day']}", weight=row['Comments_Received_Per_Day'])
            G.nodes[f"Comments_{row['Comments_Received_Per_Day']}"]['Emotion']=row['Dominant_Emotion']
            G.nodes[f"Comments_{row['Comments_Received_Per_Day']}"]['Activity']='Comments_Received_Per_Day'
            G.nodes[f"Comments_{row['Comments_Received_Per_Day']}"]['Score']=row['Comments_Received_Per_Day']
        if row['Messages_Sent_Per_Day'] > 0:
            G.add_edge(user_id, f"Messages_{row['Messages_Sent_Per_Day']}", weight=row['Messages_Sent_Per_Day'])
            G.nodes[f"Messages_{row['Messages_Sent_Per_Day']}"]['Emotion']=row['Dominant_Emotion']
            G.nodes[f"Messages_{row['Messages_Sent_Per_Day']}"]['Activity']='Messages_Sent_Per_Day'
            G.nodes[f"Messages_{row['Messages_Sent_Per_Day']}"]['Score']=row['Messages_Sent_Per_Day']
        if row['Posts_Per_Day'] > 0:
            G.add_edge(user_id, f"Posts_{row['Posts_Per_Day']}", weight=row['Posts_Per_Day'])
            G.nodes[f"Posts_{row['Posts_Per_Day']}"]['Emotion']=row['Dominant_Emotion']
            G.nodes[f"Posts_{row['Posts_Per_Day']}"]['Activity']='Posts_Per_Day'
            G.nodes[f"Posts_{row['Posts_Per_Day']}"]['Score']=row['Posts_Per_Day']
        G.nodes[user_id]['Emotion']=row['Dominant_Emotion']
        G.nodes[user_id]['Activity']='User'
    
    
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    
    centrality_values = list(betweenness.values()) 
    
    #75% of all centrality values
    percentile_75 = np.percentile(centrality_values, 75)
    #print(percentile_75)
    
    # nodes with centrality value > percentile_75
    most_important_nodes = {node: centrality for node, centrality in betweenness.items() if centrality > percentile_75 and node['Activity']!='User'}
    top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:len(most_important_nodes)]
    
    poisitive_emotions = ['Neutral', 'Happiness', 'Boredom']
    negative_emotions = ['Sadness', 'Anger', 'Anxiety']

    for node, _ in top_nodes:
        positive_score=0
        negative_score=0
        
        for _, row in df.iterrows():
            if row[node['Activity']] == node['Score']:
                if row['Dominant_Emotion'] in poisitive_emotions:
                    positive_score+=1
                else: negative_score+=1
        if positive_score>=negative_score:
            node['Majority']='Positive'
        else: node['Majority']='Negative' 
            
    
    total_positive_score=0
    total_negative_score=0
    
    for node, _ in top_nodes:    
        if  node['Majority']=='Positive':
            total_positive_score+=1
        else: total_negative_score+=1          
          
    if total_positive_score>=total_negative_score:
        print("Most important/regular activities hold Positive emotions for users")            
    else: print("Most important/regular activities hold Negative emotions for users")            

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_color="lightblue", 
        edge_color="gray", 
        node_size=700, 
        font_size=10
    )
    plt.title("User-Content Interaction Graph")
    plt.show()

    return betweenness


def bipartite_ages(df):
    age_groups = ["20-24", "25-30", "30-36"]
    social_media = ["Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "Whatsapp", "Telegram"] 
    
    B = nx.Graph()
    
    colors = ['green', 'orange','purple']

    B.add_nodes_from(age_groups, bipartite=0)
    B.add_nodes_from(social_media, bipartite=1)
    
    matching_socials=[]
    matching_computation(df,20,24,B,social_media,age_groups,matching_socials)
    
    
    index=0
    for i in matching_socials:
        sub_list=matching_socials[index]
        for j in sub_list:
            B.add_edge(age_groups[index], j ,color=colors[index])
        index+=1
        
    # Get positions for a bipartite layout
    pos = nx.drawing.layout.bipartite_layout(B, nodes=age_groups)

    # Draw the nodes
    nx.draw_networkx_nodes(B, pos, nodelist=age_groups, node_color="lightblue", label="Age Groups", node_size=1500)
    nx.draw_networkx_nodes(B, pos, nodelist=social_media, node_color="lightgreen", label="Social Media", node_size=1500)

    # Draw the edges
    edge_colors = [B[u][v]['color'] for u, v in B.edges()]
    nx.draw_networkx_edges(B, pos, edge_color=edge_colors)

    # Add labels
    nx.draw_networkx_labels(B, pos, font_size=10, font_color="black")


    plt.legend(["Age Groups", "Social Media"])
    plt.title("Social Media Preferences by Age Groups to feel Positive Emotions")
    plt.axis("off")  # Turn off axes
    plt.show()




    # each age fluctuation is a stack frame
def matching_computation(df,age1,age2,B,social_media,ages,matching_socials):
    

    if (age2>36):
        return
    
    max_counts=[-1] * 7 # platform for each age in order as it is in social_media list
    
    for _, row in df.iterrows():
        
        age = int(row['Age'])
        emotion1=row['Dominant_Emotion']
        platform=row['Platform']
        
        if age>=age1 and age<=age2:
            
            ##########################################
            #
            # Change the if statement below by adding/cutting down emotions
            # according to what emotions we want to examine
            #
            ##########################################
            
            if emotion1=='Anger': #or emotion1=='Neutral'or emotion1=='Boredom':
                    max_counts[social_media.index(platform)]+=1
                
    max_value = max(max_counts)
    
    if max_counts==-1:
        indices_of_max=[random.randint(0, 6)]
    else:       
        # get all indices with the maximum value
        indices_of_max = [i for i, x in enumerate(max_counts) if x == max_value]
                    
    matching_socials.append([social_media[index] for index in indices_of_max ])
    matching_computation(df,age1+5,age2+6,B,social_media,ages,matching_socials) 

    
if __name__ == "__main__":
    
    try:
            df = pd.read_csv("archive/test.csv")
            #print(df['Age'])
          
            df=edit_dups(df)
            #bar_plotting(df)
            data_analysis(df)
            #analyze_emotion_homophily_by_activity(df)
        
            #graph, shortest_path = find_shortest_path_to_emotion(df,'Happiness')            
            #graph, shortest_path = find_shortest_path_to_emotion(df,'Sadness')
            
            #betweenness_centrality(df)
            
            #bipartite_ages(df)
                     
    except FileNotFoundError:
        print(f"File not found.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
