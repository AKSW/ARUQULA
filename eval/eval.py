import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pySankey import sankey
import matplotlib.colors as mcolors
from scipy.stats import ttest_ind
import os

df = pd.read_json("./stats_raw.json", orient='index')

# for label in [ "corporate-en", "dbpedia-en", "dbpedia-es" ]:
#     print(df["dataset_label" == label])

aggregate = df.groupby("dataset_label").agg({
    "duration": [ np.mean, np.std ],
    "number_of_agent_steps": [ np.mean, np.std ]
})

print(aggregate.to_latex())

print("T Tests (independent):")


print("[*] steps")
steps_dbpedia_en = df[df["dataset_label"] == "dbpedia-en" ]["number_of_agent_steps"].reset_index(drop=True)
steps_corporate_en = df[df["dataset_label"] == "corporate-en" ]["number_of_agent_steps"].reset_index(drop=True)
print( ttest_ind(steps_dbpedia_en, steps_corporate_en, equal_var=False) )

print("[*] durations")
times_dbpedia_en = df[df["dataset_label"] == "dbpedia-en" ]["duration"].reset_index(drop=True)
times_corporate_en = df[df["dataset_label"] == "corporate-en" ]["duration"].reset_index(drop=True)
print( ttest_ind(times_dbpedia_en, times_corporate_en, equal_var=False) )


df['actions'] = df['actions'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
max_actions = 16
actions_df = pd.DataFrame(df['actions'].to_list(), columns=[f'{i+1}' for i in range(max_actions)])
actions_df = actions_df.map( lambda a: a.replace("_by_label","") if a is not None else None )


action_props = pd.DataFrame({
    col: actions_df[col].value_counts(normalize=False) for col in actions_df.columns
})[[str(i) for i in range(1,16)]]




action_colors_dict = {
    'execute_sparql': 'indigo',
    'get_knowledgegraph_entry': 'goldenrod',
    'get_property_examples': 'greenyellow',
    'search_class': 'maroon',
    'search_entity': 'teal',
    'search_property': 'plum',
    'stop': 'lightsteelblue'
}

keys = list(action_colors_dict.keys())
action_color_list = []
for i in keys:
    action_color_list.append(action_colors_dict[i])


ax = action_props.transpose().plot.bar(stacked=True, color= action_color_list)
ax.legend(bbox_to_anchor=(0.9,-0.1), ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.title("Absolute frequencies of actions at each agent step")
ax.figure.savefig("absolute_freqs.png")

stops_df_list = []
for i in range(14):
    stops_df_list.append( actions_df[actions_df[str(i+2)] == "stop"][str(i+1)] )

before_stops_df = pd.concat(stops_df_list)
print(before_stops_df.value_counts())

execute_df_list = []
for i in range(14):
    execute_df_list.append( actions_df[actions_df[str(i+1)] == "execute_sparql"][str(i+2)] )
execute_df = pd.concat(execute_df_list).value_counts()

plt.figure()
execute_df.plot.pie(shadow=True, colors= action_color_list, startangle=90, explode=(0.08,0.08,0,0,0,0,0), labels=None, ylabel="", autopct=lambda x: f"{x:.2f}%" if x > 5 else "")
plt.legend(bbox_to_anchor=(0.3,0.5), labels=execute_df.index)
plt.title("Ratios of actions that follow execute_sparql")
plt.tight_layout()
plt.savefig("after_execute.pdf", format="pdf")


search =  { "label": "search", "functions": [ "search_property", "search_entity", "search_class" ] }
inspect = { "label": "inspect", "functions": [ "get_knowledgegraph_entry", "get_property_examples" ] }
execute = { "label": "execute", "functions": [ "execute_sparql" ] }
stop = { "label": "stop", "functions": [ "stop" ] }

fig, ax = plt.subplots()
for action_class in [ search, inspect, execute, stop ] :
    actions_df.map(lambda x: 1 if x in action_class["functions"] else 0).apply(sum).cumsum().plot(label=action_class["label"],xlim=(0,14),ylim=(0,1100),xlabel="Number of agent steps",ylabel="Cumulative calls",ax=ax)

plt.legend()
plt.grid()
plt.savefig("total.pdf", format="pdf")