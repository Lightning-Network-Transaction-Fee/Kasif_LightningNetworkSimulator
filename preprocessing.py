import numpy as np
import networkx as nx
import pandas as pd
import math
import json
from tqdm import tqdm
import copy


def load_temp_data(json_files, node_keys=["pub_key","last_update"], edge_keys=["node1_pub","node2_pub","last_update","capacity"]):
    """Load LN graph json files from several snapshots"""
    node_info, edge_info = [], []
    for idx, json_f in enumerate(json_files):
        with open(json_f) as f:
            try:
                tmp_json = json.load(f)
            except json.JSONDecodeError:
                print("JSONDecodeError: " + json_f)
                continue
        new_nodes = pd.DataFrame(tmp_json["nodes"])[node_keys]
        new_edges = pd.DataFrame(tmp_json["edges"])[edge_keys]
        new_nodes["snapshot_id"] = idx
        new_edges["snapshot_id"] = idx
        print(json_f, len(new_nodes), len(new_edges))
        node_info.append(new_nodes)
        edge_info.append(new_edges)
    edges = pd.concat(edge_info)
    edges["capacity"] = edges["capacity"].astype("int64")
    edges["last_update"] = edges["last_update"].astype("int64")
    print("All edges:", len(edges))
    edges_no_loops = edges[edges["node1_pub"] != edges["node2_pub"]]
    print("All edges without loops:", len(edges_no_loops))
    return pd.concat(node_info), edges_no_loops

def generate_directed_graph(edges, policy_keys=['disabled', 'fee_base_msat', 'fee_rate_milli_msat', 'min_htlc']):
    """Generate directed graph data from undirected payment channels."""
    directed_edges = []
    indices = edges.index
    for idx in tqdm(indices):
        row = edges.loc[idx]
        e1 = [row[x] for x in ["snapshot_id","node1_pub","node2_pub","last_update","channel_id","capacity"]]
        e2 = [row[x] for x in ["snapshot_id","node2_pub","node1_pub","last_update","channel_id","capacity"]]
        if row["node2_policy"] == None:
            e1 += [None for x in policy_keys]
        else:
            e1 += [row["node2_policy"][x] for x in policy_keys]
        if row["node1_policy"] == None:
            e2 += [None for x in policy_keys]
        else:
            e2 += [row["node1_policy"][x] for x in policy_keys]
        directed_edges += [e1, e2]
    cols = ["snapshot_id","src","trg","last_update","channel_id","capacity"] + policy_keys
    directed_edges_df = pd.DataFrame(directed_edges, columns=cols)
    return directed_edges_df

def preprocess_json_file(json_file):
      """Generate directed graph data (traffic simulator input format) from json LN snapshot file."""
      json_files = [json_file]
      print("\ni.) Load data")
      EDGE_KEYS = ["node1_pub","node2_pub","last_update","capacity","channel_id",'node1_policy','node2_policy']
      nodes, edges = load_temp_data(json_files, edge_keys=EDGE_KEYS)
      print(len(nodes), len(edges))
      print("Remove records with missing node policy")
      print(edges.isnull().sum() / len(edges))
      origi_size = len(edges)
      edges = edges[(~edges["node1_policy"].isnull()) & (~edges["node2_policy"].isnull())]
      print(origi_size - len(edges))
      print("\nii.) Transform undirected graph into directed graph")
      directed_df = generate_directed_graph(edges)
      """
          checking the capacity distrubutaions:---->
          print('directed_edges_df')
          print(directed_df.columns)
          print('src 0:',directed_df['src'][0],'trg 0:',directed_df['trg'][0])
          print('src 1:',directed_df['src'][1],'trg 1:',directed_df['trg'][1])
          print('cap:',directed_df['capacity'][0],'onyeki',directed_df['capacity'][1])
          print('cap:',directed_df['capacity'][2],'onyeki',directed_df['capacity'][3])
      """

      print("\niii.) Fill missing policy values with most frequent values")
      print("missing values for columns:")
      print(directed_df.isnull().sum())
      directed_df = directed_df.fillna({"disabled":False,"fee_base_msat":1000,"fee_rate_milli_msat":1,"min_htlc":1000})
      for col in ["fee_base_msat","fee_rate_milli_msat","min_htlc"]:
          directed_df[col] = directed_df[col].astype("float64")
      return nodes,directed_df
    
    
def init_node_params(edges, providers, verbose=True):
    """Initialize source and target distribution of each node in order to drawn transaction at random later."""
    G = nx.from_pandas_edgelist(edges, source="src", target="trg", edge_attr=["capacity"], create_using=nx.DiGraph())
    active_providers = list(set(providers).intersection(set(G.nodes())))
    active_ratio = len(active_providers) / len(providers)
    if verbose:
        print("Total number of possible providers: %i" % len(providers))
        print("Ratio of active providers: %.2f" % active_ratio)
    degrees = pd.DataFrame(list(G.degree()), columns=["pub_key","degree"])
    total_capacity = pd.DataFrame(list(nx.degree(G, weight="capacity")), columns=["pub_key","total_capacity"])
    node_variables = degrees.merge(total_capacity, on="pub_key")
    #get_src_rayleigh_proba(node_variables)
    #get_trg_proba(node_variables, eps=0.95, active_providers)
    return node_variables, active_providers, active_ratio
  
  
 


def initiate_balances(directed_edges) :
    G = directed_edges[['src','trg','channel_id','capacity','fee_base_msat','fee_rate_milli_msat']]
    G = G.assign(balance = None)
    r = 0
    for index,row in G.iterrows():
        balance = 0
        cap = row['capacity']
        if index%2==0 :
            r = np.random.random()
            balance = r*cap
        else :
            balance = (1-r)*cap
        G.at[index,"balance"] = balance
    return G


def set_node_balance(G,src,trg,channel_id,capacity,initial_balance):
    index = G.index[(G['channel_id']==channel_id)]
    G.at[index[0],'capacity'] = capacity
    G.at[index[1],'capacity'] = capacity
    G.at[index[0],'balance'] = initial_balance
    G.at[index[1],'balance'] = capacity - initial_balance
    return G
