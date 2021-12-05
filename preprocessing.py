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
    
    

  def aggregate_edges(directed_edges):
        """aggregating multiedges"""
        grouped = directed_edges.groupby(["src","trg"])
        directed_aggr_edges = grouped.agg({
            "capacity":"sum",
            "fee_base_msat":"mean",
            "fee_rate_milli_msat":"mean",
            "last_update":"max" ,
            "channel_id":"first" ,
            "disabled":"first",
            "min_htlc":"mean",
        }).reset_index()
        return directed_aggr_edges



  def get_neighbors(G,src,trg,radius):
      """localazing the networke around the edge"""
      neighbors = [src,trg]
      for i in range(radius):
        outer_list = []
        for neighbor in neighbors :
          inner_list = list(G.neighbors(neighbor))
          outer_list += inner_list
        neighbors += outer_list
      return set(neighbors)


  #constant balance
  def initiate_balances(directed_edges) :
        """assigning random balances to the directed channel"""
        G = directed_edges[['src','trg','channel_id','capacity','fee_base_msat','fee_rate_milli_msat']]
        G = G.assign(balance = None)
        r = 0.5
        for index,row in G.iterrows():
            balance = 0
            cap = row['capacity']
            if index%2==0 :
                #r = np.random.random()
                balance = r*cap
            else :
                balance = (1-r)*cap
            G.at[index,"balance"] = balance


        return G
    
    
  
  def set_node_balance(edges,src,trg,channel_id,capacity,initial_balance):
        index = edges.index[(edges['src']==src)&(edges['trg']==trg)]
        reverse_index = edges.index[(edges['src']==trg)&(edges['trg']==src)]

        edges.at[index[0],'capacity'] = capacity
        edges.at[index[0],'balance'] = initial_balance
        edges.at[reverse_index[0],'capacity'] = capacity
        edges.at[reverse_index[0],'balance'] = capacity - initial_balance

        return edges
    
  
  def create_capacity_map(G):
    keys = list(zip(G["src"], G["trg"]))
    vals = [list(item) for item in zip([None]*len(G), G["fee_rate_milli_msat"],G['fee_base_msat'], G["capacity"])]
    capacity_map = dict(zip(keys,vals))
    for index,row in G.iterrows():
      src = row['src']
      trg = row['trg']
      capacity_map[(src,trg)][0] = row['balance']

    return capacity_map


  def create_channel_data(capacity_map,src,trg):
    channel_data={(src,trg):capacity_map[(src,trg)],
                  (trg,src):capacity_map[(trg,src)]}
    return channel_data


  
  def create_sub_network(directed_edges,providers,src,trg,radius):
      """creating capacity map, edges and providers for the local subgraph."""
      edges = initiate_balances(directed_edges)
      edges = set_node_balance(edges,src,trg,channel_id,capacity,initial_balance)
      G = nx.from_pandas_edgelist(edges,source="src",target="trg",
                                  edge_attr=['channel_id','capacity','fee_base_msat','fee_rate_milli_msat','balance'],create_using=nx.DiGraph())
      sub_nodes= get_neighbors(G,src,trg,radius)
      sub_providers = list(set(sub_nodes) & set(providers))
      sub_graph = G.subgraph(sub_nodes)
      sub_edges = nx.to_pandas_edgelist(sub_graph)
      sub_edges = sub_edges.rename(columns={'source': 'src', 'target': 'trg'})
      capacity_map = create_capacity_map(sub_edges)


      return capacity_map, sub_nodes, sub_providers, sub_edges
    
    
  def create_sub_network(directed_edges,providers,src,trg,radius):
      """creating capacity map, edges and providers for the local subgraph."""
      edges = initiate_balances(directed_edges)
      edges = set_node_balance(edges,src,trg,channel_id,capacity,initial_balance)
      G = nx.from_pandas_edgelist(edges,source="src",target="trg",
                                  edge_attr=['channel_id','capacity','fee_base_msat','fee_rate_milli_msat','balance'],create_using=nx.DiGraph())
      sub_nodes= get_neighbors(G,src,trg,radius)
      sub_providers = list(set(sub_nodes) & set(providers))
      sub_graph = G.subgraph(sub_nodes)
      sub_edges = nx.to_pandas_edgelist(sub_graph)
      sub_edges = sub_edges.rename(columns={'source': 'src', 'target': 'trg'})
      capacity_map = create_capacity_map(sub_edges)


      return capacity_map, sub_nodes, sub_providers, sub_edges
