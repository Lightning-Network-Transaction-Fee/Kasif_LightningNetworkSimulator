import numpy as np
import networkx as nx
import pandas as pd
import math
import json
from tqdm import tqdm
import copy








class simulator():
  def __init__(self,
               base_network,                #dynamic not implimented
               merchants,
               count,
               amount,
               epsilon,
               node_variables,
               active_providers):
    self.base_network = base_network
    self.count = count
    self.amout = amount

    self.merchants = merchants #list of merchants
    self.epsilon = epsilon    #ratio of marchant
    self.node_variables = node_variables
    self.active_providers = active_providers

  """  
  update of base graph balances: onChain,offChain, gamma,tx
  also updating the topology dynamc
  """    

  def dynamic_update_base_network(self, new_network): #dynamic
    pass

  


  def get_channel_id(self,src,trg):   #returns the first channel id between these two nodes
    index = self.base_network.index[(self.base_network['src']==src) & (self.base_network['trg']==trg)]
    channel_id = self.base_network.at[index[0],'channel_id']
    return channel_id




  def calculate_weight(self,edge,amount): #assuming edge is a row of dataframe
    if edge["fee_base_msat"] == math.inf:
      return math.inf
    return edge["fee_base_msat"] + edge["fee_rate_milli_msat"]*amount 
    





  def generate_temp_network(self,amount) :    #temp_network is just valid at initialization, doesn't get updated after transactions

    temp_network = copy.deepcopy(self.base_network)  
    temp_network = temp_network.assign(weight=None)
    temp_network.loc[:,'weight'] = temp_network['fee_base_msat'] + temp_network['fee_rate_milli_msat']*amount
    indexes = temp_network.index[temp_network['balance'] <= amount]
    temp_network.loc[indexes,'weight'] = math.inf

    return temp_network





  def generate_depleted_graph(self,temp_network, amount):
    depleted_graph = nx.from_pandas_edgelist(temp_network, source="src", target="trg", edge_attr=['weight'], create_using=nx.DiGraph())
    return depleted_graph




  def update_depleted_graph(self,depleted_graph, path_by_channels, amount):

    for channel_id in path_by_channels :
      index = self.base_network.index[self.base_network["channel_id"] == channel_id]

      src = self.base_network.at[index[0],'src']
      trg = self.base_network.at[index[1],'src']
      src_balance = self.base_network.at[index[0],'balance']
      trg_balance = self.base_network.at[index[1],'balance']

      if src_balance > amount :
        depleted_graph[src][trg]['weight'] = self.calculate_weight(self.base_network.iloc[index[0]],amount)
      elif src_balance <= amount :
        depleted_graph[src][trg]['weight'] = math.inf
      if trg_balance > amount :
        depleted_graph[trg][src]['weight'] = self.calculate_weight(self.base_network.iloc[index[1]],amount)
      elif trg_balance <= amount :
        depleted_graph[trg][src]['weight'] = math.inf
      
    return depleted_graph

  



  

  def update_base_network(self,path, amount):
    for i in range(len(path)-1) :
      src = path[i]
      trg = path[i+1]
      index = self.base_network.index[(self.base_network["src"] == src) & (self.base_network["trg"] == trg)]
      inverse_index = self.base_network.index[(self.base_network["src"] == trg) & (self.base_network["trg"] == src)]      
      self.base_network.at[index[0],'balance'] = self.base_network.at[index[0],'balance'] - amount
      self.base_network.at[inverse_index[0],'balance'] = self.base_network.at[inverse_index[0],'balance'] + amount



    
  def nxpath_to_path_by_channels(self,path_by_nodes):
    path_by_channels = []
    if(len(path_by_nodes) == 0):
        return []
    for i in range(len(path_by_nodes)-1) :
        src = path_by_nodes[i]
        trg = path_by_nodes[i+1]
        channel_id = self.get_channel_id(src,trg)
        path_by_channels.append(channel_id)
    return path_by_channels





  def get_balance(self,src,trg,channel_id):
      index = self.base_network.index[(self.base_network['src']==src) & (self.base_network['trg']==trg) & (self.base_network['channel_id']==channel_id)]
      return self.base_network.at[index[0],'balance']
  



  def get_base_network(self):
    return self.base_network


  
  def channel_exists_in_path(self,channel_id, path_by_channels):
      return channel_id in path_by_channels
  
  


  def get_k(self,src,trg,channel_id, transactions):
    num = 0
    for index, row in transactions.iterrows():
        path = row["path"]  
        for i in range(len(path)-1) :
          if (path[i]==src) & (path[i+1]==trg) :
              num += 1
    return num


  


  def get_total_fee(self,path_by_channels) :
    alpha_bar = 0
    beta_bar = 0
    for channel_id in path_by_channels:
      index = self.base_network.index[(self.base_network['channel_id']==channel_id)]
      alpha_bar += self.base_network.at[index[0],'fee_rate_milli_msat']
      beta_bar += self.base_network.at[index[0],'fee_base_msat']
    return alpha_bar,beta_bar


  
  def get_total_cost(self,path_by_channels,amount) :
      alpha_bar, beta_bar = self.get_total_fee(path_by_channels)
      return alpha_bar*amount + beta_bar


  #off-chain rebalancing


  def get_rebalancing_coefficients(self,rebalancing_type, src, trg, channel_id, rebalancing_amount):
      temp_network = self.generate_temp_network(rebalancing_amount)
      depleted_graph = self.generate_depleted_graph(temp_network,rebalancing_amount)   # weight(edges : balance < amount) = inf
      
      cheapest_rebalancing_path = []
      

      alpha_bar = 0
      beta_bar = 0
      reult_bit = -1

      if rebalancing_type == -1 : #clockwise
          if depleted_graph.get_edge_data(src, trg)['weight'] == math.inf :
            return 0,0,-1

          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-1,rebalancing_amount,trg,src,depleted_graph) 
          if result_bit == -1 :
            return 0,0,-2
            
          if result_bit == 1 :
            path_by_channels = self.nxpath_to_path_by_channels(cheapest_rebalancing_path)
            if path_by_channels == [channel_id] :
              return 0,0,-2
            path_by_channels.insert(0,channel_id)  #convert path to loop
            cheapest_rebalancing_path.insert(0,src)
            alpha_bar,beta_bar = self.get_total_fee(path_by_channels)
            #total_cost = self.get_total_cost(path_by_channels, rebalancing_amount)  
            self.update_base_network(cheapest_rebalancing_path, rebalancing_amount)


      elif rebalancing_type == -2 : #counter-clockwise
          if depleted_graph.get_edge_data(trg, src)['weight'] == math.inf :
            return 0,0,-1

          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-2,rebalancing_amount,src,trg,depleted_graph) 
          if result_bit == -1 :
            return 0,0,-2
          if result_bit == 1 :
            path_by_channels = self.nxpath_to_path_by_channels(cheapest_rebalancing_path)
            if path_by_channels == [channel_id] :
              return 0,0,-2
            path_by_channels.append(channel_id) #convert path to loop
            cheapest_rebalancing_path.append(src)
            alpha_bar,beta_bar = self.get_total_fee(path_by_channels)
            #total_cost = self.get_total_cost(path_by_channels, rebalancing_amount)  
            self.update_base_network(cheapest_rebalancing_path, rebalancing_amount)

   
      
      return alpha_bar,beta_bar,result_bit
      



      

  def get_r(self,src,trg,channel_id,action,bitcoin_transaction_fee = 5000):
      r = np.zeros(6)
      if(action[6]==1):
        print("operating clockwise rebalancing...")
        r[0],r[4],clockwise_result_bit = self.get_rebalancing_coefficients(rebalancing_type=-1, src=src, trg=trg, channel_id = channel_id, rebalancing_amount=action[2])
        if clockwise_result_bit == 1 :
          print("clockwise offchain rebalancing ended successfully!")
        else :
          print("clockwise offchain rebalancing failed !")
      else : 
        r[0],r[4],clockwise_result_bit = 0,0,-1

      if(action[7]==1):
        print("operating counter-clockwise rebalancing...")
        r[1],r[5],counterclockwise_result_bit = self.get_rebalancing_coefficients(rebalancing_type=-2, src=src, trg=trg, channel_id = channel_id, rebalancing_amount=action[3])
        if counterclockwise_result_bit == 1 :
          print("counter clockwise offchain rebalancing ended successfully!")
        else :
          print("counter clockwise offchain rebalancing failed !")
      else :
        r[1],r[5],counterclockwise_result_bit = 0,0,-1


      r[2] = bitcoin_transaction_fee
      return r,clockwise_result_bit,counterclockwise_result_bit


  
  def get_gamma_coeffiecients(self,action,transactions,src,trg,channel_id,simulation_amount):
      k = self.get_k(src,trg,channel_id,transactions)
      tx = simulation_amount*k
      bitcoin_transaction_fee = self.onchain_rebalancing(action[4],action[5],src,trg,channel_id)
      r,clockwise_result_bit,counterclockwise_result_bit = self.get_r(src,trg,channel_id,action,bitcoin_transaction_fee)
      return k,tx,r,clockwise_result_bit,counterclockwise_result_bit


  # onchain rebalancing
  def onchain_rebalancing(self,onchain_rebalancing_flag,onchain_rebalancing_amount,src,trg,channel_id):
    bitcoin_transaction_fee = 0
    if onchain_rebalancing_flag==1 : 
      print("operating onchain rebalancing...")
      bitcoin_transaction_fee = self.operate_rebalancing_on_blockchain(onchain_rebalancing_amount)
      index = self.base_network.index[(self.base_network['src']==src) & (self.base_network['trg']==trg) & (self.base_network['channel_id']==channel_id)]
      inverse_index = self.base_network.index[(self.base_network['src']==trg) & (self.base_network['trg']==src) & (self.base_network['channel_id']==channel_id)] 
      self.base_network.at[index[0],'balance'] += onchain_rebalancing_amount  
      self.base_network.at[index[0],'capacity'] += onchain_rebalancing_amount   
      self.base_network.at[inverse_index[0],'capacity'] += onchain_rebalancing_amount   
      print("onchain rebalancing ended successfully!")    
      

    return bitcoin_transaction_fee




  def operate_rebalancing_on_blockchain(self,onchain_rebalancing_amount):
    bitcoin_transaction_fee = 5000 #CHECK
    return bitcoin_transaction_fee
  



  def get_path_value(self,nxpath,depleted_graph) :
    val = 0 
    for i in range(len(nxpath)-1):
      u,v = nxpath[i],nxpath[i+1]
      weight = depleted_graph.get_edge_data(u, v)['weight']
      if weight == math.inf:
        return math.inf
      else :
        val += weight
    return val
    



  def set_node_fee(self,src,trg,channel_id,action):
      alpha = action[0]
      beta = action[1]
      index = self.base_network.index[(self.base_network['src']==src) & (self.base_network['trg']==trg) & (self.base_network['channel_id']==channel_id)]
      self.base_network.at[index[0],'fee_rate_milli_msat'] = alpha
      self.base_network.at[index[0],'fee_base_msat'] = beta
      



  def run_single_transaction(self,
                             transaction_id,
                             amount,
                             src,trg,
                             depleted_graph):
    
    result_bit = 0
    try:
      path = nx.shortest_path(depleted_graph, source=src, target=trg, weight="weight", method='dijkstra')
    except nx.NetworkXNoPath:
      return None,-1
    val = self.get_path_value(path,depleted_graph)
    if val == math.inf :   
        result_bit = -1
        #print("Transaction ",transaction_id ," Failed")
        return None,result_bit
    
    result_bit = 1
    return path,result_bit  




 
  def run_simulation(self, count, amount, action):
      print("simulating random transactions...")
      

      #Graph Pre-Processing
      temp_network = self.generate_temp_network(amount)
      depleted_graph = self.generate_depleted_graph(temp_network,amount)   # weight(edges : balance < amount) = inf



      #Run Transactions
      transactions = self.generate_transactions(amount, count)
      transactions = transactions.assign(path=None)
      transactions['path'] = transactions['path'].astype('object')
   
      for index, transaction in transactions.iterrows(): 
        path,result_bit = self.run_single_transaction(transaction["transaction_id"],amount,transaction["src"],transaction["trg"],depleted_graph) 

        if result_bit == 1 : #successful transaction
            path_by_channels = self.nxpath_to_path_by_channels(path)
            self.update_base_network(path,amount)
            depleted_graph = self.update_depleted_graph(depleted_graph,path_by_channels,amount)
            transactions.at[index,"result_bit"] = 1
            transactions.at[index,"path"] = path

        elif result_bit == -1 : #failed transaction
            transactions.at[index,"result_bit"] = -1   
            transactions.at[index,"path"] = []

      print("random transactions ended succussfully!")
      return transactions    #contains final result bits  #contains paths



  def sample_providers(self, K):
      provider_records = self.node_variables[self.node_variables["pub_key"].isin(self.active_providers)]
      nodes = list(provider_records["pub_key"])
      probas = list(provider_records["degree"] / provider_records["degree"].sum())
      return np.random.choice(nodes, size=K, replace=True, p=probas)

  def generate_transactions(self,amount_in_satoshi, K, verbose=False):
      nodes = list(self.node_variables['pub_key'])
      src_selected = np.random.choice(nodes, size=K, replace=True)
      if self.epsilon > 0:
          n_prov = int(self.epsilon*K)
          trg_providers = self.sample_providers(n_prov)
          trg_rnd = np.random.choice(nodes, size=K-n_prov, replace=True)
          trg_selected = np.concatenate((trg_providers,trg_rnd))
          np.random.shuffle(trg_selected)
      else:
          trg_selected = np.random.choice(nodes, size=K, replace=True)

      transactions = pd.DataFrame(list(zip(src_selected, trg_selected)), columns=["src","trg"])
      transactions["amount_SAT"] = amount_in_satoshi
      transactions["transaction_id"] = transactions.index
      transactions = transactions[transactions["src"] != transactions["trg"]]
      if verbose:
          print("Number of loop transactions (removed):", K-len(transactions))
          print("Merchant target ratio:", len(transactions[transactions["target"].isin(self.active_providers)]) / len(transactions))
      return transactions[["transaction_id","src","trg","amount_SAT"]]


