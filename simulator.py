import numpy as np
import networkx as nx
import pandas as pd
import math
import json
from tqdm import tqdm
import copy
import sys
import os
import time


#environment has an object of simulator
class simulator():
  def __init__(self,
               src,trg,channel_id,
               channel_data, capacity_map,
               merchants,
               count,
               amount,
               epsilon,
               node_variables,
               active_providers):
    
    self.src = src
    self.trg = trg
    self.channel_id = channel_id
    self.count = count
    self.amount = amount

    self.merchants = merchants #list of merchants
    self.epsilon = epsilon    #ratio of marchant
    self.node_variables = node_variables
    self.active_providers = active_providers
    self.channel_data = channel_data
    self.capacity_map = capacity_map

    self.graph = self.generate_graph(amount)
    self.transactions = self.generate_transactions(amount, count)
 

 

  def calculate_weight(self,edge,amount): #assuming edge is a row of dataframe
    return edge[2] + edge[1]*amount 
    


  def sync_capacity_map(self):
    self.capacity_map[(self.src,self.trg)] = self.channel_data[(self.src,self.trg)]
    self.capacity_map[(self.trg,self.src)] = self.channel_data[(self.trg,self.src)]
    


  def generate_graph(self, amount):
    self.sync_capacity_map()
    graph = nx.DiGraph()
    for key in self.capacity_map :
      val = self.capacity_map[key]
      if val[0] > amount :
          graph.add_edge(key[0],key[1],weight = val[1]*amount + val[2])
    
    return graph




  def update_graph(self, src, trg):
      src_trg = self.channel_data[(src,trg)]
      src_trg_balance = src_trg[0]
      trg_src = self.channel_data[(trg,src)]
      trg_src_balance = trg_src[0]
      
      if (src_trg_balance <= self.amount) and (self.graph.has_edge(src,trg)):
        self.graph.remove_edge(src,trg)
      else : 
        self.graph.add_edge(src, trg, weight = self.calculate_weight(src_trg, self.amount))
      
      if (trg_src_balance <= self.amount) and (self.graph.has_edge(trg,src)) :
        self.graph.remove_edge(trg,src)
      else :   
        self.graph.add_edge(trg, src, weight = self.calculate_weight(trg_src, self.amount))
      
    
  


  def update_channel_data(self, src, trg, transaction_amount):
      self.channel_data[(src,trg)][0] = self.channel_data[(src,trg)][0] - transaction_amount
      self.channel_data[(trg,src)][0] = self.channel_data[(trg,src)][0] + transaction_amount




  def update_network_data(self, path, transaction_amount):
      for i in range(len(path)-1) :
        src = path[i]
        trg = path[i+1]
        if (src == self.src and trg == self.trg) or  (src == self.trg and trg == self.src) :
          self.update_channel_data(src,trg,transaction_amount)
          self.update_graph(src, trg)
          
          
            
      

        

  def onchain_rebalancing(self,onchain_rebalancing_amount,src,trg,channel_id):
    self.channel_data[(src,trg)][0] += onchain_rebalancing_amount  
    self.channel_data[(src,trg)][3] += onchain_rebalancing_amount   
    self.channel_data[(trg,src)][3] += onchain_rebalancing_amount   
    self.update_graph(src, trg)
                




  def get_path_value(self,nxpath,graph) :
    val = 0 
    for i in range(len(nxpath)-1):
      u,v = nxpath[i],nxpath[i+1]
      weight = graph.get_edge_data(u, v)['weight']
      val += weight
    return val
    



  def set_node_fee(self,src,trg,channel_id,action):
      alpha = action[0]
      beta = action[1]
      self.capacity_map[(src,trg)][1] = alpha
      self.capacity_map[(src,trg)][2] = beta
      self.channel_data[(src,trg)][1] = alpha
      self.channel_data[(src,trg)][2] = beta
      


  def run_single_transaction(self,
                             transaction_id,
                             amount,
                             src,trg,
                             graph):
    
    result_bit = 0
    try:
      path = nx.shortest_path(graph, source=src, target=trg, weight="weight", method='dijkstra')
      
    except nx.NetworkXNoPath:
      return None,-1
    val = self.get_path_value(path,graph)
    result_bit = 1
    return path,result_bit  




 
  def run_simulation(self, count, amount, action):
      #print("simulating random transactions...")
      

      #Graph Pre-Processing
      if self.graph.has_edge(self.src, self.trg):
        self.graph[self.src][self.trg]['weight'] = action[0]*amount + action[1]



      #Run Transactions
      #transactions = self.generate_transactions(amount, count)
      transactions = self.transactions
      transactions = transactions.assign(path=None)
      transactions['path'] = transactions['path'].astype('object')
   
      for index, transaction in transactions.iterrows(): 
        src,trg = transaction["src"],transaction["trg"]
        if (not src in self.graph.nodes()) or (not trg in self.graph.nodes()):
          path,result_bit = [] , -1
        else : 
          path,result_bit = self.run_single_transaction(transaction["transaction_id"],amount,transaction["src"],transaction["trg"],self.graph) 
          
        if result_bit == 1 : #successful transaction
            self.update_network_data(path,amount)
            transactions.at[index,"result_bit"] = 1
            transactions.at[index,"path"] = path

        elif result_bit == -1 : #failed transaction
            transactions.at[index,"result_bit"] = -1   
            transactions.at[index,"path"] = []
      # print("random transactions ended succussfully!")
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

 
 
 
 
  """
  getting the statistics
  """

  def get_balance(self,src,trg,channel_id):
      self.sync_capacity_map()
      return self.capacity_map[(src,trg)][0]


  def get_capacity(self,src,trg,channel_id):
      self.sync_capacity_map()
      return self.capacity_map[(src,trg)][3]



  def get_capacity_map(self):
    return self.capacity_map


  def get_k(self,src,trg,channel_id, transactions):
    num = 0
    for index, row in transactions.iterrows():
        path = row["path"]  
        for i in range(len(path)-1) :
          if (path[i]==src) & (path[i+1]==trg) :
              num += 1
    return num



  def get_total_fee(self,path) :
    self.sync_capacity_map()
    alpha_bar = 0
    beta_bar = 0
    for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      src_trg = self.capacity_map[(src,trg)]
      alpha_bar += src_trg[1]
      beta_bar += src_trg[2]
    return alpha_bar,beta_bar



  def find_rebalancing_cycle(self,rebalancing_type, src, trg, channel_id, rebalancing_amount):
      rebalancing_graph = self.generate_graph(rebalancing_amount)  
      cheapest_rebalancing_path = []
      
      alpha_bar = 0
      beta_bar = 0
      reult_bit = -1

      if rebalancing_type == -1 : #clockwise
          if (not src in rebalancing_graph.nodes()) or (not trg in rebalancing_graph.nodes()) or (not rebalancing_graph.has_edge(trg, src)):
            return -1,None,0,0

          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-1,rebalancing_amount,src,trg,rebalancing_graph) 
          if result_bit == -1 :
            return -1,None,0,0
            
          if result_bit == 1 :
            if cheapest_rebalancing_path == [src,trg] :
              return -1,None,0,0
            cheapest_rebalancing_path.append(src)
            alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            

      elif rebalancing_type == -2 : #counter-clockwise
          if (not trg in rebalancing_graph.nodes()) or (not src in rebalancing_graph.nodes()) or (not rebalancing_graph.has_edge(src, trg)):
            return -1,None,0,0

          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-2,rebalancing_amount,trg,src,rebalancing_graph) 
          if result_bit == -1 :
            return -1,None,0,0
          if result_bit == 1 :
            if cheapest_rebalancing_path == [trg, src] :
              return -1,None,0,0
            cheapest_rebalancing_path.insert(0,src)
            alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            
   
      
      return result_bit,cheapest_rebalancing_path,alpha_bar,beta_bar
      



      

  def get_coeffiecients(self,action,transactions,src,trg,channel_id, simulation_amount, onchain_transaction_fee):
        k = self.get_k(src,trg,channel_id,transactions)
        tx = simulation_amount*k
        rebalancing_fee, rebalancing_type  = self.operate_rebalancing(action[2],src,trg,channel_id,onchain_transaction_fee)
        return k,tx, rebalancing_fee, rebalancing_type



  def operate_rebalancing(self,gamma,src,trg,channel_id,onchain_transaction_fee):
    fee = 0
    if gamma == 0 :
      return 0,0  # no rebalancing
    elif gamma > 0 :
      rebalancing_type = -1 #clockwise
      result_bit,cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
        cost = alpha_bar*gamma + beta_bar
        if cost <= onchain_transaction_fee:
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
        else :
          self.onchain_rebalancing(gamma,src,trg,channel_id)
          fee = onchain_transaction_fee
          rebalancing_type = -3 #onchain
      
      else : 
        self.onchain_rebalancing(gamma,src,trg,channel_id)
        fee = onchain_transaction_fee
        rebalancing_type = -3

      return fee, rebalancing_type

    else :
      rebalancing_type = -2 #counter-clockwise
      gamma = gamma*-1    
      result_bit,cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
        cost = alpha_bar*gamma + beta_bar
        if cost <= onchain_transaction_fee:
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
        else :
          self.onchain_rebalancing(gamma,trg,src,channel_id)
          fee = onchain_transaction_fee
          rebalancing_type = -3 #onchain
      
      elif result_bit == -1: 
        self.onchain_rebalancing(gamma,trg,src,channel_id)
        fee = onchain_transaction_fee
        rebalancing_type = -3

      return fee, rebalancing_type
