import numpy as np
import networkx as nx
import pandas as pd
import math
import json
from tqdm import tqdm
import copy





class simulator():
  def __init__(self,
               capacity_map,                #dynamic not implimented
               merchants,
               count,
               amount,
               epsilon,
               node_variables,
               active_providers):
    self.count = count
    self.amout = amount

    self.merchants = merchants #list of merchants
    self.epsilon = epsilon    #ratio of marchant
    self.node_variables = node_variables
    self.active_providers = active_providers
    self.capacity_map = capacity_map

  """  
  update of base graph balances: onChain,offChain, gamma,tx
  also updating the topology dynamc
  """    

  def dynamic_update_capacity_map(self, new_network): #dynamic
    pass

  




  def calculate_weight(self,edge,amount): #assuming edge is a row of dataframe
    return edge[2] + edge[1]*amount 
    





  def generate_depleted_graph(self, amount):
    depleted_graph = nx.DiGraph()
    for key in capacity_map :
      val = self.capacity_map[key]
      if val[0] > amount :
          depleted_graph.add_edge(key[0],key[1],weight = val[1]*amount + val[2])
    
    return depleted_graph




  def update_depleted_graph(self,depleted_graph, path, amount):

    for i in range(len(path)-1) :  
      src = path[i]
      trg = path[i+1]
      trg_src = self.capacity_map[(trg,src)]
      trg_src_capacity = trg_src[3]
      trg_src_balance = trg_src[0]
      src_trg_balance = trg_src_capacity - trg_src_balance
      
      if src_trg_balance <= amount :
        depleted_graph.remove_edge(src,trg)
      if trg_src_balance > amount :
        depleted_graph.add_edge(trg, src, weight=self.calculate_weight(trg_src,amount))
      
    return depleted_graph

  



  

  def update_capacity_map(self,path, amount):
    for i in range(len(path)-1) :
      src = path[i]
      trg = path[i+1]
      self.capacity_map[(src,trg)][0] = self.capacity_map[(src,trg)][0] - amount
      self.capacity_map[(trg,src)][0] = self.capacity_map[(trg,src)][0] + amount






  def get_balance(self,src,trg,channel_id):
      return self.capacity_map[(src,trg)][0]
  



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
    alpha_bar = 0
    beta_bar = 0
    for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      src_trg = self.capacity_map[(src,trg)]
      alpha_bar += src_trg[1]
      beta_bar += src_trg[2]
    return alpha_bar,beta_bar


  

  #off-chain rebalancing


  def get_rebalancing_coefficients(self,rebalancing_type, src, trg, channel_id, rebalancing_amount):
      depleted_graph = self.generate_depleted_graph(rebalancing_amount)   # weight(edges : balance < amount) = inf
      
      cheapest_rebalancing_path = []
      

      alpha_bar = 0
      beta_bar = 0
      reult_bit = -1

      if rebalancing_type == -1 : #clockwise
          if (not src in depleted_graph.nodes()) or (not trg in depleted_graph.nodes()):
            return 0,0,-1

          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-1,rebalancing_amount,trg,src,depleted_graph) 
          if result_bit == -1 :
            return 0,0,-2
            
          if result_bit == 1 :
            if cheapest_rebalancing_path == [trg,src] :
              return 0,0,-2
            cheapest_rebalancing_path.insert(0,src)
            alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            self.update_capacity_map(cheapest_rebalancing_path, rebalancing_amount)


      elif rebalancing_type == -2 : #counter-clockwise
          if (not src in depleted_graph.nodes()) or (not trg in depleted_graph.nodes()):
            return 0,0,-1

          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-2,rebalancing_amount,src,trg,depleted_graph) 
          if result_bit == -1 :
            return 0,0,-2
          if result_bit == 1 :
            if cheapest_rebalancing_path == [src, trg] :
              return 0,0,-2
            cheapest_rebalancing_path.append(src)
            alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            self.update_capacity_map(cheapest_rebalancing_path, rebalancing_amount)

   
      
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
      src_trg = self.capacity_map[(src,trg)]
      trg_src = self.capacity_map[(trg,src)]
      src_trg[0] += onchain_rebalancing_amount  
      src_trg[3] += onchain_rebalancing_amount   
      trg_src[3] += onchain_rebalancing_amount   
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
      val += weight
    return val
    



  def set_node_fee(self,src,trg,channel_id,action):
      alpha = action[0]
      beta = action[1]
      src_trg = self.capacity_map[(src,trg)]
      src_trg[1] = alpha
      src_trg[2] = beta
      

      



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
    result_bit = 1
    return path,result_bit  




 
  def run_simulation(self, count, amount, action):
      print("simulating random transactions...")
      

      #Graph Pre-Processing
      depleted_graph = self.generate_depleted_graph(amount)   # weight(edges : balance < amount) = inf



      #Run Transactions
      transactions = self.generate_transactions(amount, count)
      transactions = transactions.assign(path=None)
      transactions['path'] = transactions['path'].astype('object')
   
      for index, transaction in transactions.iterrows(): 
        src,trg = transaction["src"],transaction["trg"]
        if (not src in depleted_graph.nodes()) or (not trg in depleted_graph.nodes()):
          path,result_bit = [] , -1
        else : 
          #t1 = time.time()
          path,result_bit = self.run_single_transaction(transaction["transaction_id"],amount,transaction["src"],transaction["trg"],depleted_graph) 
          #t2 = time.time()-t1
          #print("run_single_transaction : ",t2)

        if result_bit == 1 : #successful transaction
            #t1 = time.time()
            self.update_capacity_map(path,amount)
            #t2 = time.time()
            #print("update base network : ", t2-t1)
            depleted_graph = self.update_depleted_graph(depleted_graph,path,amount)
            #t3 = time.time()
            #print("update depleted graph : ", t3-t2)
            transactions.at[index,"result_bit"] = 1
            transactions.at[index,"path"] = path

        elif result_bit == -1 : #failed transaction
            transactions.at[index,"result_bit"] = -1   
            transactions.at[index,"path"] = []
        #print("----------------------------------------------")
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


