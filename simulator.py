#invironment has an object of simulator
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

  


  def get_channel_id(self,src,trg):
    index = self.base_network.index[(self.base_network['src']==src) & (self.base_network['trg']==trg)]
    channel_id = self.base_network.iloc[index[0]]['channel_id']
    return channel_id




  def calculate_weight(self,edge,amount): #assuming edge is a row of dataframe
    return edge["fee_base_msat"] + edge["fee_rate_milli_msat"]*amount 
    

  def generate_temp_network(self,amount) :    #temp_network is just valid at initialization, doesn't get updated after transactions
    temp_network = copy.deepcopy(self.base_network)  
    for index, row in temp_network.iterrows():
        if row['balance'] <= amount :
          temp_network.at[index,'fee_base_msat'] = math.inf
    return temp_network


  def generate_depleted_graph(self,temp_network, amount):
    depleted_graph = nx.DiGraph()
    for index, row in temp_network.iterrows():
        depleted_graph.add_edge(row['src'], row['trg'], weight= self.calculate_weight(row,amount))      

    return depleted_graph



  def update_depleted_graph(self,depleted_graph, path_by_channels, amount):
    
    for channel_id in path_by_channels :
      index = self.base_network.index[self.base_network["channel_id"] == channel_id]

      src = self.base_network.at[index[0],'src']
      trg = self.base_network.at[index[1],'src']
      src_balance = self.base_network.at[index[0],'balance']
      trg_balance = self.base_network.at[index[1],'balance']

      if src_balance >= amount :
        depleted_graph[src][trg]['weight'] = self.calculate_weight(self.base_network.iloc[index[0]],amount)
      if trg_balance >= amount :
        depleted_graph[trg][src]['weight'] = self.calculate_weight(self.base_network.iloc[index[1]],amount)
        
    return depleted_graph



  def update_base_network(self,path_by_channels, amount):
    
    for channel_id in path_by_channels:
      index = self.base_network.index[self.base_network["channel_id"] == channel_id]

      self.base_network.at[index[0],'balance'] = self.base_network.at[index[0],'balance'] - amount
      self.base_network.at[index[1],'balance'] = self.base_network.at[index[1],'balance'] + amount



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




  def get_k(self,channel_id, transactions):
    num = 0
    for index, row in transactions.iterrows():
        path_by_channels = row["path"]  
        if(self.channel_exists_in_path(channel_id, path_by_channels)):
            num += 1
    return num



  def get_total_fee(self,path_by_nodes):
    G = self.base_network
    total_alpha = 0
    total_beta = 0
    if(len(path_by_nodes) == 0):
        return 0,0
    for i in range(len(path_by_nodes)-1) :
        src = path_by_nodes[i]
        trg = path_by_nodes[i+1]
        index = G.index[(G['src']==src) & (G['trg']==trg)]
        if(index.empty):
            print("invalid rebalancing query")
            return 0,0
        total_alpha += G.at[index[0],'fee_rate_milli_msat']
        total_beta += G.at[index[0],'fee_base_msat']
    return total_alpha,total_beta




  #off-chain rebalancing


  def get_rebalancing_coefficients(self,rebalancing_type, src,trg,rebalancing_amount):
      temp_network = self.generate_temp_network(rebalancing_amount)
      depleted_graph = self.generate_depleted_graph(temp_network,rebalancing_amount)   # weight(edges : balance < amount) = inf
      cheapest_rebalancing_path = []


      if rebalancing_type == -1 : #clockwise
          cheapest_rebalancing_path,result_bit = self.run_single_transaction(rebalancing_amount,src,trg,depleted_graph) 
          if result_bit == 1 :
            path_by_channels = self.nxpath_to_path_by_channels(cheapest_rebalancing_path)
            self.update_base_network(path_by_channels, rebalancing_amount)
      elif rebalancing_type == -2 : #counter-clockwise
          cheapest_rebalancing_path,result_bit = self.run_single_transaction(rebalancing_amount,trg,src,depleted_graph) 
          if result_bit == 1 :
            path_by_channels = self.nxpath_to_path_by_channels(cheapest_rebalancing_path)
            self.update_base_network(path_by_channels, rebalancing_amount)


      rebalancing_path = cheapest_rebalancing_path


      if not rebalancing_path : # loop doesnt exist
          return 0,0
      else :
          if rebalancing_type == -1 : #clockwise
              rebalancing_path[len(rebalancing_path)-1] = rebalancing_path[len(rebalancing_path)-1]
              rebalancing_path.append(src) #(u -> v) -> u
              rebalancing_path[len(rebalancing_path)-1] = rebalancing_path[len(rebalancing_path)-1]
          elif rebalancing_type == -2 : #counter-clockwise
              rebalancing_path.insert(0,src) # u -> (v -> u)
      alpha_bar,beta_bar = self.get_total_fee(rebalancing_path)   
      
      return alpha_bar,beta_bar
      



      

  def get_r(self,src,trg,gamma_1,gamma_2,bitcoin_transaction_fee = 5000):
      r = np.zeros(6)
      r[0],r[4] = self.get_rebalancing_coefficients(rebalancing_type=-1, src=src,trg=trg,rebalancing_amount=gamma_1)
      r[1],r[5] = self.get_rebalancing_coefficients(rebalancing_type=-2, src=src,trg=trg,rebalancing_amount=gamma_2)
      r[2] = bitcoin_transaction_fee
      return r


  
  def get_gamma_coeffiecients(self,action,transactions,src,trg,simulation_amount):
      channel_id = self.get_channel_id(src,trg)
      k = self.get_k(channel_id,transactions)
      tx = simulation_amount*k
      bitcoin_transaction_fee = self.onchain_rebalancing(action[4],action[5],src,trg)
      r = self.get_r(src,trg,action[2],action[3],bitcoin_transaction_fee)
      return k,tx,r



  # onchain rebalancing
  def onchain_rebalancing(self,onchain_rebalancing_flag,onchain_rebalancing_amount,src,trg):
    if onchain_rebalancing_flag==1 : #CHECK : 1 or True
      bitcoin_transaction_fee = self.operate_rebalancing_on_blockchain(onchain_rebalancing_amount)
      index = self.base_network.index[(self.base_network['src']==src) & (self.base_network['trg']==trg) ]
      self.base_network.at[index[0],'balance'] = self.base_network.at[index[0],'balance'] + amount - bitcoin_transaction_fee  #CHECK

    return bitcoin_transaction_fee


  def operate_rebalancing_on_blockchain(self,onchain_rebalancing_amount):
    bitcoin_transaction_fee = 5000 #CHECK
    return bitcoin_transaction_fee
  

  
  def get_path_value(self,nxpath,depleted_graph):
    val = 0 
    for i in range(len(nxpath)-1):
      u,v = nxpath[i],nxpath[i+1]
      weight = depleted_graph[u][v]['weight']
      if weight == math.inf:
        return math.inf
      else :
        val += weight
    return val

  


  def run_single_transaction(self,
                             transaction_id,
                             amount,
                             src,trg,
                             depleted_graph):
    
    result_bit = 0
    path = nx.shortest_path(depleted_graph, source=src, target=trg, weight="weight", method='dijkstra')
    val = self.get_path_value(path,depleted_graph)
    if val == math.inf :   
        result_bit = -1
        print("Transaction ",transaction_id ," Failed")
        return None,result_bit
    
    result_bit = 1
    return path,result_bit  




 
  def run_simulation(self, count, amount, action):
    

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
            self.update_base_network(path_by_channels,amount)
            depleted_graph = self.update_depleted_graph(depleted_graph,path_by_channels,amount)
            transactions.at[index,"result_bit"] = 1
            transactions.at[index,"path"] = path_by_channels

        elif result_bit == -1 : #failed transaction
            transactions.at[index,"result_bit"] = -1   
            transactions.at[index,"path"] = []


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


